import cv2
import numpy as np
import onnxruntime as ort
import math

# ================= 配置区域 =================
DET_MODEL_PATH = "rtmdet_hand.onnx"
POSE_MODEL_PATH = "rtmpose_hand.onnx"

# 阈值设置 (降低阈值以适应小目标)
DET_SCORE_THRESH = 0.4   # 检测阈值降低，抓取更远/更小的手
POSE_SCORE_THRESH = 0.3  # 关键点置信度

# 图像增强
GAMMA_VALUE = 1.5        # 稍微降低Gamma，避免过曝导致边缘模糊

# 骨架定义
SKELETON = [
    (0,1), (1,2), (2,3), (3,4),       # 拇指
    (0,5), (5,6), (6,7), (7,8),       # 食指
    (0,9), (9,10), (10,11), (11,12),  # 中指
    (0,13), (13,14), (14,15), (15,16),# 无名指
    (0,17), (17,18), (18,19), (19,20) # 小指
]
# ===========================================

def adjust_gamma(image, gamma=1.0):
    if gamma == 1.0: return image
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

class HandDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.h, self.w = self.input_shape[2], self.input_shape[3]
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    def preprocess(self, img):
        im_h, im_w = img.shape[:2]
        scale = min(self.h / im_h, self.w / im_w)
        new_h, new_w = int(im_h * scale), int(im_w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        pad_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pad_img[:new_h, :new_w] = img_resized
        blob = (pad_img - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        return blob, scale

    def infer(self, img):
        blob, scale = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = outputs[0][0]
        valid_mask = dets[:, 4] > DET_SCORE_THRESH
        valid_dets = dets[valid_mask]
        if len(valid_dets) > 0:
            valid_dets[:, :4] /= scale
        return valid_dets

class HandPose:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_h, self.input_w = 256, 256
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def preprocess_with_padding(self, img_crop):
        """
        【关键优化】带黑边的缩放
        解决“横着的手”被压扁的问题
        """
        h, w = img_crop.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img_crop, (new_w, new_h))
        
        # 创建全黑画布
        padded_img = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        
        # 居中粘贴
        start_h = (self.input_h - new_h) // 2
        start_w = (self.input_w - new_w) // 2
        padded_img[start_h:start_h+new_h, start_w:start_w+new_w] = img_resized
        
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        blob = (img_rgb - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        
        return blob, scale, start_h, start_w

    def infer(self, img, box):
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = img.shape[:2]
        
        # 【优化】扩大裁剪范围 1.25 -> 1.4
        # 横着的手容易手腕出框，扩大范围能包含更多信息
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        width, height = x2-x1, y2-y1
        # 选取长宽中较大的边作为基准，保证尽量切出正方形区域
        max_side = max(width, height) * 1.4
        
        x1 = max(0, int(center_x - max_side/2))
        y1 = max(0, int(center_y - max_side/2))
        x2 = min(w_img, int(center_x + max_side/2))
        y2 = min(h_img, int(center_y + max_side/2))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        # 使用带Padding的预处理
        blob, scale, pad_h, pad_w = self.preprocess_with_padding(crop)
        
        outputs = self.session.run(None, {self.input_name: blob})
        simcc_x, simcc_y = outputs[0], outputs[1]
        
        # 解码
        locs = self.decode_simcc(simcc_x, simcc_y)
        
        # 坐标还原 (考虑 Padding 的偏移)
        final_kpts = []
        for x, y in locs:
            # 1. 减去 Padding
            x_no_pad = x - pad_w
            y_no_pad = y - pad_h
            # 2. 除以缩放系数
            x_orig_crop = x_no_pad / scale
            y_orig_crop = y_no_pad / scale
            # 3. 加上原图偏移
            final_kpts.append((int(x_orig_crop + x1), int(y_orig_crop + y1)))
            
        return final_kpts

    def decode_simcc(self, simcc_x, simcc_y):
        x_locs = np.argmax(simcc_x, axis=2)[0]
        y_locs = np.argmax(simcc_y, axis=2)[0]
        simcc_w = simcc_x.shape[2]
        simcc_h = simcc_y.shape[2]
        locs = np.stack([x_locs / (simcc_w / self.input_w), 
                         y_locs / (simcc_h / self.input_h)], axis=1)
        return locs

class GestureLogic:
    def get_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_angle(self, p1, p2, p3):
        # 计算三点夹角 (p1-p2-p3), p2是顶点
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def analyze(self, kpts):
        """
        【优化】基于角度和相对长度的分类，解决旋转失效问题
        """
        wrist = kpts[0]
        # 手掌参考长度 (手腕到中指根部)
        palm_len = self.get_dist(wrist, kpts[9])
        
        # 1. 判断五根手指的弯曲状态 (基于关节夹角，而不是坐标)
        # 角度 > 150度 算伸直
        # 拇指 (特殊，看指尖-关节-手腕)
        thumb_angle = self.get_angle(kpts[4], kpts[2], kpts[0]) 
        # 其他四指 (看指尖-远节-近节)
        index_angle = self.get_angle(kpts[8], kpts[6], kpts[5])
        middle_angle = self.get_angle(kpts[12], kpts[10], kpts[9])
        ring_angle = self.get_angle(kpts[16], kpts[14], kpts[13])
        pinky_angle = self.get_angle(kpts[20], kpts[18], kpts[17])

        # 宽松阈值 (有些角度受拍摄影响)
        # 辅助判断：指尖到手腕的距离 vs 手指收缩时的距离
        fingers_open = [False] * 5
        
        # 拇指判断 (比较难，结合距离)
        fingers_open[0] = self.get_dist(kpts[4], kpts[17]) > palm_len * 0.8
        
        # 食指到小指 (结合角度和距离)
        fingers_open[1] = index_angle > 140 or self.get_dist(kpts[8], wrist) > palm_len * 1.5
        fingers_open[2] = middle_angle > 140 or self.get_dist(kpts[12], wrist) > palm_len * 1.5
        fingers_open[3] = ring_angle > 140 or self.get_dist(kpts[16], wrist) > palm_len * 1.5
        fingers_open[4] = pinky_angle > 140 or self.get_dist(kpts[20], wrist) > palm_len * 1.4

        # 捏合检测
        is_pinch = self.get_dist(kpts[4], kpts[8]) < palm_len * 0.3

        # === 状态机 ===
        if all(fingers_open[1:]): # 食指到小指全开
            return "STOP"
        
        if not any(fingers_open[1:]): # 食指到小指全闭
            return "FIST"
        
        if is_pinch:
            return "PINCH"
            
        if fingers_open[1] and not any(fingers_open[2:]): # 只有食指开
            # 计算指向角度
            dx = kpts[8][0] - wrist[0]
            dy = kpts[8][1] - wrist[1]
            deg = math.degrees(math.atan2(dy, dx))
            return f"POINT {int(deg)}"

        return "UNKNOWN"

def main():
    print("Loading optimized models...")
    detector = HandDetector(DET_MODEL_PATH)
    pose = HandPose(POSE_MODEL_PATH)
    logic = GestureLogic()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 增强
        frame_enhanced = adjust_gamma(frame, GAMMA_VALUE)
        
        # 2. 检测
        dets = detector.infer(frame_enhanced)
        
        # 找最大的手
        best_box = None
        max_area = 0
        for det in dets:
            x1, y1, x2, y2, score = det
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area
                best_box = det[:4]
            cv2.rectangle(frame_enhanced, (int(x1), int(y1)), (int(x2), int(y2)), (100,100,100), 1)

        if best_box is not None:
            # 3. 姿态 (带 Padding)
            kpts = pose.infer(frame_enhanced, best_box)
            
            if kpts:
                # 无论是否识别出手势，都先画出骨架，方便调试！！
                for p1, p2 in SKELETON:
                    cv2.line(frame_enhanced, kpts[p1], kpts[p2], (0,255,255), 2)
                for p in kpts:
                    cv2.circle(frame_enhanced, p, 3, (0,0,255), -1)
                
                # 4. 逻辑判断
                gesture = logic.analyze(kpts)
                
                # 绘制结果
                bx1, by1 = int(best_box[0]), int(best_box[1])
                color = (0, 255, 0) if gesture != "UNKNOWN" else (0, 0, 255)
                
                cv2.putText(frame_enhanced, f"Gesture: {gesture}", (bx1, by1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 绘制指向箭头
                if "POINT" in gesture:
                    cv2.arrowedLine(frame_enhanced, kpts[0], kpts[8], (255,0,0), 4)

        cv2.imshow("Optimized Hand Control", frame_enhanced)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()