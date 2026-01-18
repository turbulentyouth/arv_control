use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use ndarray::{Array, ArrayView};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::{Path, PathBuf};

// Configuration from Python
const DET_SCORE_THRESH: f32 = 0.4;
const DET_NMS_IOU_THRESH: f32 = 0.45;

pub(crate) fn resolve_model_path(file_name: &str) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(file_name));
        candidates.push(cwd.join("models").join(file_name));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(file_name));
            candidates.push(exe_dir.join("models").join(file_name));
        }
    }

    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(file_name));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join(file_name));

    for p in candidates {
        if p.is_file() {
            return Ok(p);
        }
    }

    anyhow::bail!("Model file not found: {file_name}");
}

pub struct HandDetector {
    session: Session,
    input_h: usize,
    input_w: usize,
}

impl HandDetector {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("Failed to load detector model: {}", model_path.as_ref().display()))?;
        
        Ok(Self {
            session,
            input_h: 640,
            input_w: 640,
        })
    }

    pub fn infer(&mut self, img: &RgbImage) -> Result<Vec<[f32; 5]>> {
        let (im_w, im_h) = img.dimensions();
        let scale = (self.input_w as f32 / im_w as f32).min(self.input_h as f32 / im_h as f32);
        let new_w = (im_w as f32 * scale).round().max(1.0) as u32;
        let new_h = (im_h as f32 * scale).round().max(1.0) as u32;
        let pad_w = (self.input_w as f32 - new_w as f32) / 2.0;
        let pad_h = (self.input_h as f32 - new_h as f32) / 2.0;

        let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);
        
        let pad_val = 114.0f32 / 255.0;
        let mut input_tensor = Array::<f32, _>::from_elem((1, 3, self.input_h, self.input_w), pad_val);
        let pad_left = pad_w.floor() as u32;
        let pad_top = pad_h.floor() as u32;
        let pad_x = pad_left as f32;
        let pad_y = pad_top as f32;

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = resized.get_pixel(x, y);
                let dx = x + pad_left;
                let dy = y + pad_top;
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    input_tensor[[0, c, dy as usize, dx as usize]] = val;
                }
            }
        }
        
        let shape = vec![1, 3, self.input_h, self.input_w];
        let (data, _offset) = input_tensor.into_raw_vec_and_offset();
        let input_value = ort::value::Value::from_array((shape, data))?;
        let outputs = self.session.run(ort::inputs![input_value])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?; // (Shape, &[f32])

        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let raw = Self::parse_yolo_preds(&shape_usize, data)?;
        let normalized_coords = Self::coords_are_normalized(&raw);

        let mut dets: Vec<[f32; 5]> = raw
            .into_iter()
            .filter_map(|(x, y, w, h, score)| {
                if score < DET_SCORE_THRESH {
                    return None;
                }
                let (mut x, mut y, mut w, mut h) = (x, y, w, h);
                if normalized_coords {
                    x *= self.input_w as f32;
                    w *= self.input_w as f32;
                    y *= self.input_h as f32;
                    h *= self.input_h as f32;
                }
                let x1 = x - w / 2.0;
                let y1 = y - h / 2.0;
                let x2 = x + w / 2.0;
                let y2 = y + h / 2.0;

                let x1 = (x1 - pad_x) / scale;
                let y1 = (y1 - pad_y) / scale;
                let x2 = (x2 - pad_x) / scale;
                let y2 = (y2 - pad_y) / scale;

                let x1 = x1.clamp(0.0, im_w as f32);
                let y1 = y1.clamp(0.0, im_h as f32);
                let x2 = x2.clamp(0.0, im_w as f32);
                let y2 = y2.clamp(0.0, im_h as f32);

                if x2 <= x1 || y2 <= y1 {
                    return None;
                }
                Some([x1, y1, x2, y2, score])
            })
            .collect();

        dets.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));
        let dets = Self::nms(dets, DET_NMS_IOU_THRESH);
        Ok(dets)
    }

    fn parse_yolo_preds(shape: &[usize], data: &[f32]) -> Result<Vec<(f32, f32, f32, f32, f32)>> {
        let mut out = Vec::new();

        let (num, dim, layout) = match shape {
            [1, a, b] => {
                if *b >= 5 && *b <= 512 {
                    (*a, *b, 0usize)
                } else if *a >= 5 && *a <= 512 {
                    (*b, *a, 1usize)
                } else {
                    (*a, *b, 0usize)
                }
            }
            [a, b] => {
                if *b >= 5 && *b <= 512 {
                    (*a, *b, 0usize)
                } else if *a >= 5 && *a <= 512 {
                    (*b, *a, 1usize)
                } else {
                    (*a, *b, 0usize)
                }
            }
            _ => anyhow::bail!("Unexpected detector output shape: {shape:?}"),
        };

        match layout {
            0 => {
                for i in 0..num {
                    let base = i * dim;
                    if base + 4 >= data.len() {
                        break;
                    }
                    let x = data[base];
                    let y = data[base + 1];
                    let w = data[base + 2];
                    let h = data[base + 3];
                    let score = if dim <= 6 {
                        data[base + 4]
                    } else {
                        let obj = data[base + 4];
                        let mut best = 0.0f32;
                        for j in 5..dim {
                            best = best.max(data[base + j]);
                        }
                        obj * best
                    };
                    out.push((x, y, w, h, score));
                }
            }
            _ => {
                for i in 0..num {
                    let x = data.get(i).copied().unwrap_or(0.0);
                    let y = data.get(num + i).copied().unwrap_or(0.0);
                    let w = data.get(2 * num + i).copied().unwrap_or(0.0);
                    let h = data.get(3 * num + i).copied().unwrap_or(0.0);
                    let score = if dim <= 6 {
                        data.get(4 * num + i).copied().unwrap_or(0.0)
                    } else {
                        let obj = data.get(4 * num + i).copied().unwrap_or(0.0);
                        let mut best = 0.0f32;
                        for j in 5..dim {
                            best = best.max(data.get(j * num + i).copied().unwrap_or(0.0));
                        }
                        obj * best
                    };
                    out.push((x, y, w, h, score));
                }
            }
        }

        Ok(out)
    }

    fn coords_are_normalized(raw: &[(f32, f32, f32, f32, f32)]) -> bool {
        let mut max_v = 0.0f32;
        for (i, (x, y, w, h, _)) in raw.iter().enumerate() {
            if i >= 100 {
                break;
            }
            max_v = max_v.max(*x).max(*y).max(*w).max(*h);
        }
        max_v <= 2.0
    }

    fn nms(mut dets: Vec<[f32; 5]>, iou_thresh: f32) -> Vec<[f32; 5]> {
        let mut keep: Vec<[f32; 5]> = Vec::new();
        while let Some(det) = dets.first().copied() {
            keep.push(det);
            dets.remove(0);
            dets.retain(|d| Self::iou(det, *d) < iou_thresh);
            if keep.len() >= 20 {
                break;
            }
        }
        keep
    }

    fn iou(a: [f32; 5], b: [f32; 5]) -> f32 {
        let ax1 = a[0];
        let ay1 = a[1];
        let ax2 = a[2];
        let ay2 = a[3];
        let bx1 = b[0];
        let by1 = b[1];
        let bx2 = b[2];
        let by2 = b[3];

        let inter_x1 = ax1.max(bx1);
        let inter_y1 = ay1.max(by1);
        let inter_x2 = ax2.min(bx2);
        let inter_y2 = ay2.min(by2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let inter = inter_w * inter_h;
        let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
        let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
        let union = area_a + area_b - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

pub struct HandPose {
    session: Session,
    input_h: usize,
    input_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl HandPose {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("Failed to load pose model: {}", model_path.as_ref().display()))?;
            
        Ok(Self {
            session,
            input_h: 256,
            input_w: 256,
            mean: [123.675, 116.28, 103.53], // RGB mean
            std: [58.395, 57.12, 57.375],
        })
    }

    pub fn infer(&mut self, img: &RgbImage, box_: [f32; 4]) -> Result<Vec<(f32, f32)>> {
        let (x1, y1, x2, y2) = (box_[0], box_[1], box_[2], box_[3]);
        let center_x = (x1 + x2) / 2.0;
        let center_y = (y1 + y2) / 2.0;
        let width = x2 - x1;
        let height = y2 - y1;
        let max_side = width.max(height) * 1.4; // 1.4 scale from python
        
        let im_w = img.width() as f32;
        let im_h = img.height() as f32;
        
        let crop_x1 = (center_x - max_side / 2.0).max(0.0) as u32;
        let crop_y1 = (center_y - max_side / 2.0).max(0.0) as u32;
        let crop_x2 = (center_x + max_side / 2.0).min(im_w) as u32;
        let crop_y2 = (center_y + max_side / 2.0).min(im_h) as u32;
        
        if crop_x2 <= crop_x1 || crop_y2 <= crop_y1 {
            return Ok(vec![]);
        }
        
        let crop = image::imageops::crop_imm(img, crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1).to_image();
        
        // Preprocess with padding
        let (crop_w, crop_h) = crop.dimensions();
        let scale = (self.input_h as f32 / crop_h as f32).min(self.input_w as f32 / crop_w as f32);
        let new_w = (crop_w as f32 * scale) as u32;
        let new_h = (crop_h as f32 * scale) as u32;
        
        let resized = image::imageops::resize(&crop, new_w, new_h, FilterType::Triangle);
        
        let mut input_tensor = Array::<f32, _>::zeros((1, 3, self.input_h, self.input_w));
        for c in 0..3 {
            let pad = (0.0 - self.mean[c]) / self.std[c];
            for y in 0..self.input_h {
                for x in 0..self.input_w {
                    input_tensor[[0, c, y, x]] = pad;
                }
            }
        }
        
        let pad_h = (self.input_h as u32 - new_h) / 2;
        let pad_w = (self.input_w as u32 - new_w) / 2;
        
        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = resized.get_pixel(x, y);
                for c in 0..3 {
                    // Python uses cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB) so it expects RGB
                    // image crate is RGB.
                    // Mean/Std in python: mean=[123.675, 116.28, 103.53] (RGB order?)
                    // Python code: `img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)`
                    // `blob = (img_rgb - self.mean) / self.std`
                    // So yes, RGB.
                    let val = pixel[c] as f32;
                    input_tensor[[0, c, (y + pad_h) as usize, (x + pad_w) as usize]] = (val - self.mean[c]) / self.std[c];
                }
            }
        }
        
        let shape = vec![1, 3, self.input_h, self.input_w];
        let (data, _offset) = input_tensor.into_raw_vec_and_offset();
        let input_value = ort::value::Value::from_array((shape, data))?;
        let outputs = self.session.run(ort::inputs![input_value])?;
        // simcc_x: [1, 21, 256*2], simcc_y: [1, 21, 256*2] usually?
        
        let (shape_x, data_x) = outputs[0].try_extract_tensor::<f32>()?;
        let (shape_y, data_y) = outputs[1].try_extract_tensor::<f32>()?;
        
        let shape_x_usize: Vec<usize> = shape_x.iter().map(|&x| x as usize).collect();
        let shape_y_usize: Vec<usize> = shape_y.iter().map(|&x| x as usize).collect();
        
        let simcc_x = ArrayView::from_shape(shape_x_usize, data_x)?;
        let simcc_y = ArrayView::from_shape(shape_y_usize, data_y)?;
        
        let mut kpts = Vec::new();
        let num_kpts = simcc_x.shape()[1]; // 21
        let simcc_w = simcc_x.shape()[2];
        let simcc_h = simcc_y.shape()[2];
        
        for i in 0..num_kpts {
            // argmax
            let mut max_x_idx = 0;
            let mut max_x_val = -f32::INFINITY;
            for j in 0..simcc_w {
                let val = simcc_x[[0, i, j]];
                if val > max_x_val {
                    max_x_val = val;
                    max_x_idx = j;
                }
            }
            
            let mut max_y_idx = 0;
            let mut max_y_val = -f32::INFINITY;
            for j in 0..simcc_h {
                let val = simcc_y[[0, i, j]];
                if val > max_y_val {
                    max_y_val = val;
                    max_y_idx = j;
                }
            }
            
            let x_loc = max_x_idx as f32 / (simcc_w as f32 / self.input_w as f32);
            let y_loc = max_y_idx as f32 / (simcc_h as f32 / self.input_h as f32);
            
            // Restore coordinates
            let x_no_pad = x_loc - pad_w as f32;
            let y_no_pad = y_loc - pad_h as f32;
            
            let x_orig_crop = x_no_pad / scale;
            let y_orig_crop = y_no_pad / scale;
            
            let final_x = x_orig_crop + crop_x1 as f32;
            let final_y = y_orig_crop + crop_y1 as f32;
            
            kpts.push((final_x, final_y));
        }
        
        Ok(kpts)
    }
}

pub struct GestureLogic;

impl GestureLogic {
    pub fn new() -> Self { Self }

    fn get_dist(&self, p1: (f32, f32), p2: (f32, f32)) -> f32 {
        ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
    }

    fn get_angle(&self, p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
        let ba = (p1.0 - p2.0, p1.1 - p2.1);
        let bc = (p3.0 - p2.0, p3.1 - p2.1);
        
        let dot = ba.0 * bc.0 + ba.1 * bc.1;
        let len_ba = (ba.0.powi(2) + ba.1.powi(2)).sqrt();
        let len_bc = (bc.0.powi(2) + bc.1.powi(2)).sqrt();
        
        let cosine = dot / (len_ba * len_bc + 1e-6);
        cosine.clamp(-1.0, 1.0).acos().to_degrees()
    }

    pub fn analyze(&self, kpts: &[(f32, f32)]) -> String {
        if kpts.len() < 21 { return "UNKNOWN".to_string(); }
        
        let wrist = kpts[0];
        let palm_len = self.get_dist(wrist, kpts[9]);
        
        // Thumb
        // let thumb_angle = self.get_angle(kpts[4], kpts[2], kpts[0]);
        let thumb_open = self.get_dist(kpts[4], kpts[17]) > palm_len * 0.8;
        
        // Fingers angle
        let index_angle = self.get_angle(kpts[8], kpts[6], kpts[5]);
        let middle_angle = self.get_angle(kpts[12], kpts[10], kpts[9]);
        let ring_angle = self.get_angle(kpts[16], kpts[14], kpts[13]);
        let pinky_angle = self.get_angle(kpts[20], kpts[18], kpts[17]);
        
        let mut fingers_open = [false; 5];
        fingers_open[0] = thumb_open;
        fingers_open[1] = index_angle > 140.0 || self.get_dist(kpts[8], wrist) > palm_len * 1.5;
        fingers_open[2] = middle_angle > 140.0 || self.get_dist(kpts[12], wrist) > palm_len * 1.5;
        fingers_open[3] = ring_angle > 140.0 || self.get_dist(kpts[16], wrist) > palm_len * 1.5;
        fingers_open[4] = pinky_angle > 140.0 || self.get_dist(kpts[20], wrist) > palm_len * 1.4;

        // Logic Mapping
        // Fingers all open (0-4 open)
        if fingers_open[1] && fingers_open[2] && fingers_open[3] && fingers_open[4] {
            // Middle finger direction?
            // "中指向上为上浮" -> Middle finger tip vs Wrist
            // Screen coordinates: Y is down. So Up is y_tip < y_wrist.
            
            let dx = kpts[12].0 - wrist.0;
            let dy = kpts[12].1 - wrist.1;
            
            if dy.abs() > dx.abs() {
                if dy < 0.0 { return "UP".to_string(); } // Ascend
                else { return "DOWN".to_string(); } // Descend
            } else {
                if dx < 0.0 { return "LEFT".to_string(); } // Move Left
                else { return "RIGHT".to_string(); } // Move Right
            }
        }
        
        // V-sign (Index + Middle open, others closed)
        let ring_tip_dist = self.get_dist(kpts[16], wrist);
        let pinky_tip_dist = self.get_dist(kpts[20], wrist);
        let ring_closed = ring_angle < 150.0 && ring_tip_dist < palm_len * 1.35;
        let pinky_closed = pinky_angle < 150.0 && pinky_tip_dist < palm_len * 1.25;
        let v_spread = self.get_dist(kpts[8], kpts[12]);

        if fingers_open[1] && fingers_open[2] && ring_closed && pinky_closed && v_spread > palm_len * 0.35 {
             // Direction based on Index+Middle vector or just average
             let mid_tip_x = (kpts[8].0 + kpts[12].0) / 2.0;
             let mid_tip_y = (kpts[8].1 + kpts[12].1) / 2.0;
             let dx = mid_tip_x - wrist.0;
             let dy = mid_tip_y - wrist.1;
             
             if dy.abs() > dx.abs() {
                if dy < 0.0 { return "FORWARD".to_string(); } // Forward (User says UP -> Forward)
                else { return "BACKWARD".to_string(); } // Backward
            } else {
                if dx < 0.0 { return "TURN_LEFT".to_string(); } // Turn Left
                else { return "TURN_RIGHT".to_string(); } // Turn Right
            }
        }

        "UNKNOWN".to_string()
    }
}
