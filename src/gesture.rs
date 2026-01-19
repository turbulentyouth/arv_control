use anyhow::Result;
use pyo3::prelude::*;
use std::{env, path::PathBuf};

pub struct PyMediaPipe {
    func: Py<PyAny>,
}

impl PyMediaPipe {
    pub fn new() -> Result<Self> {
        let code = r#"
import numpy as np
_init_err = None
try:
    import mediapipe as mp
except Exception as e:
    _init_err = str(e)
    mp = None

def _create_hands(video=True):
    if mp is None:
        raise RuntimeError(f'Mediapipe import failed: {_init_err}')
    return mp.solutions.hands.Hands(static_image_mode=not video, max_num_hands=1, model_complexity=1, min_detection_confidence=0.15, min_tracking_confidence=0.30)

_hands = _create_hands(video=True)

def process_frame(width, height, frame_bytes):
    global _hands
    img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
    img = np.clip(img.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    results = _hands.process(img)
    bbox = None
    kpts = []
    if results.multi_hand_landmarks:
        best_area = -1.0
        best = None
        for lmks in results.multi_hand_landmarks:
            xs = [lm.x for lm in lmks.landmark]
            ys = [lm.y for lm in lmks.landmark]
            min_x = min(xs) * width
            min_y = min(ys) * height
            max_x = max(xs) * width
            max_y = max(ys) * height
            area = (max_x - min_x) * (max_y - min_y)
            if area > best_area:
                best_area = area
                best = lmks
                bbox = [min_x, min_y, max_x, max_y]
        if best is not None:
            for lm in best.landmark:
                kpts.append((lm.x * width, lm.y * height))
    return bbox, kpts
"#;
        use std::ffi::CString;
        let filename = CString::new("mediapipe_hands_embed.py")?;
        let modname = CString::new("mediapipe_hands")?;
        let code_c = CString::new(code)?;
        if env::var_os("PYTHONHOME").is_none() {
            if let Some(prefix) = env::var_os("CONDA_PREFIX") {
                env::set_var("PYTHONHOME", &prefix);
                let mut lib = PathBuf::from(&prefix);
                lib.push("Lib");
                let mut site = lib.clone();
                site.push("site-packages");
                let py_path = format!("{};{}", lib.display(), site.display());
                env::set_var("PYTHONPATH", py_path);
            }
        }
        Python::attach(|py| -> PyResult<Self> {
            let module = PyModule::from_code(py, &code_c, &filename, &modname)?;
            let func: Py<PyAny> = module.getattr("process_frame")?.unbind();
            Ok(Self { func })
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn infer(&self, width: u32, height: u32, bytes: &[u8]) -> Result<(Option<[f32; 4]>, Vec<(f32, f32)>)> {
        Python::attach(|py| -> PyResult<(Option<[f32; 4]>, Vec<(f32, f32)>)> {
            let ret = self.func.call1(py, (width, height, bytes))?;
            let (bbox_rs, kpts_rs): (Option<Vec<f32>>, Vec<(f32, f32)>) = ret.extract(py)?;
            let bbox = bbox_rs.and_then(|seq| if seq.len() >= 4 { Some([seq[0], seq[1], seq[2], seq[3]]) } else { None });

            Ok((bbox, kpts_rs))
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))
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
        
        // Index pointing: Index open, others closed
        let ring_tip_dist = self.get_dist(kpts[16], wrist);
        let pinky_tip_dist = self.get_dist(kpts[20], wrist);
        let middle_tip_dist = self.get_dist(kpts[12], wrist);
        let ring_closed = ring_angle < 150.0 && ring_tip_dist < palm_len * 1.35;
        let pinky_closed = pinky_angle < 150.0 && pinky_tip_dist < palm_len * 1.25;
        let middle_closed = middle_angle < 150.0 && middle_tip_dist < palm_len * 1.35;

        if fingers_open[1] && middle_closed && ring_closed && pinky_closed {
            let dx = kpts[8].0 - wrist.0;
            let dy = kpts[8].1 - wrist.1;
            if dy.abs() > dx.abs() {
                if dy < 0.0 { return "FORWARD".to_string(); }
                else { return "BACKWARD".to_string(); }
            } else {
                if dx < 0.0 { return "TURN_LEFT".to_string(); }
                else { return "TURN_RIGHT".to_string(); }
            }
        }

        "UNKNOWN".to_string()
    }
}
