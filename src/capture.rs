use std::sync::mpsc;

pub enum CaptureCommand {
    TakePhoto { save_path: String },
    StartRecording { save_path: String },
    StopRecording,
}

pub enum VideoSource {
    Rtsp { url: String },
    Camera { device: String },
}

pub fn channel() -> (mpsc::Sender<CaptureCommand>, mpsc::Receiver<CaptureCommand>) {
    mpsc::channel()
}
