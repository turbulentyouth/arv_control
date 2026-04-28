use std::sync::mpsc;

pub enum CaptureCommand {
    TakePhoto {
        save_path: String,
    },
    StartRecording {
        save_path: String,
    },
    StopRecording,
    /// 运行时切换视频源（无需重启应用）
    ChangeSource {
        source: VideoSource,
    },
}

#[derive(Clone)]
pub enum VideoSource {
    Rtsp { url: String },
    Camera { device: String },
}

pub fn channel() -> (mpsc::Sender<CaptureCommand>, mpsc::Receiver<CaptureCommand>) {
    mpsc::channel()
}
