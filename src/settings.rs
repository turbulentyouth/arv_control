use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_stereo_url() -> String {
    "rtsp://192.168.137.2:8555/video_s".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppSettings {
    pub video_source_index: i32,
    pub video_stream_url: String,
    #[serde(default = "default_stereo_url")]
    pub video_stream_stereo_url: String,
    pub rov_ip: String,
    pub keyboard_enabled: bool,
    pub throttle_limit: f32,
    pub photo_save_path: String,
    pub video_save_path: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            video_source_index: 0,
            video_stream_url: "rtsp://192.168.137.2:8554/video".to_string(),
            video_stream_stereo_url: "rtsp://192.168.137.2:8555/video_s".to_string(),
            rov_ip: "192.168.137.2".to_string(),
            keyboard_enabled: true,
            throttle_limit: 20.0,
            photo_save_path: String::new(),
            video_save_path: String::new(),
        }
    }
}

fn settings_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("arv_control")
        .join("settings.json")
}

pub fn load() -> AppSettings {
    let path = settings_path();
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Ok(settings) = serde_json::from_str::<AppSettings>(&content) {
            println!("已加载设置: {}", path.display());
            return settings;
        }
    }
    println!("使用默认设置（未找到 {}）", path.display());
    AppSettings::default()
}

pub fn save(settings: &AppSettings) {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    match serde_json::to_string_pretty(settings) {
        Ok(content) => {
            if let Err(e) = std::fs::write(&path, content) {
                eprintln!("设置保存失败: {}", e);
            }
        }
        Err(e) => eprintln!("设置序列化失败: {}", e),
    }
}
