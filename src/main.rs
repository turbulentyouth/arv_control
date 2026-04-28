mod input;
mod video;
mod mav;
mod capture;
mod settings;

use std::thread;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::mem;
use crate::input::InputState;
use crate::capture::{CaptureCommand, VideoSource};
use crate::settings::AppSettings;
use slint::{ComponentHandle, SharedPixelBuffer};

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    env_logger::init();

    let ui = AppWindow::new()?;

    ui.set_version_text(env!("CARGO_PKG_VERSION").into());

    // ── 加载持久化设置 ──
    let saved = settings::load();

    // 将保存的设置应用到 UI
    ui.set_video_source_index(saved.video_source_index);
    ui.set_video_stream_url(saved.video_stream_url.clone().into());
    ui.set_rov_ip(saved.rov_ip.clone().into());
    ui.set_keyboard_enabled(saved.keyboard_enabled);
    ui.set_throttle_limit_percent(saved.throttle_limit);

    // 路径：若设置中有值则用设置值，否则使用系统默认目录
    let photo_path = if saved.photo_save_path.is_empty() {
        dirs::picture_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("arv_control")
            .to_string_lossy()
            .to_string()
    } else {
        saved.photo_save_path.clone()
    };
    let video_path = if saved.video_save_path.is_empty() {
        dirs::video_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("arv_control")
            .to_string_lossy()
            .to_string()
    } else {
        saved.video_save_path.clone()
    };
    ui.set_photo_save_path(photo_path.into());
    ui.set_video_save_path(video_path.into());

    // ── 共享帧缓存（拉模型）──
    let latest_frame = Arc::new(video::LatestFrame::new());

    let (capture_tx, capture_rx) = capture::channel();

    // 从（已更新的）UI 读取初始视频源
    let source_index = ui.get_video_source_index();
    let stream_url = ui.get_video_stream_url().to_string();
    let initial_source = match source_index {
        1 => VideoSource::Camera { device: stream_url },
        _ => VideoSource::Rtsp { url: stream_url },
    };

    // ── 视频播放线程（自动重连，支持运行时切换源）──
    let ui_handle_video = ui.as_weak();
    let video_latest = latest_frame.clone();
    thread::spawn(move || {
        if let Err(e) = video::run_video_player(ui_handle_video, capture_rx, initial_source, video_latest) {
            eprintln!("视频播放器致命错误: {}", e);
        }
    });

    // ── Input State ──
    let input_state = Arc::new(Mutex::new(InputState::default()));

    // ── MAVLink 连接状态（None 表示断开，后台线程自动重连）──
    let mav_conn_str = "udpin:0.0.0.0:14550";
    println!("MAVLink 监听地址: {}", mav_conn_str);
    let mav_state: mav::MavState = Arc::new(Mutex::new(mav::try_connect(mav_conn_str)));

    // 后台自动重连线程
    {
        let mav_state_conn = mav_state.clone();
        thread::spawn(move || {
            mav::run_mavlink_connector(mav_conn_str.to_string(), mav_state_conn);
        });
    }

    // MAVLink 接收线程
    {
        let mav_state_recv = mav_state.clone();
        let ui_handle_mav = ui.as_weak();
        thread::spawn(move || {
            if let Err(e) = mav::run_mavlink_recv(mav_state_recv, ui_handle_mav) {
                eprintln!("MAVLink 接收线程错误: {}", e);
            }
        });
    }

    // 飞控指令发送线程（10Hz RC override）
    {
        let mav_state_send = mav_state.clone();
        let input_state_send = input_state.clone();
        thread::spawn(move || {
            mav::run_control_loop(mav_state_send, input_state_send);
        });
    }

    // ── 快门回调（录像 / 拍照）──
    {
        let capture_tx = capture_tx.clone();
        let ui_handle = ui.as_weak();
        ui.on_shutter_clicked(move |mode| {
            if let Some(ui) = ui_handle.upgrade() {
                match mode {
                    0 => {
                        let is_recording = ui.get_is_recording();
                        if is_recording {
                            capture_tx.send(CaptureCommand::StopRecording).ok();
                            ui.set_is_recording(false);
                        } else {
                            let save_path = ui.get_video_save_path().to_string();
                            capture_tx
                                .send(CaptureCommand::StartRecording { save_path })
                                .ok();
                            ui.set_is_recording(true);
                            ui.set_rec_seconds(0);
                        }
                    }
                    1 => {
                        let save_path = ui.get_photo_save_path().to_string();
                        capture_tx
                            .send(CaptureCommand::TakePhoto { save_path })
                            .ok();
                    }
                    _ => {}
                }
            }
        });
    }

    // ── 键盘启用/禁用回调 ──
    {
        let input_state_kb = input_state.clone();
        ui.on_keyboard_enabled_toggled(move |enabled| {
            if !enabled {
                let mut state = input_state_kb.lock().unwrap();
                state.w = false;
                state.s = false;
                state.a = false;
                state.d = false;
                state.i = false;
                state.k = false;
                state.j = false;
                state.l = false;
                state.gripper_open = false;
                state.gripper_close = false;
            }
        });
    }

    // ── 键盘事件处理 ──
    {
        let mav_state_key = mav_state.clone();
        let input_state_key = input_state.clone();
        let ui_handle = ui.as_weak();
        ui.on_key_event(move |text, pressed| {
            let text = text.as_str();
            println!("Key event: {} pressed: {}", text, pressed);
            let mut state = input_state_key.lock().unwrap();

            match text {
                "w" => state.w = pressed,
                "s" => state.s = pressed,
                "a" => state.a = pressed,
                "d" => state.d = pressed,
                "i" => state.i = pressed,
                "k" => state.k = pressed,
                "j" => state.j = pressed,
                "l" => state.l = pressed,
                "-" => state.gripper_open = pressed,
                "=" => state.gripper_close = pressed,
                " " if pressed => {
                    if let Some(ui) = ui_handle.upgrade() {
                        let text = ui.get_armed_text();
                        let target_arm = text == "锁定";
                        let current = mav_state_key.lock().unwrap().clone();
                        if let Some(mav) = current {
                            mav::send_arm_disarm(&mav, target_arm);
                        }
                    }
                }
                "1" if pressed => {
                    let current = mav_state_key.lock().unwrap().clone();
                    if let Some(mav) = current {
                        mav::send_mode(&mav, 19); // MANUAL
                    }
                }
                "2" if pressed => {
                    let current = mav_state_key.lock().unwrap().clone();
                    if let Some(mav) = current {
                        mav::send_mode(&mav, 2); // ALT_HOLD
                    }
                }
                _ => {}
            }
        });
    }

    // ── 任务按钮（打开终端）──
    ui.on_task_clicked(move || {
        println!("Task button clicked - Opening terminal");
        #[cfg(target_os = "windows")]
        {
            if let Err(e) = std::process::Command::new("cmd")
                .args(["/C", "start"])
                .spawn()
            {
                eprintln!("Failed to open terminal: {}", e);
            }
        }
        #[cfg(target_os = "linux")]
        {
            if let Err(e) = std::process::Command::new("x-terminal-emulator").spawn() {
                eprintln!("Failed to open terminal: {}", e);
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Err(e) = std::process::Command::new("open")
                .arg("-a")
                .arg("Terminal")
                .spawn()
            {
                eprintln!("Failed to open terminal: {}", e);
            }
        }
    });

    // ── 油门上限回调 ──
    {
        let input_state_throttle = input_state.clone();
        ui.on_throttle_limit_changed(move |value| {
            let mut state = input_state_throttle.lock().unwrap();
            let v = value.round() as i32;
            let v = v.clamp(0, 100);
            state.power_percent = v as u8;
        });
    }

    // ── 应用设置回调：点击"应用设置"按钮时切换视频源并持久化 ──
    {
        let capture_tx_settings = capture_tx.clone();
        let ui_handle_settings = ui.as_weak();
        ui.on_apply_video_settings(move || {
            let ui = match ui_handle_settings.upgrade() {
                Some(ui) => ui,
                None => return,
            };

            let current_url = ui.get_video_stream_url().to_string();
            let current_index = ui.get_video_source_index();

            let new_source = match current_index {
                1 => VideoSource::Camera { device: current_url.clone() },
                _ => VideoSource::Rtsp { url: current_url.clone() },
            };
            capture_tx_settings
                .send(CaptureCommand::ChangeSource { source: new_source })
                .ok();
            println!("应用视频设置: {} (index={})", current_url, current_index);

            let s = AppSettings {
                video_source_index: current_index,
                video_stream_url: current_url,
                rov_ip: ui.get_rov_ip().to_string(),
                keyboard_enabled: ui.get_keyboard_enabled(),
                throttle_limit: ui.get_throttle_limit_percent(),
                photo_save_path: ui.get_photo_save_path().to_string(),
                video_save_path: ui.get_video_save_path().to_string(),
            };
            settings::save(&s);
        });
    }

    // ── 定时保存非视频类设置（油门、路径等），500ms 轮询但不触发视频切换 ──
    {
        let ui_handle_save = ui.as_weak();
        let timer = slint::Timer::default();
        timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(500),
            move || {
                let ui = match ui_handle_save.upgrade() {
                    Some(ui) => ui,
                    None => return,
                };
                let s = AppSettings {
                    video_source_index: ui.get_video_source_index(),
                    video_stream_url: ui.get_video_stream_url().to_string(),
                    rov_ip: ui.get_rov_ip().to_string(),
                    keyboard_enabled: ui.get_keyboard_enabled(),
                    throttle_limit: ui.get_throttle_limit_percent(),
                    photo_save_path: ui.get_photo_save_path().to_string(),
                    video_save_path: ui.get_video_save_path().to_string(),
                };
                let s = s.clone();
                std::thread::spawn(move || {
                    settings::save(&s);
                });
            },
        );
        std::mem::forget(timer);
    }

    let weak_ui = ui.as_weak();
    slint::invoke_from_event_loop(move || {
        let ui = weak_ui.unwrap();
        ui.window().set_maximized(true);
    })
    .unwrap();

    // ── 视频帧拉取定时器（60Hz，在主线程）──
    let timer_latest = latest_frame.clone();
    let timer_ui = ui.as_weak();
    let last_seq = Arc::new(std::sync::Mutex::new(0u64));
    let timer_last_seq = last_seq.clone();
    let timer = slint::Timer::default();

    // 预备空帧作为回收缓冲区，用于 swap 交换
    let spare = SharedPixelBuffer::<slint::Rgb8Pixel>::new(1, 1);
    let spare = Arc::new(std::sync::Mutex::new(spare));

    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16),
        {
            let timer_latest = timer_latest.clone();
            let timer_ui = timer_ui.clone();
            let timer_last_seq = timer_last_seq.clone();
            let spare = spare.clone();
            move || {
                if let Some(ui) = timer_ui.upgrade() {
                    let mut spare_guard = spare.lock().unwrap();
                    let current_spare = mem::replace(&mut *spare_guard, SharedPixelBuffer::<slint::Rgb8Pixel>::new(1, 1));
                    if let Some((frame, seq)) = timer_latest.swap(current_spare) {
                        let mut last = timer_last_seq.lock().unwrap();
                        if seq != *last {
                            *last = seq;
                            ui.set_video_frame(slint::Image::from_rgb8(frame));
                        } else {
                            // seq 未变，把帧放回 spare 供下次使用
                            *spare_guard = frame;
                        }
                    }
                }
            }
        },
    );
    std::mem::forget(timer);

    ui.run()
}
