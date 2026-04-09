mod input;
mod video;
mod mav;
mod capture;

use std::thread;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use mavlink::ardupilotmega::MavMessage;
use crate::input::InputState;
use crate::capture::{CaptureCommand, VideoSource};
use slint::ComponentHandle;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    env_logger::init();

    let ui = AppWindow::new()?;

    ui.set_version_text(env!("CARGO_PKG_VERSION").into());

    // Set default save paths from system directories
    let photo_path = dirs::picture_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("arv_control");
    let video_path = dirs::video_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("arv_control");
    ui.set_photo_save_path(photo_path.to_string_lossy().to_string().into());
    ui.set_video_save_path(video_path.to_string_lossy().to_string().into());

    let (capture_tx, capture_rx) = capture::channel();

    // Build video source from settings
    let source_index = ui.get_video_source_index();
    let stream_url = ui.get_video_stream_url().to_string();
    let source = match source_index {
        1 => VideoSource::Camera { device: stream_url },
        _ => VideoSource::Rtsp { url: stream_url },
    };

    // Spawn video player thread
    let ui_handle_video = ui.as_weak();
    thread::spawn(move || {
        if let Err(e) = video::run_video_player(ui_handle_video, capture_rx, source) {
            eprintln!("Video player error: {}", e);
        }
    });

    // Mavlink Connection
    let mav_conn_str = "udpin:0.0.0.0:14550";
    println!("MAVLink listening on {}", mav_conn_str);
    let mav = mavlink::connect::<MavMessage>(mav_conn_str).expect("Failed to connect to MAVLink");
    let mav = Arc::new(mav);

    // Input State
    let input_state = Arc::new(Mutex::new(InputState::default()));

    // Recv Thread
    let mav_clone_recv = mav.clone();
    let ui_handle_mav = ui.as_weak();
    thread::spawn(move || {
        if let Err(e) = mav::run_mavlink_recv(mav_clone_recv, ui_handle_mav) {
            eprintln!("MAVLink recv error: {}", e);
        }
    });

    // Control/Send Thread
    let mav_clone_send = mav.clone();
    let input_state_send = input_state.clone();
    thread::spawn(move || {
        mav::run_control_loop(mav_clone_send, input_state_send);
    });

    // Shutter callback (video recording / photo capture)
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

    // Keyboard enable/disable callback — reset all keys when disabled
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
            }
        });
    }

    // Key Event Handler
    {
        let mav_clone_key = mav.clone();
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
                " " if pressed => {
                    if let Some(ui) = ui_handle.upgrade() {
                        let text = ui.get_armed_text();
                        let target_arm = text == "锁定";
                        mav::send_arm_disarm(&mav_clone_key, target_arm);
                    }
                }
                "1" if pressed => mav::send_mode(&mav_clone_key, 19), // MANUAL
                "2" if pressed => mav::send_mode(&mav_clone_key, 2),  // ALT_HOLD
                _ => {}
            }
        });
    }

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

    {
        let input_state_throttle = input_state.clone();
        ui.on_throttle_limit_changed(move |value| {
            let mut state = input_state_throttle.lock().unwrap();
            let v = value.round() as i32;
            let v = v.clamp(0, 100);
            state.power_percent = v as u8;
        });
    }

    let weak_ui = ui.as_weak();
    slint::invoke_from_event_loop(move || {
        let ui = weak_ui.unwrap();
        ui.window().set_maximized(true);
    })
    .unwrap();

    ui.run()
}
