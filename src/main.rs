mod input;
mod video;
mod mav;
mod gesture;

use std::thread;
use std::sync::{Arc, Mutex};
use mavlink::ardupilotmega::MavMessage;
use crate::input::InputState;
use slint::ComponentHandle;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    // Initialize logging if needed
    env_logger::init();

    let ui = AppWindow::new()?;
    
    let input_state = Arc::new(Mutex::new(InputState::default()));

    let ui_handle = ui.as_weak();
    let input_state_video = input_state.clone();
    thread::spawn(move || {
        if let Err(e) = video::run_video_player(ui_handle, input_state_video) {
            eprintln!("Video player error: {}", e);
        }
    });

    // Mavlink Connection
    let mav_conn_str = "udpin:0.0.0.0:14550";
    println!("MAVLink listening on {}", mav_conn_str);
    let mav = mavlink::connect::<MavMessage>(mav_conn_str).expect("Failed to connect to MAVLink");
    let mav = Arc::new(mav);

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

    // Key Event Handler
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
                // Toggle Arm
                if let Some(ui) = ui_handle.upgrade() {
                    let text = ui.get_armed_text();
                    let target_arm = text == "锁定";
                    mav::send_arm_disarm(&mav_clone_key, target_arm);
                }
            }
            "1" if pressed => {
                state.gesture_mode = false;
                if let Some(ui) = ui_handle.upgrade() {
                    ui.set_gesture_text("OFF".into());
                }
                mav::send_mode(&mav_clone_key, 19); // MANUAL
            }
            "2" if pressed => {
                state.gesture_mode = false;
                 if let Some(ui) = ui_handle.upgrade() {
                    ui.set_gesture_text("OFF".into());
                }
                mav::send_mode(&mav_clone_key, 2); // ALT_HOLD
            }
            "9" if pressed => {
                state.gesture_mode = true;
                if let Some(ui) = ui_handle.upgrade() {
                    ui.set_gesture_text("ON (WAITING)".into());
                }
            }
            _ => {}
        }
    });

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
             if let Err(e) = std::process::Command::new("open").arg("-a").arg("Terminal").spawn() {
                eprintln!("Failed to open terminal: {}", e);
             }
        }
    });

    let weak_ui = ui.as_weak();
    slint::invoke_from_event_loop(move || {
        let ui = weak_ui.unwrap();
        ui.window().set_maximized(true);
    })
    .unwrap();

    ui.run()
}
