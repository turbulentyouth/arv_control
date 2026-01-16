use crate::AppWindow;
use crate::input::InputState;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use slint::Weak;
use mavlink::ardupilotmega::{MavMessage, MavModeFlag, MavCmd, COMMAND_LONG_DATA, RC_CHANNELS_OVERRIDE_DATA};
use mavlink::MavConnection;

pub type MavConn = Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>;

pub fn run_control_loop(mav: MavConn, input: Arc<Mutex<InputState>>) {
    loop {
        thread::sleep(Duration::from_millis(100)); // 10Hz
        let (x, y, z, r) = {
            let s = input.lock().unwrap();
            let mut x: u16 = 1500;
            let mut y: u16 = 1500;
            let mut z: u16 = 1500;
            let mut r: u16 = 1500;
            
            // Power limit: 20% ( +/- 80 )
            if s.w { x = 1580; }
            if s.s { x = 1420; }
            if s.j { y = 1420; } // Lateral Left
            if s.l { y = 1580; } // Lateral Right
            if s.i { z = 1580; } // Ascend (Throttle Up)
            if s.k { z = 1420; } // Descend
            if s.a { r = 1420; } // Yaw Left
            if s.d { r = 1580; } // Yaw Right
            (x, y, z, r)
        };
        
        let msg = MavMessage::RC_CHANNELS_OVERRIDE(RC_CHANNELS_OVERRIDE_DATA {
            chan1_raw: 65535, // Pitch
            chan2_raw: 65535, // Roll
            chan3_raw: z,     // Throttle
            chan4_raw: r,     // Yaw
            chan5_raw: x,     // Forward
            chan6_raw: y,     // Lateral
            chan7_raw: 65535,
            chan8_raw: 65535,
            target_system: 1, 
            target_component: 1, 
            ..Default::default()
        });
        
        // Ignore errors
        if let Err(e) = mav.send(&Default::default(), &msg) {
             eprintln!("Failed to send RC override: {}", e);
        }
    }
}

pub fn send_arm_disarm(mav: &MavConn, arm: bool) {
    println!("Sending arm/disarm command: {}", arm);
    let target_system = 1;
    let target_component = 1;

    let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
        target_system,
        target_component,
        command: MavCmd::MAV_CMD_COMPONENT_ARM_DISARM,
        confirmation: 0,
        param1: if arm { 1.0 } else { 0.0 },
        param2: 21196.0, // Force
        param3: 0.0,
        param4: 0.0,
        param5: 0.0,
        param6: 0.0,
        param7: 0.0,
    });
    let _ = mav.send(&Default::default(), &msg);
}

pub fn send_mode(mav: &MavConn, mode: u32) {
    println!("Sending set mode command: {}", mode);
    let target_system = 1;
    let target_component = 1;

    // Set Mode: MAV_CMD_DO_SET_MODE = 176
    // param1: Mode (MAV_MODE), param2: Custom Mode
    
    // For ArduSub, we usually use MAV_CMD_DO_SET_MODE with param1=MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (1)
    // and param2 = custom_mode number.
    
    let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
        target_system,
        target_component,
        command: MavCmd::MAV_CMD_DO_SET_MODE,
        confirmation: 0,
        param1: 1.0, // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        param2: mode as f32,
        param3: 0.0,
        param4: 0.0,
        param5: 0.0,
        param6: 0.0,
        param7: 0.0,
    });
    let _ = mav.send(&Default::default(), &msg);
}

pub fn run_mavlink_recv(mav: MavConn, ui_handle: Weak<AppWindow>) -> anyhow::Result<()> {
    loop {
        match mav.recv() {
            Ok((_header, msg)) => {
                let ui_weak = ui_handle.clone();
                slint::invoke_from_event_loop(move || {
                    let ui = match ui_weak.upgrade() {
                        Some(ui) => ui,
                        None => return,
                    };
                    
                    match msg {
                        MavMessage::ATTITUDE(att) => {
                            let pitch_deg = att.pitch.to_degrees();
                            let roll_deg = att.roll.to_degrees();
                            ui.set_pitch_text(format!("{:.1}°", pitch_deg).into());
                            ui.set_roll_text(format!("{:.1}°", roll_deg).into());
                        },
                        MavMessage::VFR_HUD(hud) => {
                            ui.invoke_scroll_to_heading(hud.heading as f32);
                            ui.set_depth_text(format!("{:.2} m", hud.alt).into());
                            ui.set_throttle_text(format!("{}%", hud.throttle).into());
                        },
                        MavMessage::SYS_STATUS(status) => {
                            ui.set_battery_text(format!("{}%", status.battery_remaining).into());
                        },
                        MavMessage::HEARTBEAT(hb) => {
                            // Simple mode display for now
                            // ArduSub custom modes: 
                            // 0: Stabilize, 2: AltHold, 19: Manual
                            let mode_name = match hb.custom_mode {
                                0 => "STABILIZE",
                                2 => "ALT_HOLD",
                                19 => "MANUAL",
                                _ => "UNKNOWN",
                            };
                            if mode_name != "UNKNOWN" {
                                ui.set_mode_text(mode_name.into());
                            }

                            // Check armed status
                            let is_armed = hb.base_mode.contains(MavModeFlag::MAV_MODE_FLAG_SAFETY_ARMED);
                            let armed_text = if is_armed { "解锁" } else { "锁定" };
                            ui.set_armed_text(armed_text.into());
                        },
                        _ => {}
                    }
                }).ok();
            },
            Err(e) => {
                eprintln!("MAVLink recv error: {}", e);
                // Errors happen (e.g. timeout), just ignore and retry
            }
        }
    }
}
