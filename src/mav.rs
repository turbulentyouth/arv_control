use crate::AppWindow;
use crate::input::InputState;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use slint::Weak;
use mavlink::ardupilotmega::{MavMessage, MavModeFlag, MavCmd, COMMAND_LONG_DATA, RC_CHANNELS_OVERRIDE_DATA};
use mavlink::MavConnection;

pub type MavConn = Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>;

/// 共享的 MAVLink 连接状态，None 表示当前断开
pub type MavState = Arc<Mutex<Option<MavConn>>>;

/// 尝试连接 MAVLink，失败时返回 None（不 panic）
pub fn try_connect(conn_str: &str) -> Option<MavConn> {
    match mavlink::connect::<MavMessage>(conn_str) {
        Ok(conn) => {
            println!("MAVLink 已连接: {}", conn_str);
            Some(Arc::new(conn))
        }
        Err(e) => {
            eprintln!("MAVLink 连接失败 ({}): {}，将在后台重试", conn_str, e);
            None
        }
    }
}

/// 后台自动重连线程：每 3 秒检查一次，断开时尝试重新连接
pub fn run_mavlink_connector(conn_str: String, mav_state: MavState) {
    loop {
        thread::sleep(Duration::from_secs(3));
        let is_connected = mav_state.lock().unwrap().is_some();
        if !is_connected {
            println!("尝试重新连接 MAVLink: {}...", conn_str);
            if let Some(conn) = try_connect(&conn_str) {
                *mav_state.lock().unwrap() = Some(conn);
                println!("MAVLink 重连成功！");
            }
        }
    }
}

pub fn run_control_loop(mav_state: MavState, input: Arc<Mutex<InputState>>) {
    loop {
        thread::sleep(Duration::from_millis(100)); // 10Hz
        let (x, y, z, r, chan7) = {
            let s = input.lock().unwrap();
            let mut x: u16 = 1500;
            let mut y: u16 = 1500;
            let mut z: u16 = 1500;
            let mut r: u16 = 1500;
            let mut chan7: u16 = 65535;
            let amp = (s.power_percent as i32) * 4;
            if s.w { x = (1500 + amp) as u16; }
            if s.s { x = (1500 - amp) as u16; }
            if s.j { y = (1500 - amp) as u16; }
            if s.l { y = (1500 + amp) as u16; }
            if s.i { z = (1500 + amp) as u16; }
            if s.k { z = (1500 - amp) as u16; }
            if s.a { r = (1500 - amp) as u16; }
            if s.d { r = (1500 + amp) as u16; }
            if s.gripper_open  { chan7 = 1900; }
            if s.gripper_close { chan7 = 1100; }
            (x, y, z, r, chan7)
        };

        let msg = MavMessage::RC_CHANNELS_OVERRIDE(RC_CHANNELS_OVERRIDE_DATA {
            chan1_raw: 65535, // Pitch
            chan2_raw: 65535, // Roll
            chan3_raw: z,     // Throttle
            chan4_raw: r,     // Yaw
            chan5_raw: x,     // Forward
            chan6_raw: y,     // Lateral
            chan7_raw: chan7,
            chan8_raw: 65535,
            target_system: 1,
            target_component: 1,
            ..Default::default()
        });

        // 获取当前连接（短暂加锁后立即释放）
        let current = { mav_state.lock().unwrap().clone() };
        if let Some(mav) = current {
            if let Err(e) = mav.send(&Default::default(), &msg) {
                eprintln!("RC override 发送失败: {}", e);
                // 发送失败视为连接中断，清除状态以触发重连
                *mav_state.lock().unwrap() = None;
            }
        }
    }
}

pub fn send_arm_disarm(mav: &MavConn, arm: bool) {
    println!("发送解锁/上锁命令: {}", arm);
    let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
        target_system: 1,
        target_component: 1,
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
    println!("发送切换模式命令: {}", mode);
    let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
        target_system: 1,
        target_component: 1,
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

pub fn run_mavlink_recv(mav_state: MavState, ui_handle: Weak<AppWindow>) -> anyhow::Result<()> {
    loop {
        // 短暂加锁，克隆 Arc 后立即释放——recv() 不持有锁
        let current = { mav_state.lock().unwrap().clone() };

        match current {
            None => {
                // 未连接时休眠，等待重连线程建立连接
                thread::sleep(Duration::from_millis(500));
                continue;
            }
            Some(mav) => {
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
                                }
                                MavMessage::VFR_HUD(hud) => {
                                    ui.invoke_scroll_to_heading(hud.heading as f32);
                                    ui.set_throttle_text(format!("{}%", hud.throttle).into());
                                }
                                MavMessage::GLOBAL_POSITION_INT(pos) => {
                                    let depth = -pos.relative_alt as f32 / 1000.0;
                                    ui.set_depth_text(format!("{:.2} m", depth).into());
                                }
                                MavMessage::SYS_STATUS(status) => {
                                    ui.set_battery_text(
                                        format!("{}%", status.battery_remaining).into(),
                                    );
                                }
                                MavMessage::HEARTBEAT(hb) => {
                                    let mode_name = match hb.custom_mode {
                                        0 => "STABILIZE",
                                        2 => "ALT_HOLD",
                                        19 => "MANUAL",
                                        _ => "UNKNOWN",
                                    };
                                    if mode_name != "UNKNOWN" {
                                        ui.set_mode_text(mode_name.into());
                                    }
                                    let is_armed = hb
                                        .base_mode
                                        .contains(MavModeFlag::MAV_MODE_FLAG_SAFETY_ARMED);
                                    let armed_text = if is_armed { "解锁" } else { "锁定" };
                                    ui.set_armed_text(armed_text.into());
                                }
                                _ => {}
                            }
                        })
                        .ok();
                    }
                    Err(e) => {
                        use mavlink::error::MessageReadError;
                        // IO 超时 / WouldBlock 属于正常情况（无消息时），直接继续循环
                        let is_transient = matches!(
                            &e,
                            MessageReadError::Io(io_err)
                                if io_err.kind() == std::io::ErrorKind::WouldBlock
                                    || io_err.kind() == std::io::ErrorKind::TimedOut
                        );
                        if !is_transient {
                            eprintln!("MAVLink 接收错误: {}，标记连接断开等待重连", e);
                            // 清除连接引用，旧连接 Arc 引用归零后自动关闭
                            // 重连线程负责在端口空闲后重新建立连接
                            *mav_state.lock().unwrap() = None;
                        }
                    }
                }
            }
        }
    }
}
