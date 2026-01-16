slint::include_modules!();

use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use mavlink::ardupilotmega::{MavMessage, MavModeFlag, MavCmd, COMMAND_LONG_DATA, RC_CHANNELS_OVERRIDE_DATA};
// use mavlink::common::{RcChannelsOverride, CommandLong};
use mavlink::MavConnection;

#[derive(Default)]
struct InputState {
    w: bool, s: bool,
    a: bool, d: bool,
    i: bool, k: bool,
    j: bool, l: bool,
}

fn main() -> Result<(), slint::PlatformError> {
    // Initialize logging if needed
    env_logger::init();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    // Spawn video player thread
    thread::spawn(move || {
        if let Err(e) = run_video_player(ui_handle) {
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
        if let Err(e) = run_mavlink_recv(mav_clone_recv, ui_handle_mav) {
            eprintln!("MAVLink recv error: {}", e);
        }
    });

    // Control/Send Thread
    let mav_clone_send = mav.clone();
    let input_state_send = input_state.clone();
    thread::spawn(move || {
        run_control_loop(mav_clone_send, input_state_send);
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
                    send_arm_disarm(&mav_clone_key, target_arm);
                }
            }
            "1" if pressed => send_mode(&mav_clone_key, 19), // MANUAL
            "2" if pressed => send_mode(&mav_clone_key, 2),  // ALT_HOLD
            _ => {}
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

fn run_control_loop(mav: Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>, input: Arc<Mutex<InputState>>) {
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

fn send_arm_disarm(mav: &Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>, arm: bool) {
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

fn send_mode(mav: &Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>, mode: u32) {
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

fn run_mavlink_recv(mav: Arc<Box<dyn MavConnection<MavMessage> + Send + Sync>>, ui_handle: Weak<AppWindow>) -> anyhow::Result<()> {
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

fn run_video_player(ui_handle: Weak<AppWindow>) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize FFmpeg
    ffmpeg::init()?;

    let url = "rtsp://192.168.137.2:8554/video";
    println!("Connecting to stream: {}", url);

    // Set input options for low latency
    let mut opts = ffmpeg::util::dictionary::Owned::new();
    opts.set("fflags", "nobuffer");
    opts.set("flags", "low_delay");
    opts.set("rtsp_transport", "tcp"); // Use TCP for reliability, or udp for lower latency
    opts.set("probesize", "32");       // Reduce probe size to start faster
    opts.set("analyzeduration", "0");  // Reduce analyze duration

    let mut ictx = ffmpeg::format::input_with_dictionary(&url, opts)?;

    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;
    
    let video_stream_index = input.index();

    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;

    println!("Video stream found: {}x{}, codec: {:?}", decoder.width(), decoder.height(), decoder.id());

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg::util::frame::Video::empty();
            
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = ffmpeg::util::frame::Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;

                let width = rgb_frame.width();
                let height = rgb_frame.height();
                let data = rgb_frame.data(0);
                let linesize = rgb_frame.stride(0);

                // Create Slint SharedPixelBuffer
                let mut pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
                let buffer_bytes = pixel_buffer.make_mut_bytes();

                // Copy data row by row
                for y in 0..height as usize {
                    let src_start = y * linesize;
                    let src_end = src_start + (width as usize * 3);
                    let dst_start = y * (width as usize * 3);
                    let dst_end = dst_start + (width as usize * 3);

                    if src_end <= data.len() && dst_end <= buffer_bytes.len() {
                        buffer_bytes[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
                    }
                }

                let ui_weak = ui_handle.clone();
                slint::invoke_from_event_loop(move || {
                    let image = Image::from_rgb8(pixel_buffer);
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_video_frame(image);
                    }
                }).ok(); // Ignore errors if UI is gone
            }
        }
    }

    Ok(())
}
