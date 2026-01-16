slint::include_modules!();

use std::thread;
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use mavlink::ardupilotmega::MavMessage;

fn main() -> Result<(), slint::PlatformError> {
    // Initialize logging if needed
    env_logger::init();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();
    let ui_handle_mav = ui.as_weak();

    // Spawn video player thread
    thread::spawn(move || {
        if let Err(e) = run_video_player(ui_handle) {
            eprintln!("Video player error: {}", e);
        }
    });

    // Spawn MAVLink thread
    thread::spawn(move || {
        if let Err(e) = run_mavlink(ui_handle_mav) {
            eprintln!("MAVLink error: {}", e);
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

fn run_mavlink(ui_handle: Weak<AppWindow>) -> anyhow::Result<()> {
    // Listen on port 14550 (standard for GCS)
    let mav_conn_str = "udpin:0.0.0.0:14550";
    println!("MAVLink listening on {}", mav_conn_str);
    
    // Connect to the ROV (or wait for it to connect to us)
    let mav = mavlink::connect::<MavMessage>(mav_conn_str)?;

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
                        },
                        _ => {}
                    }
                }).ok();
            },
            Err(_e) => {
                // eprintln!("MAVLink recv error: {}", e);
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
