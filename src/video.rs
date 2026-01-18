use crate::AppWindow;
use crate::gesture::{GestureLogic, HandDetector, HandPose, resolve_model_path};
use crate::input::InputState;
use image::RgbImage;
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use std::sync::{Arc, Mutex, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const DET_MODEL_FILE: &str = "hand_yolov8n.onnx";
const POSE_MODEL_FILE: &str = "rtmpose_hand.onnx";
const GESTURE_INFER_INTERVAL_MS: u64 = 250;
const GESTURE_OVERLAY_MAX_AGE_MS: u64 = 800;

const SKELETON: &[(usize, usize)] = &[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
];

fn set_pixel(buf: &mut [u8], width: u32, height: u32, x: i32, y: i32, rgb: [u8; 3]) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    if x >= width || y >= height {
        return;
    }
    let idx = ((y * width + x) as usize) * 3;
    if idx + 2 < buf.len() {
        buf[idx] = rgb[0];
        buf[idx + 1] = rgb[1];
        buf[idx + 2] = rgb[2];
    }
}

fn draw_circle(buf: &mut [u8], width: u32, height: u32, cx: i32, cy: i32, r: i32, rgb: [u8; 3]) {
    if r <= 0 {
        set_pixel(buf, width, height, cx, cy, rgb);
        return;
    }
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                set_pixel(buf, width, height, cx + dx, cy + dy, rgb);
            }
        }
    }
}

fn draw_line(buf: &mut [u8], width: u32, height: u32, x0: i32, y0: i32, x1: i32, y1: i32, rgb: [u8; 3]) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        draw_circle(buf, width, height, x0, y0, 1, rgb);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_rect(buf: &mut [u8], width: u32, height: u32, x1: i32, y1: i32, x2: i32, y2: i32, rgb: [u8; 3]) {
    draw_line(buf, width, height, x1, y1, x2, y1, rgb);
    draw_line(buf, width, height, x2, y1, x2, y2, rgb);
    draw_line(buf, width, height, x2, y2, x1, y2, rgb);
    draw_line(buf, width, height, x1, y2, x1, y1, rgb);
}

fn apply_gesture_cmd(state: &mut InputState, gesture_cmd: &str) {
    state.w = false;
    state.s = false;
    state.a = false;
    state.d = false;
    state.i = false;
    state.k = false;
    state.j = false;
    state.l = false;

    match gesture_cmd {
        "UP" => state.i = true,
        "DOWN" => state.k = true,
        "LEFT" => state.j = true,
        "RIGHT" => state.l = true,
        "FORWARD" => state.w = true,
        "BACKWARD" => state.s = true,
        "TURN_LEFT" => state.a = true,
        "TURN_RIGHT" => state.d = true,
        _ => {}
    }
}

struct GestureOverlay {
    bbox: Option<[f32; 4]>,
    kpts: Vec<(f32, f32)>,
    cmd: String,
    updated_at: Instant,
}

pub fn run_video_player(ui_handle: Weak<AppWindow>, input_state: Arc<Mutex<InputState>>) -> anyhow::Result<()> {
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

    let overlay = Arc::new(Mutex::new(GestureOverlay {
        bbox: None,
        kpts: Vec::new(),
        cmd: "UNKNOWN".to_string(),
        updated_at: Instant::now()
            .checked_sub(Duration::from_millis(GESTURE_OVERLAY_MAX_AGE_MS * 10))
            .unwrap_or_else(Instant::now),
    }));

    let (frame_tx, frame_rx) = mpsc::sync_channel::<(u32, u32, Vec<u8>)>(1);
    let gesture_ready = Arc::new(AtomicBool::new(false));

    {
        let overlay = overlay.clone();
        let ui_handle = ui_handle.clone();
        let input_state = input_state.clone();
        let gesture_ready = gesture_ready.clone();
        std::thread::spawn(move || {
            let gesture_init = std::panic::catch_unwind(|| -> anyhow::Result<(HandDetector, HandPose, GestureLogic)> {
                let det_path = resolve_model_path(DET_MODEL_FILE)?;
                let pose_path = resolve_model_path(POSE_MODEL_FILE)?;
                let detector = HandDetector::new(det_path)?;
                let pose = HandPose::new(pose_path)?;
                let logic = GestureLogic::new();
                Ok((detector, pose, logic))
            });

            let (mut detector, mut pose, logic) = match gesture_init {
                Ok(Ok(p)) => {
                    gesture_ready.store(true, Ordering::Relaxed);
                    p
                }
                Ok(Err(e)) => {
                    let ui_weak = ui_handle.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.set_gesture_text(format!("GESTURE INIT ERROR: {e}").into());
                        }
                    })
                    .ok();
                    return;
                }
                Err(_) => {
                    let ui_weak = ui_handle.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.set_gesture_text("GESTURE INIT PANIC".into());
                        }
                    })
                    .ok();
                    return;
                }
            };

            while let Ok((width, height, bytes)) = frame_rx.recv() {
                let mode_active = {
                    let s = input_state.lock().unwrap();
                    s.gesture_mode
                };
                if !mode_active {
                    continue;
                }

                let img = match RgbImage::from_raw(width, height, bytes) {
                    Some(i) => i,
                    None => continue,
                };

                let infer_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> anyhow::Result<(String, Option<[f32; 4]>, Vec<(f32, f32)>)> {
                    let dets = detector.infer(&img)?;
                    let mut best_box: Option<[f32; 4]> = None;
                    let mut max_area = 0.0;
                    for d in dets {
                        let area = (d[2] - d[0]) * (d[3] - d[1]);
                        if area > max_area {
                            max_area = area;
                            best_box = Some([d[0], d[1], d[2], d[3]]);
                        }
                    }

                    let mut cmd = "UNKNOWN".to_string();
                    let mut kpts = Vec::new();
                    if let Some(bbox) = best_box {
                        let out = pose.infer(&img, bbox)?;
                        if !out.is_empty() {
                            cmd = logic.analyze(&out);
                            kpts = out;
                        }
                    }

                    Ok((cmd, best_box, kpts))
                }));

                let (cmd, bbox, kpts) = match infer_res {
                    Ok(Ok(v)) => v,
                    Ok(Err(_e)) => continue,
                    Err(_) => continue,
                };

                {
                    let mut o = overlay.lock().unwrap();
                    o.cmd = cmd.clone();
                    o.bbox = bbox;
                    o.kpts = kpts;
                    o.updated_at = Instant::now();
                }

                {
                    let mut s = input_state.lock().unwrap();
                    apply_gesture_cmd(&mut s, &cmd);
                }
            }
        });
    }

    let mut last_send_at = Instant::now()
        .checked_sub(Duration::from_millis(GESTURE_INFER_INTERVAL_MS))
        .unwrap_or_else(Instant::now);

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

                let mut pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
                let buffer_bytes = pixel_buffer.make_mut_bytes();

                for y in 0..height as usize {
                    let src_start = y * linesize;
                    let src_end = src_start + (width as usize * 3);
                    let dst_start = y * (width as usize * 3);
                    let dst_end = dst_start + (width as usize * 3);

                    if src_end <= data.len() && dst_end <= buffer_bytes.len() {
                        buffer_bytes[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
                    }
                }

                let mode_active = {
                    let s = input_state.lock().unwrap();
                    s.gesture_mode
                };

                let mut gesture_text: Option<String> = None;
                if mode_active && gesture_ready.load(Ordering::Relaxed) {
                    let now = Instant::now();
                    if now.duration_since(last_send_at) >= Duration::from_millis(GESTURE_INFER_INTERVAL_MS) {
                        last_send_at = now;
                        let _ = frame_tx.try_send((width, height, buffer_bytes.to_vec()));
                    }

                    let (cmd, bbox, kpts, updated_at) = {
                        let o = overlay.lock().unwrap();
                        (o.cmd.clone(), o.bbox, o.kpts.clone(), o.updated_at)
                    };

                    if now.duration_since(updated_at) <= Duration::from_millis(GESTURE_OVERLAY_MAX_AGE_MS) {
                        if let Some(bbox) = bbox {
                            draw_rect(
                                buffer_bytes,
                                width,
                                height,
                                bbox[0].round() as i32,
                                bbox[1].round() as i32,
                                bbox[2].round() as i32,
                                bbox[3].round() as i32,
                                [100, 100, 100],
                            );
                            if !kpts.is_empty() {
                                for &(a, b) in SKELETON {
                                    if a < kpts.len() && b < kpts.len() {
                                        let (x0, y0) = kpts[a];
                                        let (x1, y1) = kpts[b];
                                        draw_line(
                                            buffer_bytes,
                                            width,
                                            height,
                                            x0.round() as i32,
                                            y0.round() as i32,
                                            x1.round() as i32,
                                            y1.round() as i32,
                                            [0, 255, 255],
                                        );
                                    }
                                }
                                for &(x, y) in &kpts {
                                    draw_circle(
                                        buffer_bytes,
                                        width,
                                        height,
                                        x.round() as i32,
                                        y.round() as i32,
                                        3,
                                        [255, 0, 0],
                                    );
                                }
                            }
                        }
                        gesture_text = Some(cmd);
                    }
                }

                let ui_weak = ui_handle.clone();
                slint::invoke_from_event_loop(move || {
                    let image = Image::from_rgb8(pixel_buffer);
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_video_frame(image);
                        if let Some(text) = gesture_text {
                            ui.set_gesture_text(text.into());
                        }
                    }
                }).ok();
            }
        }
    }

    Ok(())
}
