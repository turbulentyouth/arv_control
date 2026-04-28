use crate::AppWindow;
use crate::capture::{CaptureCommand, VideoSource};
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

// ── MJPEG SOF0 补丁工具 ──────────────────────────────────────────────────────

/// 在 JPEG 字节流中定位 SOF0 (FF C0) 标记并将 H/W 修补为指定值。
/// 返回修补后的拷贝；若未找到 SOF0 则返回 None。
fn patch_jpeg_dimensions(data: &[u8], width: u16, height: u16) -> Option<Cow<'_, [u8]>> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        match data[i + 1] {
            0xD8 | 0xD9 | 0xD0..=0xD7 | 0x01 => { i += 2; }
            0xC0 => {
                if i + 8 < data.len() {
                    let existing_h = u16::from_be_bytes([data[i + 5], data[i + 6]]);
                    let existing_w = u16::from_be_bytes([data[i + 7], data[i + 8]]);
                    if existing_w == width && existing_h == height {
                        return Some(Cow::Borrowed(data));
                    }
                    let mut out = data.to_vec();
                    out[i + 5] = (height >> 8) as u8;
                    out[i + 6] = (height & 0xFF) as u8;
                    out[i + 7] = (width >> 8) as u8;
                    out[i + 8] = (width & 0xFF) as u8;
                    return Some(Cow::Owned(out));
                }
                return None;
            }
            _ => {
                if i + 3 < data.len() {
                    let seg = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    i = i.saturating_add(2 + seg);
                } else {
                    return None;
                }
            }
        }
    }
    None
}

/// 从 JPEG 字节切片创建 FFmpeg AVPacket（用于修补 SOF0 后重解码）。
fn packet_from_slice(data: &[u8]) -> ffmpeg::Packet {
    ffmpeg::Packet::copy(data)
}

// MJPEG 分辨率自动检测候选（宽幅优先，从大到小）
const MJPEG_RES_CANDIDATES: &[(u16, u16)] = &[
    (3840, 1080), (3840, 1088),
    (2560, 720),  (2560, 736),  (2560, 960),
    (1280, 480),  (1280, 496),
    (1920, 1080), (1920, 1088),
    (1280, 720),  (1280, 736),
    (640,  480),
    (4096, 1080), (4096, 1088),
    (2048, 1080), (2048, 1088),
    (3840, 2160),
    (4096, 2048),
];

const MIN_BYTES_PER_PIXEL: f64 = 0.05;

/// 尝试用各种候选分辨率修补 MJPEG 帧并交给 FFmpeg 解码器解码。
/// 成功时返回解码帧并缓存分辨率。
fn try_mjpeg_candidates(
    dec: &mut ffmpeg::decoder::Video,
    jpeg_data: &[u8],
    dim_hint: &mut Option<(u16, u16)>,
) -> Option<ffmpeg::util::frame::Video> {
    // 1. 用缓存分辨率
    if let Some((w, h)) = *dim_hint {
        if let Some(patched) = patch_jpeg_dimensions(jpeg_data, w, h) {
            let pkt = packet_from_slice(&patched);
            dec.send_packet(&pkt).ok()?;
            let mut f = ffmpeg::util::frame::Video::empty();
            if dec.receive_frame(&mut f).is_ok() {
                return Some(f);
            }
        }
    }

    // 2. 候选自动检测
    for &(w, h) in MJPEG_RES_CANDIDATES {
        let pixels = w as f64 * h as f64;
        if (jpeg_data.len() as f64) / pixels < MIN_BYTES_PER_PIXEL {
            continue;
        }
        if let Some(patched) = patch_jpeg_dimensions(jpeg_data, w, h) {
            let pkt = packet_from_slice(&patched);
            dec.send_packet(&pkt).ok()?;
            let mut f = ffmpeg::util::frame::Video::empty();
            if dec.receive_frame(&mut f).is_ok() {
                println!("MJPEG 分辨率自动检测成功: {}x{}", w, h);
                *dim_hint = Some((w, h));
                return Some(f);
            }
        }
    }

    None
}

struct VideoRecorder {
    output: ffmpeg::format::context::Output,
    in_time_base: ffmpeg::Rational,
    out_time_base: ffmpeg::Rational,
    first_dts: Option<i64>,
}

impl VideoRecorder {
    fn start(
        path: &str,
        params: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> anyhow::Result<Self> {
        let mut output = ffmpeg::format::output(&path)?;
        let codec = ffmpeg::encoder::find(params.id());
        let mut stream = output.add_stream(codec)?;
        stream.set_parameters(params);
        output.write_header()?;
        let out_time_base = output.stream(0).unwrap().time_base();
        Ok(Self {
            output,
            in_time_base: time_base,
            out_time_base,
            first_dts: None,
        })
    }

    fn write_packet(&mut self, packet: &ffmpeg::Packet) -> anyhow::Result<()> {
        let mut pkt = packet.clone();
        if self.first_dts.is_none() {
            self.first_dts = pkt.dts();
        }
        if let Some(first) = self.first_dts {
            if let Some(dts) = pkt.dts() {
                pkt.set_dts(Some(dts - first));
            }
            if let Some(pts) = pkt.pts() {
                pkt.set_pts(Some(pts - first));
            }
        }
        pkt.set_stream(0);
        pkt.rescale_ts(self.in_time_base, self.out_time_base);
        pkt.write_interleaved(&mut self.output)?;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.output.write_trailer()?;
        Ok(())
    }
}

fn notify(ui: &Weak<AppWindow>, msg: &str, success: bool) {
    let msg = msg.to_string();
    let ui = ui.clone();
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui.upgrade() {
            ui.set_notification_text(msg.into());
            ui.set_notification_success(success);
            ui.set_notif_tick(0);
            ui.set_notification_visible(true);
        }
    })
    .ok();
}

fn set_recording_ui(ui: &Weak<AppWindow>, recording: bool) {
    let ui = ui.clone();
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui.upgrade() {
            ui.set_is_recording(recording);
            if !recording {
                ui.set_rec_seconds(0);
            }
        }
    })
    .ok();
}

fn save_photo(data: &[u8], width: u32, height: u32, path: &str) -> anyhow::Result<()> {
    let img = image::RgbImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from frame data"))?;
    img.save(path)?;
    Ok(())
}

fn open_input(
    source: &VideoSource,
) -> Result<ffmpeg::format::context::Input, Box<dyn std::error::Error>> {
    match source {
        VideoSource::Rtsp { url } => {
            let mut opts = ffmpeg::util::dictionary::Owned::new();
            opts.set("fflags", "nobuffer+discardcorrupt");
            opts.set("flags", "low_delay");
            opts.set("rtsp_transport", "tcp");
            opts.set("probesize", "1000000");
            opts.set("analyzeduration", "500000");
            println!("Connecting to RTSP: {}", url);
            Ok(ffmpeg::format::input_with_dictionary(url, opts)?)
        }
        VideoSource::Camera { device } => {
            let (fmt_name, url) = camera_params(device);
            println!("Opening camera: format={}, device={}", fmt_name, url);

            unsafe {
                let fmt_c = std::ffi::CString::new(fmt_name)?;
                let fmt_ptr = ffmpeg::ffi::av_find_input_format(fmt_c.as_ptr());
                if fmt_ptr.is_null() {
                    return Err(format!("Input format '{}' not found", fmt_name).into());
                }

                let url_c = std::ffi::CString::new(url.as_str())?;
                let mut ps: *mut ffmpeg::ffi::AVFormatContext = std::ptr::null_mut();

                let ret = ffmpeg::ffi::avformat_open_input(
                    &mut ps,
                    url_c.as_ptr(),
                    fmt_ptr,
                    std::ptr::null_mut(),
                );
                if ret < 0 {
                    return Err(format!("Failed to open camera (error code {})", ret).into());
                }

                let ret = ffmpeg::ffi::avformat_find_stream_info(ps, std::ptr::null_mut());
                if ret < 0 {
                    ffmpeg::ffi::avformat_close_input(&mut ps);
                    return Err(
                        format!("Failed to find stream info (error code {})", ret).into(),
                    );
                }

                Ok(ffmpeg::format::context::Input::wrap(ps))
            }
        }
    }
}

fn camera_params(device: &str) -> (&'static str, String) {
    if cfg!(target_os = "windows") {
        let url = if device.is_empty() {
            "video=Integrated Webcam".to_string()
        } else {
            format!("video={}", device)
        };
        ("dshow", url)
    } else {
        let url = if device.is_empty() {
            "/dev/video0".to_string()
        } else {
            device.to_string()
        };
        ("v4l2", url)
    }
}

/// 视频播放入口：RTSP 断线自动重连，支持运行时切换视频源
pub fn run_video_player(
    ui_handle: Weak<AppWindow>,
    capture_rx: mpsc::Receiver<CaptureCommand>,
    initial_source: VideoSource,
) -> Result<(), Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let mut current_source = initial_source;

    loop {
        match run_session(&ui_handle, &capture_rx, &current_source) {
            // 收到 ChangeSource 命令，切换到新视频源
            Ok(Some(new_source)) => {
                notify(&ui_handle, "正在切换视频源...", true);
                current_source = new_source;
            }
            // 正常退出（UI 关闭）
            Ok(None) => break,
            // 连接/解码出错，3 秒后用同一视频源重试
            Err(e) => {
                eprintln!("视频流错误: {}，3 秒后重连...", e);
                notify(&ui_handle, "视频断线，正在重连...", false);
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }
    }

    Ok(())
}

/// 单次视频会话：返回 None 表示正常结束，Some(source) 表示需要切换视频源
fn run_session(
    ui_handle: &Weak<AppWindow>,
    capture_rx: &mpsc::Receiver<CaptureCommand>,
    source: &VideoSource,
) -> Result<Option<VideoSource>, Box<dyn std::error::Error>> {
    let mut ictx = open_input(source)?;

    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;

    let video_stream_index = input.index();
    let recording_params = input.parameters();
    let recording_time_base = input.time_base();
    let codec_id = input.parameters().id();
    let is_mjpeg = codec_id == ffmpeg::codec::Id::MJPEG;

    // 统一路径：所有编解码器都走 FFmpeg 解码 + swscale
    let ctx = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
    let mut decoder = ctx.decoder().video()?;
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut scaler_key: (ffmpeg::format::Pixel, u32, u32) = (ffmpeg::format::Pixel::None, 0, 0);

    println!(
        "视频流已连接，编解码器: {:?}，解码路径: FFmpeg{}",
        codec_id,
        if is_mjpeg { "（MJPEG SOF0 自动修补）" } else { "" }
    );

    // 复用 SharedPixelBuffer：只在分辨率变化时重新分配
    let mut pb: Option<SharedPixelBuffer<Rgb8Pixel>> = None;
    // 帧丢弃：UI 渲染中标记，防止 invoke_from_event_loop 队列积压
    let rendering = Arc::new(AtomicBool::new(false));
    // 帧推送频率控制：限制为 ~30 FPS 以减少 UI 线程压力
    let mut last_push_time = std::time::Instant::now();

    // 录像异步写入通道
    let mut recorder_tx: Option<mpsc::Sender<ffmpeg::Packet>> = None;
    let mut take_photo_path: Option<String> = None;
    let mut first_frame_logged = false;
    let mut mjpeg_dim_hint: Option<(u16, u16)> = None;

    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }

        // 处理控制命令（非阻塞）
        while let Ok(cmd) = capture_rx.try_recv() {
            match cmd {
                CaptureCommand::TakePhoto { save_path } => {
                    std::fs::create_dir_all(&save_path).ok();
                    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                    take_photo_path = Some(format!("{}/photo_{}.png", save_path, ts));
                }
                CaptureCommand::StartRecording { save_path } => {
                    if recorder_tx.is_none() {
                        std::fs::create_dir_all(&save_path).ok();
                        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                        let path = format!("{}/video_{}.mp4", save_path, ts);
                        match VideoRecorder::start(
                            &path,
                            recording_params.clone(),
                            recording_time_base,
                        ) {
                            Ok(recorder) => {
                                let (tx, rx) = mpsc::channel();
                                recorder_tx = Some(tx);
                                let record_ui = ui_handle.clone();
                                thread::spawn(move || {
                                    let mut rec = recorder;
                                    while let Ok(packet) = rx.recv() {
                                        if let Err(e) = rec.write_packet(&packet) {
                                            eprintln!("录像写入错误: {}", e);
                                        }
                                    }
                                    match rec.finish() {
                                        Ok(()) => notify(&record_ui, "录像已保存", true),
                                        Err(e) => notify(
                                            &record_ui,
                                            &format!("录像保存失败: {}", e),
                                            false,
                                        ),
                                    }
                                    set_recording_ui(&record_ui, false);
                                });
                                notify(ui_handle, &format!("开始录像: {}", path), true);
                            }
                            Err(e) => {
                                notify(ui_handle, &format!("录像启动失败: {}", e), false);
                                set_recording_ui(ui_handle, false);
                            }
                        }
                    }
                }
                CaptureCommand::StopRecording => {
                    recorder_tx.take();
                }
                CaptureCommand::ChangeSource { source: new_source } => {
                    recorder_tx.take();
                    set_recording_ui(ui_handle, false);
                    return Ok(Some(new_source));
                }
            }
        }

        // 异步发送包到录像线程
        if let Some(ref tx) = recorder_tx {
            tx.send(packet.clone()).ok();
        }

        // MJPEG：先保存一份 JPEG 数据拷贝，用于解码失败时的 SOF0 修补回退
        let mjpeg_data: Option<Vec<u8>> = if is_mjpeg {
            packet.data().map(|d| d.to_vec())
        } else {
            None
        };

        decoder.send_packet(&packet)?;
        let mut decoded = ffmpeg::util::frame::Video::empty();
        let mut received = false;

        while decoder.receive_frame(&mut decoded).is_ok() {
            received = true;

            let width = decoded.width();
            let height = decoded.height();
            let fmt = decoded.format();

            if width == 0 || height == 0 {
                continue;
            }

            if !first_frame_logged {
                println!("视频流首帧分辨率: {}x{}, 编解码器: {:?}", width, height, codec_id);
                first_frame_logged = true;
            }

            // swscale：按需重建
            // A. 修复 swscaler 警告：将 YUVJ420P 转换为 YUV420P
            let src_format = if fmt == ffmpeg::format::Pixel::YUVJ420P {
                ffmpeg::format::Pixel::YUV420P
            } else {
                fmt
            };
            let key = (src_format, width, height);
            if scaler.is_none() || scaler_key != key {
                scaler = Some(ffmpeg::software::scaling::context::Context::get(
                    src_format,
                    width,
                    height,
                    ffmpeg::format::Pixel::RGB24,
                    width,
                    height,
                    ffmpeg::software::scaling::flag::Flags::BILINEAR,
                )?);
                scaler_key = key;
            }

            let sc = scaler.as_mut().unwrap();
            let mut rgb_frame = ffmpeg::util::frame::Video::empty();
            sc.run(&decoded, &mut rgb_frame)?;

            let src_data = rgb_frame.data(0);
            let src_linesize = rgb_frame.stride(0);
            let dst_linesize = width as usize * 3;

            // 复用 SharedPixelBuffer：仅在分辨率变化时重新分配
            if pb.as_ref().is_none_or(|p| p.width() != width || p.height() != height) {
                pb = Some(SharedPixelBuffer::<Rgb8Pixel>::new(width, height));
            }
            let buf = pb.as_mut().unwrap();

            {
                let dst = buf.make_mut_bytes();
                // C. 优化内存拷贝：如果步长相等，直接拷贝整个 buffer
                if src_linesize == dst_linesize {
                    let copy_size = dst_linesize * height as usize;
                    if copy_size <= src_data.len() && copy_size <= dst.len() {
                        dst[..copy_size].copy_from_slice(&src_data[..copy_size]);
                    }
                } else {
                    for y in 0..height as usize {
                        let src_start = y * src_linesize;
                        let dst_start = y * dst_linesize;
                        if src_start + dst_linesize <= src_data.len()
                            && dst_start + dst_linesize <= dst.len()
                        {
                            dst[dst_start..dst_start + dst_linesize]
                                .copy_from_slice(&src_data[src_start..src_start + dst_linesize]);
                        }
                    }
                }

                if let Some(photo_path) = take_photo_path.take() {
                    match save_photo(dst, width, height, &photo_path) {
                        Ok(()) => notify(ui_handle, &format!("拍照成功: {}", photo_path), true),
                        Err(e) => notify(ui_handle, &format!("拍照失败: {}", e), false),
                    }
                }
            }

            // B. 限制帧推送频率：控制为约 30 FPS (33ms)
            let now = std::time::Instant::now();
            if now.duration_since(last_push_time) < std::time::Duration::from_millis(33) {
                continue;
            }
            // 帧丢弃：UI 仍在处理上一帧时跳过
            if rendering.load(Ordering::Acquire) {
                continue;
            }
            rendering.store(true, Ordering::Release);
            last_push_time = now;

            let pb_clone = buf.clone();
            let ui_weak = ui_handle.clone();
            let rendering_clone = rendering.clone();
            slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_video_frame(Image::from_rgb8(pb_clone));
                }
                rendering_clone.store(false, Ordering::Release);
            })
            .ok();
        }

        // MJPEG 回退：原包解码失败时，修补 SOF0 后重试
        if !received && is_mjpeg {
            if let Some(ref data) = mjpeg_data {
                if let Some(patched_frame) = try_mjpeg_candidates(&mut decoder, data, &mut mjpeg_dim_hint) {
                    let width = patched_frame.width();
                    let height = patched_frame.height();
                    let fmt = patched_frame.format();

                    if width == 0 || height == 0 {
                        continue;
                    }

                    if !first_frame_logged {
                        println!("视频流首帧分辨率: {}x{}（SOF0 已修补）", width, height);
                        first_frame_logged = true;
                    }

                    // swscale：按需重建
                    // A. 修复 swscaler 警告：将 YUVJ420P 转换为 YUV420P
                    let src_format = if fmt == ffmpeg::format::Pixel::YUVJ420P {
                        ffmpeg::format::Pixel::YUV420P
                    } else {
                        fmt
                    };
                    let key = (src_format, width, height);
                    if scaler.is_none() || scaler_key != key {
                        scaler = Some(ffmpeg::software::scaling::context::Context::get(
                            src_format,
                            width,
                            height,
                            ffmpeg::format::Pixel::RGB24,
                            width,
                            height,
                            ffmpeg::software::scaling::flag::Flags::BILINEAR,
                        )?);
                        scaler_key = key;
                    }

                    let sc = scaler.as_mut().unwrap();
                    let mut rgb_frame = ffmpeg::util::frame::Video::empty();
                    sc.run(&patched_frame, &mut rgb_frame)?;

                    let src_data = rgb_frame.data(0);
                    let src_linesize = rgb_frame.stride(0);
                    let dst_linesize = width as usize * 3;

                    if pb.as_ref().is_none_or(|p| p.width() != width || p.height() != height) {
                        pb = Some(SharedPixelBuffer::<Rgb8Pixel>::new(width, height));
                    }
                    let buf = pb.as_mut().unwrap();

                    {
                        let dst = buf.make_mut_bytes();
                        // C. 优化内存拷贝：如果步长相等，直接拷贝整个 buffer
                        if src_linesize == dst_linesize {
                            let copy_size = dst_linesize * height as usize;
                            if copy_size <= src_data.len() && copy_size <= dst.len() {
                                dst[..copy_size].copy_from_slice(&src_data[..copy_size]);
                            }
                        } else {
                            for y in 0..height as usize {
                                let src_start = y * src_linesize;
                                let dst_start = y * dst_linesize;
                                if src_start + dst_linesize <= src_data.len()
                                    && dst_start + dst_linesize <= dst.len()
                                {
                                    dst[dst_start..dst_start + dst_linesize]
                                        .copy_from_slice(&src_data[src_start..src_start + dst_linesize]);
                                }
                            }
                        }
                    }

                    // B. 限制帧推送频率：控制为约 30 FPS (33ms)
                    let now = std::time::Instant::now();
                    if now.duration_since(last_push_time) < std::time::Duration::from_millis(33) {
                        continue;
                    }
                    // 帧丢弃：UI 仍在处理上一帧时跳过
                    if rendering.load(Ordering::Acquire) {
                        continue;
                    }
                    rendering.store(true, Ordering::Release);
                    last_push_time = now;

                    let pb_clone = buf.clone();
                    let ui_weak = ui_handle.clone();
                    let rendering_clone = rendering.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.set_video_frame(Image::from_rgb8(pb_clone));
                        }
                        rendering_clone.store(false, Ordering::Release);
                    })
                    .ok();
                }
            }
        }
    }

    drop(recorder_tx);

    Ok(None)
}
