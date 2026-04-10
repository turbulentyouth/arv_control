use crate::AppWindow;
use crate::capture::{CaptureCommand, VideoSource};
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use std::sync::mpsc;

/// 视频帧写入槽：Primary 对应单目（video-frame），Secondary 对应双目（video-frame-stereo）
#[derive(Clone, Copy)]
pub enum VideoSlot {
    Primary,
    Secondary,
}

// ── MJPEG SOF0 补丁工具 ──────────────────────────────────────────────────────

/// 在 JPEG 字节流中定位 SOF0 (FF C0) 标记并将 H/W 修补为指定值。
/// 返回修补后的拷贝；若未找到 SOF0 则返回 None。
fn patch_jpeg_dimensions(data: &[u8], width: u16, height: u16) -> Option<Vec<u8>> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        match data[i + 1] {
            // 无 length 字段的标记
            0xD8 | 0xD9 | 0xD0..=0xD7 | 0x01 => { i += 2; }
            0xC0 => {
                // SOF0: FF C0 [len 2] [precision 1] [height 2] [width 2] ...
                if i + 8 < data.len() {
                    let mut out = data.to_vec();
                    out[i + 5] = (height >> 8) as u8;
                    out[i + 6] = (height & 0xFF) as u8;
                    out[i + 7] = (width >> 8) as u8;
                    out[i + 8] = (width & 0xFF) as u8;
                    return Some(out);
                }
                return None;
            }
            _ => {
                // 含 length 字段，length 包含自身 2 字节
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

/// 解码 MJPEG 帧，自动处理 SOF0 中 W/H=0 的情况。
///
/// 策略：
/// 1. 直接解码（SOF0 正常时直接成功）
/// 2. 用缓存的已知分辨率修补 SOF0 后解码
/// 3. 从大到小尝试常见双目分辨率候选，找到能成功解码的就缓存并返回
///
/// 候选从大到小排列可避免"用小分辨率读大图前半段"的误判：
/// 给大图声明小尺寸时 Huffman 流会在 EOI 前耗尽，解码器报错；
/// 给大图声明正确大尺寸则完整解码成功。
fn auto_decode_mjpeg(
    data: &[u8],
    dim_hint: &mut Option<(u16, u16)>,
) -> Option<image::RgbImage> {
    // 1. 直接尝试
    if let Ok(img) = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg) {
        return Some(img.into_rgb8());
    }

    // 2. 缓存分辨率
    if let Some((w, h)) = *dim_hint {
        if let Some(ref patched) = patch_jpeg_dimensions(data, w, h) {
            if let Ok(img) = image::load_from_memory_with_format(patched, image::ImageFormat::Jpeg) {
                return Some(img.into_rgb8());
            }
        }
    }

    // 3. 候选自动检测
    // 排列规则：双目常用宽幅分辨率优先；用"最低字节/像素"过滤器
    // 排除像素量远超实际数据量的候选（避免 zune-jpeg 宽松模式的误判）。
    // 典型 MJPEG：0.05 bpp（极低质量）~ 2.0 bpp（无压缩）；
    // 若 data_len / pixels < 0.05 则该分辨率不可信。
    const MIN_BYTES_PER_PIXEL: f64 = 0.05;

    const CANDIDATES: &[(u16, u16)] = &[
        // 双目常见宽幅（优先）
        (3840, 1080), (3840, 1088),
        (2560, 720),  (2560, 736),  (2560, 960),
        (1280, 480),  (1280, 496),
        // 单目常见
        (1920, 1080), (1920, 1088),
        (1280, 720),  (1280, 736),
        (640,  480),
        // 较少见的宽幅
        (4096, 1080), (4096, 1088),
        (2048, 1080), (2048, 1088),
        (3840, 2160),
        (4096, 2048),
    ];

    for &(w, h) in CANDIDATES {
        // 字节/像素合理性过滤
        let pixels = w as f64 * h as f64;
        if (data.len() as f64) / pixels < MIN_BYTES_PER_PIXEL {
            continue;
        }
        if let Some(ref patched) = patch_jpeg_dimensions(data, w, h) {
            if let Ok(img) = image::load_from_memory_with_format(patched, image::ImageFormat::Jpeg) {
                println!("MJPEG 分辨率自动检测成功: {}x{}", w, h);
                *dim_hint = Some((w, h));
                return Some(img.into_rgb8());
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
            // MJPEG over RTSP 的 SDP 通常不含分辨率，probesize 够用即可
            // 实际尺寸由解码器从每帧 JPEG 数据中自动读取（延迟初始化 scaler）
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
    slot: VideoSlot,
) -> Result<(), Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let mut current_source = initial_source;

    loop {
        match run_session(&ui_handle, &capture_rx, &current_source, slot) {
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
    slot: VideoSlot,
) -> Result<Option<VideoSource>, Box<dyn std::error::Error>> {
    let mut ictx = open_input(source)?;

    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;

    let video_stream_index = input.index();
    let recording_params = input.parameters();
    let recording_time_base = input.time_base();

    // MJPEG over RTSP：FFmpeg 的 MJPEG 解码器在 SDP 无分辨率时会拒绝解码，
    // 直接把每个 AVPacket（由 RTSP 解复用器重组好的完整 JPEG 帧）交给 image crate 解码
    let is_mjpeg = input.parameters().id() == ffmpeg::codec::Id::MJPEG;

    // 非 MJPEG 路径：使用 FFmpeg 解码器 + swscale
    let mut decoder: Option<ffmpeg::decoder::Video> = None;
    let mut scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut scaler_key: (ffmpeg::format::Pixel, u32, u32) = (ffmpeg::format::Pixel::None, 0, 0);

    if !is_mjpeg {
        let ctx = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        decoder = Some(ctx.decoder().video()?);
    }

    println!(
        "视频流已连接，编解码器: {:?}，解码路径: {}",
        input.parameters().id(),
        if is_mjpeg { "image-crate (JPEG直解)" } else { "FFmpeg解码器" }
    );

    let mut recorder: Option<VideoRecorder> = None;
    let mut take_photo_path: Option<String> = None;
    let mut first_frame_logged = false;
    // MJPEG 自动检测缓存：首帧确定后直接使用，避免每帧重试
    let mut mjpeg_dim_hint: Option<(u16, u16)> = None;

    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }

        // 处理控制命令（非阻塞）
        while let Ok(cmd) = capture_rx.try_recv() {
            match cmd {
                CaptureCommand::TakePhoto { save_path } => {
                    // 仅主槽（单目）响应拍照
                    if matches!(slot, VideoSlot::Primary) {
                        std::fs::create_dir_all(&save_path).ok();
                        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                        take_photo_path = Some(format!("{}/photo_{}.png", save_path, ts));
                    }
                }
                CaptureCommand::StartRecording { save_path } => {
                    // 仅主槽（单目）响应录像
                    if matches!(slot, VideoSlot::Primary) && recorder.is_none() {
                        std::fs::create_dir_all(&save_path).ok();
                        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                        let path = format!("{}/video_{}.mp4", save_path, ts);
                        match VideoRecorder::start(
                            &path,
                            recording_params.clone(),
                            recording_time_base,
                        ) {
                            Ok(r) => {
                                recorder = Some(r);
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
                    // 仅主槽（单目）响应停止录像
                    if matches!(slot, VideoSlot::Primary) {
                        if let Some(rec) = recorder.take() {
                            match rec.finish() {
                                Ok(()) => notify(ui_handle, "录像已保存", true),
                                Err(e) => {
                                    notify(ui_handle, &format!("录像保存失败: {}", e), false);
                                }
                            }
                        }
                    }
                }
                CaptureCommand::ChangeSource { source: new_source } => {
                    if let Some(rec) = recorder.take() {
                        rec.finish().ok();
                        if matches!(slot, VideoSlot::Primary) {
                            set_recording_ui(ui_handle, false);
                        }
                    }
                    return Ok(Some(new_source));
                }
            }
        }

        if let Some(ref mut rec) = recorder {
            if let Err(e) = rec.write_packet(&packet) {
                eprintln!("录像写入错误: {}", e);
            }
        }

        if is_mjpeg {
            // MJPEG 路径：AVPacket 数据是 FFmpeg RFC2435 重组后的 JPEG。
            // 部分相机的 RTP 头中 W/H=0，导致 SOF0 也是 0/0；
            // auto_decode_mjpeg 会自动修补 SOF0 并缓存正确分辨率。
            let jpeg_data = match packet.data() {
                Some(d) if !d.is_empty() => d,
                _ => continue,
            };

            let rgb = match auto_decode_mjpeg(jpeg_data, &mut mjpeg_dim_hint) {
                Some(img) => img,
                None => {
                    eprintln!("MJPEG 帧解码失败，跳过");
                    continue;
                }
            };

            let width = rgb.width();
            let height = rgb.height();

            if !first_frame_logged {
                println!("视频流首帧分辨率: {}x{}", width, height);
                first_frame_logged = true;
            }

            if let Some(photo_path) = take_photo_path.take() {
                match save_photo(rgb.as_raw(), width, height, &photo_path) {
                    Ok(()) => notify(ui_handle, &format!("拍照成功: {}", photo_path), true),
                    Err(e) => notify(ui_handle, &format!("拍照失败: {}", e), false),
                }
            }

            let raw = rgb.into_raw();
            let mut pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
            pixel_buffer.make_mut_bytes().copy_from_slice(&raw);

            let ui_weak = ui_handle.clone();
            slint::invoke_from_event_loop(move || {
                let image = Image::from_rgb8(pixel_buffer);
                if let Some(ui) = ui_weak.upgrade() {
                    match slot {
                        VideoSlot::Primary => ui.set_video_frame(image),
                        VideoSlot::Secondary => ui.set_video_frame_stereo(image),
                    }
                }
            })
            .ok();
        } else {
            // 非 MJPEG 路径：FFmpeg 解码器 + swscale 转 RGB24
            let dec = decoder.as_mut().unwrap();
            dec.send_packet(&packet)?;
            let mut decoded = ffmpeg::util::frame::Video::empty();

            while dec.receive_frame(&mut decoded).is_ok() {
                let width = decoded.width();
                let height = decoded.height();
                let fmt = decoded.format();

                if width == 0 || height == 0 {
                    continue;
                }

                let key = (fmt, width, height);
                if scaler.is_none() || scaler_key != key {
                    if !first_frame_logged {
                        println!("视频流首帧分辨率: {}x{}, 编解码器: {:?}", width, height, dec.id());
                        first_frame_logged = true;
                    }
                    scaler = Some(ffmpeg::software::scaling::context::Context::get(
                        fmt,
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

                let data = rgb_frame.data(0);
                let linesize = rgb_frame.stride(0);

                if let Some(photo_path) = take_photo_path.take() {
                    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
                    for y in 0..height as usize {
                        let start = y * linesize;
                        let end = start + (width as usize * 3);
                        if end <= data.len() {
                            rgb_data.extend_from_slice(&data[start..end]);
                        }
                    }
                    match save_photo(&rgb_data, width, height, &photo_path) {
                        Ok(()) => notify(ui_handle, &format!("拍照成功: {}", photo_path), true),
                        Err(e) => notify(ui_handle, &format!("拍照失败: {}", e), false),
                    }
                }

                let mut pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
                let buffer_bytes = pixel_buffer.make_mut_bytes();

                for y in 0..height as usize {
                    let src_start = y * linesize;
                    let src_end = src_start + (width as usize * 3);
                    let dst_start = y * (width as usize * 3);
                    let dst_end = dst_start + (width as usize * 3);

                    if src_end <= data.len() && dst_end <= buffer_bytes.len() {
                        buffer_bytes[dst_start..dst_end]
                            .copy_from_slice(&data[src_start..src_end]);
                    }
                }

                let ui_weak = ui_handle.clone();
                slint::invoke_from_event_loop(move || {
                    let image = Image::from_rgb8(pixel_buffer);
                    if let Some(ui) = ui_weak.upgrade() {
                        match slot {
                            VideoSlot::Primary => ui.set_video_frame(image),
                            VideoSlot::Secondary => ui.set_video_frame_stereo(image),
                        }
                    }
                })
                .ok();
            }
        }
    }

    if let Some(rec) = recorder.take() {
        rec.finish().ok();
    }

    Ok(None)
}
