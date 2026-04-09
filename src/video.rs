use crate::AppWindow;
use crate::capture::{CaptureCommand, VideoSource};
use slint::{SharedPixelBuffer, Image, Rgb8Pixel, Weak};
use ffmpeg_next as ffmpeg;
use std::sync::mpsc;

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
            opts.set("fflags", "nobuffer");
            opts.set("flags", "low_delay");
            opts.set("rtsp_transport", "tcp");
            opts.set("probesize", "32");
            opts.set("analyzeduration", "0");
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

pub fn run_video_player(
    ui_handle: Weak<AppWindow>,
    capture_rx: mpsc::Receiver<CaptureCommand>,
    source: VideoSource,
) -> Result<(), Box<dyn std::error::Error>> {
    ffmpeg::init()?;

    let mut ictx = open_input(&source)?;

    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;

    let video_stream_index = input.index();
    let recording_params = input.parameters();
    let recording_time_base = input.time_base();

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

    println!(
        "Video stream: {}x{}, codec: {:?}",
        decoder.width(),
        decoder.height(),
        decoder.id()
    );

    let mut recorder: Option<VideoRecorder> = None;
    let mut take_photo_path: Option<String> = None;

    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }

        while let Ok(cmd) = capture_rx.try_recv() {
            match cmd {
                CaptureCommand::TakePhoto { save_path } => {
                    std::fs::create_dir_all(&save_path).ok();
                    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                    take_photo_path = Some(format!("{}/photo_{}.png", save_path, ts));
                }
                CaptureCommand::StartRecording { save_path } => {
                    if recorder.is_none() {
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
                                notify(&ui_handle, &format!("开始录像: {}", path), true);
                            }
                            Err(e) => {
                                notify(
                                    &ui_handle,
                                    &format!("录像启动失败: {}", e),
                                    false,
                                );
                                set_recording_ui(&ui_handle, false);
                            }
                        }
                    }
                }
                CaptureCommand::StopRecording => {
                    if let Some(rec) = recorder.take() {
                        match rec.finish() {
                            Ok(()) => notify(&ui_handle, "录像已保存", true),
                            Err(e) => {
                                notify(
                                    &ui_handle,
                                    &format!("录像保存失败: {}", e),
                                    false,
                                );
                            }
                        }
                    }
                }
            }
        }

        if let Some(ref mut rec) = recorder {
            if let Err(e) = rec.write_packet(&packet) {
                eprintln!("Recording write error: {}", e);
            }
        }

        decoder.send_packet(&packet)?;
        let mut decoded = ffmpeg::util::frame::Video::empty();

        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb_frame = ffmpeg::util::frame::Video::empty();
            scaler.run(&decoded, &mut rgb_frame)?;

            let width = rgb_frame.width();
            let height = rgb_frame.height();
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
                    Ok(()) => notify(&ui_handle, &format!("拍照成功: {}", photo_path), true),
                    Err(e) => notify(&ui_handle, &format!("拍照失败: {}", e), false),
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
                    buffer_bytes[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
                }
            }

            let ui_weak = ui_handle.clone();
            slint::invoke_from_event_loop(move || {
                let image = Image::from_rgb8(pixel_buffer);
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_video_frame(image);
                }
            })
            .ok();
        }
    }

    if let Some(rec) = recorder.take() {
        rec.finish().ok();
    }

    Ok(())
}
