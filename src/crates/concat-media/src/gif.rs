// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! GIF 导出：两遍调色板编码。
//!
//! 第一遍把整段视频经 `fps,scale,palettegen` 滤镜得到一张 256 色调色板
//! （即命令行 `palettegen` 写出的那张 PNG，这里留在内存里，不落盘）；
//! 第二遍再解码一次视频，与调色板一起送进 `paletteuse`，得到 PAL8 画面交给
//! `gif` 编码器，GIF muxer 的 `loop` 选项按 [`GifOptions::loop_value`] 设置。
//! 全部通过链接的 libavfilter / libavcodec 完成，不调用 `ffmpeg` 程序。

use std::path::{Path, PathBuf};

use ffmpeg_the_third as ffmpeg;
use ffmpeg_the_third::codec::encoder;
use ffmpeg_the_third::filter;
use ffmpeg_the_third::util::format::Pixel;
use ffmpeg_the_third::util::frame::video::Video;

use crate::error::{Error, Result};
use crate::export_formats::GifOptions;
use crate::ffi;

/// 把 `input_video` 编码为 GIF 写到 `output`。
///
/// 参数越界时在打开任何文件之前返回 [`Error::InvalidOptions`]；链接的 FFmpeg
/// 缺少 `gif` 编码器或所需滤镜时返回 [`Error::Missing`]。
pub fn encode_gif(input_video: &Path, output: &Path, opts: &GifOptions) -> Result<()> {
    opts.validate().map_err(|error| Error::InvalidOptions {
        detail: error.to_string(),
    })?;
    ffi::init();
    let codec = encoder::find_by_name("gif").ok_or_else(|| Error::Missing {
        what: "encoder",
        name: "gif".to_owned(),
    })?;

    // 第一遍：调色板。
    let palette = {
        let mut reader = Reader::open(input_video)?;
        let spec = format!("[in]{}[out]", opts.palettegen_filter());
        let mut graph = build(&reader, &spec, None)?;
        let mut palette = None;
        run(&mut reader, &mut graph, |frame| {
            if palette.is_none() {
                palette = Some(frame);
            }
            Ok(())
        })?;
        palette.ok_or_else(|| Error::Ffi {
            operation: "palettegen",
            path: input_video.to_path_buf(),
            detail: "the filter produced no palette".to_owned(),
        })?
    };

    // 第二遍：按调色板编码。CLI 里调色板是第二个输入 `[1:v]`，这里是名为
    // `pal` 的源。
    let mut reader = Reader::open(input_video)?;
    let spec = format!(
        "[in]{}[out]",
        opts.paletteuse_filter().replace("[1:v]", "[pal]")
    );
    let mut graph = build(&reader, &spec, Some(&palette))?;
    {
        let mut pal = palette.clone();
        pal.set_pts(Some(0));
        let mut context = graph.get("pal").expect("the graph has a palette source");
        let mut source = context.source();
        source
            .add(&pal)
            .map_err(|error| ffi::fail("filter", output, error))?;
        source
            .flush()
            .map_err(|error| ffi::fail("filter", output, error))?;
    }

    let time_base = ffmpeg::Rational::new(1, opts.fps as i32);
    let mut sink: Option<GifSink> = None;
    run(&mut reader, &mut graph, |frame| {
        if sink.is_none() {
            sink = Some(GifSink::open(
                codec,
                output,
                &frame,
                time_base,
                opts.loop_value(),
            )?);
        }
        sink.as_mut().expect("just opened").encode(Some(&frame))
    })?;
    match sink {
        Some(mut sink) => {
            sink.encode(None)?;
            sink.output
                .write_trailer()
                .map_err(|error| ffi::fail("write trailer", output, error))
        }
        None => Err(Error::NoFrame {
            path: input_video.to_path_buf(),
        }),
    }
}

/// 一个打开的视频输入：解封装器与软件解码器。
struct Reader {
    path: PathBuf,
    input: ffmpeg::format::context::Input,
    stream: usize,
    time_base: ffmpeg::Rational,
    aspect: ffmpeg::Rational,
    decoder: ffmpeg::codec::decoder::Video,
    held: crate::feed::Held<ffmpeg::Packet>,
}

impl Reader {
    fn open(path: &Path) -> Result<Self> {
        let input = ffmpeg::format::input(path).map_err(|error| ffi::fail("open", path, error))?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| Error::NoVideoStream {
                path: path.to_path_buf(),
            })?;
        let index = stream.index();
        let time_base = stream.time_base();
        let decoder = ffmpeg::codec::Context::from_parameters(stream.parameters())
            .and_then(|context| context.decoder().video())
            .map_err(|error| ffi::fail("open decoder", path, error))?;
        let aspect = decoder.aspect_ratio();
        let aspect = if aspect.numerator() <= 0 || aspect.denominator() <= 0 {
            ffmpeg::Rational::new(1, 1)
        } else {
            aspect
        };
        Ok(Self {
            path: path.to_path_buf(),
            input,
            stream: index,
            time_base,
            aspect,
            decoder,
            held: None,
        })
    }

    /// 下一帧解码画面，文件结束时为 `None`。
    fn next(&mut self) -> Result<Option<Video>> {
        let (input, stream) = (&mut self.input, self.stream);
        let read = || crate::feed::read_packet(input, stream);
        crate::feed::next_frame(&mut self.decoder, &mut self.held, read)
            .map_err(|stall| crate::feed::fail(stall, &self.path))
    }
}

/// `spec` 的滤镜图：视频从 `in` 进、`out` 出；给了 `palette` 时另有名为
/// `pal` 的源，按调色板帧的格式与尺寸描述。
fn build(reader: &Reader, spec: &str, palette: Option<&Video>) -> Result<filter::Graph> {
    let path = reader.path.as_path();
    let missing = |name: &str| Error::Missing {
        what: "filter",
        name: name.to_owned(),
    };
    let mut graph = filter::Graph::new();
    // 单线程：与 treat.rs 同理，FFmpeg 8 的滤镜线程池在 Windows 上会死锁。
    // SAFETY: 图刚分配、尚无滤镜，FFmpeg 正是在此时读取该字段。
    unsafe {
        (*graph.as_mut_ptr()).nb_threads = 1;
    }
    let buffer = filter::find("buffer").ok_or_else(|| missing("buffer"))?;
    let tb = reader.time_base;
    let args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
        reader.decoder.width(),
        reader.decoder.height(),
        Into::<ffmpeg::sys::AVPixelFormat>::into(reader.decoder.format()).0,
        tb.numerator(),
        tb.denominator().max(1),
        reader.aspect.numerator(),
        reader.aspect.denominator(),
    );
    graph
        .add(&buffer, "in", &args)
        .map_err(|error| ffi::fail("buffer source", path, error))?;
    if let Some(palette) = palette {
        let args = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect=1/1",
            palette.width(),
            palette.height(),
            Into::<ffmpeg::sys::AVPixelFormat>::into(palette.format()).0,
            tb.numerator(),
            tb.denominator().max(1),
        );
        graph
            .add(&buffer, "pal", &args)
            .map_err(|error| ffi::fail("buffer source", path, error))?;
    }
    graph
        .add(
            &filter::find("buffersink").ok_or_else(|| missing("buffersink"))?,
            "out",
            "",
        )
        .map_err(|error| ffi::fail("buffer sink", path, error))?;
    let mut parser = graph
        .output("in", 0)
        .map_err(|error| ffi::fail("filter graph", path, error))?;
    if palette.is_some() {
        parser = parser
            .output("pal", 0)
            .map_err(|error| ffi::fail("filter graph", path, error))?;
    }
    parser
        .input("out", 0)
        .and_then(|parser| parser.parse(spec))
        .map_err(|error| ffi::fail("filter graph", path, error))?;
    graph
        .validate()
        .map_err(|error| ffi::fail("filter graph", path, error))?;
    Ok(graph)
}

/// 把 `reader` 的每一帧送进 `graph` 的 `in`，`out` 出来的每一帧交给 `take`，
/// 直到图在输入结束后吐完为止。
fn run(
    reader: &mut Reader,
    graph: &mut filter::Graph,
    mut take: impl FnMut(Video) -> Result<()>,
) -> Result<()> {
    let path = reader.path.clone();
    let mut drain = |graph: &mut filter::Graph| -> Result<bool> {
        loop {
            let mut filtered = Video::empty();
            let pulled = {
                let mut context = graph.get("out").expect("the graph has an output");
                context.sink().frame(&mut filtered)
            };
            match pulled {
                Ok(()) => take(filtered)?,
                Err(ffmpeg::Error::Eof) => return Ok(true),
                Err(error) if ffi::is_again(&error) => return Ok(false),
                Err(error) => return Err(ffi::fail("filter output", &path, error)),
            }
        }
    };
    while let Some(mut frame) = reader.next()? {
        let at = frame.timestamp();
        frame.set_pts(at);
        {
            let mut context = graph.get("in").expect("the graph has an input");
            context
                .source()
                .add(&frame)
                .map_err(|error| ffi::fail("filter", &reader.path, error))?;
        }
        if drain(graph)? {
            return Ok(());
        }
    }
    {
        let mut context = graph.get("in").expect("the graph has an input");
        context
            .source()
            .flush()
            .map_err(|error| ffi::fail("filter", &reader.path, error))?;
    }
    drain(graph).map(|_| ())
}

/// 打开的 `gif` 编码器与它写入的文件。
struct GifSink {
    path: PathBuf,
    encoder: encoder::video::Encoder,
    output: ffmpeg::format::context::Output,
    encoder_time_base: ffmpeg::Rational,
    stream_time_base: ffmpeg::Rational,
}

impl GifSink {
    /// 按第一帧 PAL8 画面的尺寸打开编码器并写文件头，`loop_value` 设给
    /// GIF muxer 的 `loop` 选项。
    fn open(
        codec: ffmpeg::Codec,
        path: &Path,
        first: &Video,
        time_base: ffmpeg::Rational,
        loop_value: i32,
    ) -> Result<Self> {
        let mut output = ffmpeg::format::output_as(path, "gif")
            .map_err(|error| ffi::fail("create", path, error))?;
        let mut video = ffmpeg::codec::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|error| ffi::fail("gif encoder", path, error))?;
        video.set_width(first.width());
        video.set_height(first.height());
        video.set_format(Pixel::PAL8);
        video.set_time_base(time_base);
        video.set_frame_rate(Some(ffmpeg::Rational::new(
            time_base.denominator(),
            time_base.numerator(),
        )));
        let encoder = video
            .open()
            .map_err(|error| ffi::fail("open encoder", path, error))?;
        {
            let mut stream = output
                .add_stream(codec)
                .map_err(|error| ffi::fail("add stream", path, error))?;
            stream.copy_parameters_from_context(&encoder);
            stream.set_time_base(time_base);
        }
        let mut options = ffmpeg::Dictionary::new();
        let loop_text = loop_value.to_string();
        options.set("loop", &loop_text);
        output
            .write_header_with(options)
            .map_err(|error| ffi::fail("write header", path, error))?;
        let stream_time_base = output
            .stream(0)
            .map(|stream| stream.time_base())
            .unwrap_or(time_base);
        Ok(Self {
            path: path.to_path_buf(),
            encoder,
            output,
            encoder_time_base: time_base,
            stream_time_base,
        })
    }

    /// 编码一帧并写出编码器给出的包；`None` 冲刷编码器。
    fn encode(&mut self, frame: Option<&Video>) -> Result<()> {
        let path = self.path.as_path();
        match frame {
            Some(frame) => self.encoder.send_frame(frame),
            None => self.encoder.send_eof(),
        }
        .map_err(|error| ffi::fail("encode", path, error))?;
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(0);
                    packet.rescale_ts(self.encoder_time_base, self.stream_time_base);
                    packet
                        .write_interleaved(&mut self.output)
                        .map_err(|error| ffi::fail("write packet", path, error))?;
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(error) if ffi::is_again(&error) => return Ok(()),
                Err(error) => return Err(ffi::fail("encode", path, error)),
            }
        }
    }
}
