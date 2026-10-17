// openvoice-server: the child process Mutelink starts for English speech in a reference voice (see engine.rs, protocol.rs).
// Commands (the request header's "cmd"):
//   load            {models_dir, data_dir, ort_dll, cuda}                    -> {ok, speakers}
//   register_voice  {out}  body: a WAV file (any rate, mono or stereo)       -> {ok}      writes the voice file `out`
//   synthesize      {text, voice, speaker?, tau?, speed?, ...}               -> one {ok, more: true, ...} per piece of text, body a
//                   16-bit mono WAV at 22.05 kHz, then {ok, more: false}
// Anything printed for a human goes to stderr: stdout carries the frames only.

use anyhow::{anyhow, bail, Context, Result};
use openvoice_server::dsp::CONVERTER_RATE;
use openvoice_server::engine::{Engine, EngineConfig, Params};
use openvoice_server::protocol::{read_frame, write_frame};
use openvoice_server::voice;
use serde_json::{json, Value};
use std::io::{self, BufReader, BufWriter, Write};

fn text_param<'a>(header: &'a Value, key: &str) -> Result<&'a str> {
    header[key].as_str().ok_or_else(|| anyhow!("`{key}` is missing"))
}

fn f32_param(header: &Value, key: &str, default: f32) -> f32 {
    header[key].as_f64().map(|v| v as f32).unwrap_or(default)
}

fn params_from(header: &Value) -> Params {
    let d = Params::default();
    Params {
        speaker: header["speaker"].as_str().map(str::to_string).unwrap_or(d.speaker),
        sdp_ratio: f32_param(header, "sdp_ratio", d.sdp_ratio),
        noise_scale: f32_param(header, "noise_scale", d.noise_scale),
        noise_scale_w: f32_param(header, "noise_scale_w", d.noise_scale_w),
        speed: f32_param(header, "speed", d.speed),
        tau: f32_param(header, "tau", d.tau),
    }
}

fn handle(engine: &mut Option<Engine>, header: &Value, body: &[u8], out: &mut impl Write) -> Result<()> {
    match text_param(header, "cmd")? {
        "load" => {
            let config = EngineConfig {
                models_dir: text_param(header, "models_dir")?.into(),
                data_dir: text_param(header, "data_dir")?.into(),
                ort_dll: text_param(header, "ort_dll")?.into(),
                use_cuda: header["cuda"].as_bool().unwrap_or(true),
            };
            let mut loaded = Engine::load(&config)?;
            loaded.warm_up()?;
            let speakers = loaded.speakers();
            *engine = Some(loaded);
            write_frame(out, &json!({"ok": true, "speakers": speakers}), &[])?;
        }
        "register_voice" => {
            let engine = engine.as_mut().ok_or_else(|| anyhow!("load first"))?;
            let (audio, rate) = voice::decode_wav(body)?;
            let embedding = engine.embed_voice(&audio, rate)?;
            voice::save(std::path::Path::new(text_param(header, "out")?), &embedding)?;
            write_frame(out, &json!({"ok": true}), &[])?;
        }
        "synthesize" => {
            let engine = engine.as_mut().ok_or_else(|| anyhow!("load first"))?;
            let embedding = voice::load(std::path::Path::new(text_param(header, "voice")?))?;
            let params = params_from(header);
            let pieces = engine.pieces(text_param(header, "text")?);
            if pieces.is_empty() {
                bail!("there is nothing to say in that text");
            }
            for (index, piece) in pieces.iter().enumerate() {
                let audio = engine.synthesize_piece(piece, &embedding, &params).with_context(|| format!("piece {index}: {piece:?}"))?;
                let timings = engine.last_timings.clone();
                write_frame(
                    out,
                    &json!({"ok": true, "more": true, "piece": index, "pieces": pieces.len(), "text": piece, "rate": CONVERTER_RATE,
                            "ms": {"bert": timings.bert, "base": timings.melo, "convert": timings.converter}}),
                    &voice::encode_wav(&audio, CONVERTER_RATE),
                )?;
            }
            write_frame(out, &json!({"ok": true, "more": false}), &[])?;
        }
        other => bail!("unknown command {other:?}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut input = BufReader::new(io::stdin().lock());
    let mut output = BufWriter::new(io::stdout().lock());
    let mut engine: Option<Engine> = None;
    while let Some(frame) = read_frame(&mut input)? {
        // A failed request is an answer, not the end of the process: the app keeps the loaded models. A panic is turned into an answer
        // too (the code is written not to panic, this is the last wall): the engine is still usable, and a process that dies costs the
        // user a reload of several seconds.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&mut engine, &frame.header, &frame.body, &mut output)));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("{error:#}")),
            Err(panic) => {
                let what = panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default();
                Some(format!("internal error: {what}"))
            }
        };
        if let Some(message) = failure {
            eprintln!("openvoice-server: {message}");
            write_frame(&mut output, &json!({"ok": false, "error": message}), &[])?;
        }
    }
    Ok(())
}
