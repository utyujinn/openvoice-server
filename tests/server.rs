// The server process end to end over its stdin / stdout, the way the app uses it: load, register a reference voice, speak a text of
// two sentences (one answer per piece), and survive a bad request. Like tests/engine.rs it needs the models and onnxruntime.dll:
//   OPENVOICE_ORT_DLL=<path>  [OPENVOICE_CUDA=1]  cargo test --release --test server -- --ignored --nocapture

use openvoice_server::protocol::{read_frame, write_frame};
use openvoice_server::voice;
use serde_json::{json, Value};
use std::io::BufReader;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Server {
    fn start() -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_openvoice-server")).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("the server starts");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Server { child, stdin, stdout }
    }

    fn send(&mut self, header: Value, body: &[u8]) {
        write_frame(&mut self.stdin, &header, body).unwrap();
    }

    fn receive(&mut self) -> (Value, Vec<u8>) {
        let frame = read_frame(&mut self.stdout).unwrap().expect("the server answers");
        (frame.header, frame.body)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[ignore = "needs the ONNX models and onnxruntime.dll, see the top of this file"]
fn the_server_loads_registers_a_voice_and_speaks() {
    let mut server = Server::start();

    // before `load` nothing can be done, and that is an answer, not a crash
    server.send(json!({"cmd": "synthesize", "text": "Hi.", "voice": "x"}), &[]);
    let (answer, _) = server.receive();
    assert_eq!(answer["ok"], false);
    server.send(json!({"cmd": "nonsense"}), &[]);
    assert_eq!(server.receive().0["ok"], false);

    server.send(
        json!({"cmd": "load", "models_dir": root().join("models"), "data_dir": root().join("data"), "ort_dll": std::env::var("OPENVOICE_ORT_DLL").unwrap(),
               "cuda": std::env::var("OPENVOICE_CUDA").is_ok_and(|v| v == "1")}),
        &[],
    );
    let (loaded, _) = server.receive();
    assert_eq!(loaded["ok"], true, "{loaded}");
    assert!(loaded["speakers"].as_array().unwrap().iter().any(|s| s == "EN-US"));

    let voice_path = std::env::temp_dir().join(format!("openvoice-server-test-{}.ovvoice", std::process::id()));
    let wav = std::fs::read(root().join("tests/golden/ref_en.wav")).unwrap();
    server.send(json!({"cmd": "register_voice", "out": voice_path}), &wav);
    assert_eq!(server.receive().0["ok"], true);
    let embedding = voice::load(&voice_path).unwrap();
    assert_eq!(embedding.len(), 256);

    // two sentences long enough to be two pieces would need > 256 characters; one short text is one piece
    server.send(json!({"cmd": "synthesize", "text": "Hello, nice to meet you. This is a test of the voice.", "voice": voice_path}), &[]);
    let mut pieces = 0;
    loop {
        let (header, body) = server.receive();
        assert_eq!(header["ok"], true, "{header}");
        if header["more"] == false {
            assert!(body.is_empty());
            break;
        }
        pieces += 1;
        let (audio, rate) = voice::decode_wav(&body).expect("each piece is a WAV file");
        assert_eq!(rate, 22050);
        assert!(audio.len() > 22050 / 2, "a piece of speech is longer than half a second");
        let loudest = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(loudest > 0.05, "the speech is not silent (peak {loudest})");
        println!("piece {}: {:?} {:.1} s, {}", header["piece"], header["text"], audio.len() as f64 / 22050.0, header["ms"]);
    }
    assert!(pieces >= 1);

    // a request that fails does not end the process
    server.send(json!({"cmd": "synthesize", "text": "   ", "voice": voice_path}), &[]);
    assert_eq!(server.receive().0["ok"], false);
    server.send(json!({"cmd": "synthesize", "text": "Still here.", "voice": "no-such-voice.ovvoice"}), &[]);
    assert_eq!(server.receive().0["ok"], false);
    server.send(json!({"cmd": "synthesize", "text": "Still here.", "voice": voice_path}), &[]);
    let (header, _) = server.receive();
    assert_eq!(header["ok"], true);
    let _ = std::fs::remove_file(voice_path);
}
