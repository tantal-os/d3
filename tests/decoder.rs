use d3::lm::Model;
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const WIDTH: usize = 64;
const VOCAB: [&str; 8] = ["<pad>", "<unk>", "<en>", "\n", " ", "a", "b", "c"];
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "d3-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let fixture = Self(dir);
        fixture.write("config.json", r#"{"layers":1}"#);
        fixture.write("vocab.json", serde_json::to_vec(&VOCAB).unwrap());

        let mut index = String::new();
        let mut tensor = |name: &str, shape: &[usize], values: Vec<f32>| {
            index.push_str(name);
            for dim in shape {
                index.push_str(&format!(" {dim}"));
            }
            index.push('\n');
            fixture.write(
                &format!("{name}.bin"),
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        };
        let embedding = [0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 1.5]
            .into_iter()
            .flat_map(|v| {
                let mut row = [0.0; WIDTH];
                row[0] = v;
                row[1] = -v;
                row
            })
            .collect();
        tensor("emb.weight", &[VOCAB.len(), WIDTH], embedding);
        for norm in ["ln", "layers.0.ln1", "layers.0.ln2"] {
            tensor(&format!("{norm}.weight"), &[WIDTH], vec![1.0; WIDTH]);
            tensor(&format!("{norm}.bias"), &[WIDTH], vec![0.0; WIDTH]);
        }
        for dense in ["q", "k", "v", "o", "up", "down"] {
            tensor(
                &format!("layers.0.{dense}.weight"),
                &[WIDTH, WIDTH],
                vec![0.0; WIDTH * WIDTH],
            );
            tensor(
                &format!("layers.0.{dense}.bias"),
                &[WIDTH],
                vec![0.0; WIDTH],
            );
        }
        fixture.write("index.txt", index);
        fixture.write("lexicon.tsv", "a\t100\nb\t100\nc\t100\n");
        fixture.write(
            "layouts.json",
            serde_json::to_vec(&json!([{
                "lang": "en",
                "width": 100,
                "keys": [[97, 5, 5, 10, 10], [98, 6, 5, 10, 10], [99, 7, 5, 10, 10]]
            }]))
            .unwrap(),
        );
        let offset = json!({"mean": [0.0, 0.0], "cov": [[1.0, 0.0], [0.0, 1.0]]});
        fixture.write(
            "touch.json",
            serde_json::to_vec(&json!({
                "keyboard_px": [100, 100],
                "key_px": {"a": [10, 10], "SPACE": [10, 10]},
                "all_letters": offset,
                "keys": {"SPACE": offset}
            }))
            .unwrap(),
        );
        fixture.write(
            "taps.jsonl",
            concat!(
                "{\"taps\":[[120,10,10],[32,-1,-1]]}\n",
                "{\"taps\":[[98,-1,-1],[32,-1,-1]]}\n"
            ),
        );
        fixture
    }

    fn write(&self, name: &str, data: impl AsRef<[u8]>) {
        fs::write(self.0.join(name), data).unwrap();
    }

    fn load(&self) -> Result<Model, String> {
        Model::load(self.0.to_str().unwrap())
    }

    fn decode(&self, runtime: &Path) -> Output {
        Command::new(env!("CARGO_BIN_EXE_decoder"))
            .current_dir(&self.0)
            .arg("--libs")
            .arg(runtime)
            .args([
                "--model",
                ".",
                "--layouts",
                "layouts.json",
                "--lang",
                "en",
                "--lexicon",
                "lexicon.tsv",
                "--touch",
                "touch.json",
                "--taps",
                "taps.jsonl",
                "--out",
                "decoded.jsonl",
                "--beam",
                "4",
                "--ctx",
                "16",
                "--suffix",
                "4",
                "--preds",
                "0",
            ])
            .output()
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_dir_all(&self.0) {
            eprintln!("{}: {e}", self.0.display());
        }
    }
}

#[test]
fn reads_typed_model_files_and_rejects_malformed_assets() {
    let fixture = Fixture::new();
    assert_eq!(fixture.load().unwrap().vocabulary(), VOCAB);
    for invalid in ["{}", r#"{"layers":"1"}"#, r#"{"layers":"#] {
        fixture.write("config.json", invalid);
        assert!(fixture.load().err().unwrap().contains("config.json"));
    }
    fixture.write("config.json", r#"{"layers":1}"#);
    fixture.write("vocab.json", "[0]");
    assert!(fixture.load().err().unwrap().contains("vocab.json"));
    fixture.write("vocab.json", r#"["a","b"]"#);
    assert!(fixture.load().err().unwrap().contains("vocabulary"));
    fixture.write("vocab.json", serde_json::to_vec(&VOCAB).unwrap());
    fixture.write("emb.weight.bin", [0; 3]);
    assert!(fixture.load().is_err());
    let result = fixture.decode(&fixture.0);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("emb.weight"));
}

#[test]
fn reports_a_missing_runtime() {
    let fixture = Fixture::new();
    let result = fixture.decode(&fixture.0);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("LiteRT:"));
}

#[test]
fn orders_candidates_and_commits_on_space() {
    let Some(runtime) = std::env::var_os("TEST_LITERT_DIR") else {
        eprintln!("Set TEST_LITERT_DIR to run the LiteRT inference test");
        return;
    };
    let fixture = Fixture::new();
    let result = fixture.decode(&PathBuf::from(runtime));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let decoded = fs::read_to_string(fixture.0.join("decoded.jsonl")).unwrap();
    let results: Vec<Value> = decoded
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["words"][0]["cands"], json!(["a", "b", "c"]));
    assert_eq!(results[0]["words"][0]["commit"], "a");
    assert_eq!(results[0]["out"], "a");
    assert_eq!(results[1]["words"][0]["commit"], "b");
    assert_eq!(results[1]["out"], "b");

    fixture.write("taps.jsonl", r#"{"taps":[["sensitive text",10,10]]}"#);
    let result = fixture.decode(&PathBuf::from(std::env::var_os("TEST_LITERT_DIR").unwrap()));
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("Invalid taps document"));
    assert!(!error.contains("sensitive text"));
}

#[test]
fn exported_graphs_load_in_the_keyboard_session() {
    use d3::keyboard::{Key, ModelAssets, Session};
    use npusvc_sdk::lrt::{CPU, Lrt};

    let Some(runtime) = std::env::var_os("TEST_LITERT_DIR") else {
        eprintln!("Set TEST_LITERT_DIR to run the keyboard inference test");
        return;
    };
    let fixture = Fixture::new();
    fixture.write("lexicon.tsv", "a\t100\nb\t100\nc\t100\nuncommon\t2\n");
    let result = Command::new(env!("CARGO_BIN_EXE_decoder"))
        .current_dir(&fixture.0)
        .args([
            "--model",
            ".",
            "--export-keyboard",
            "keyboard",
            "--lexicon",
            "lexicon.tsv",
            "--ctx",
            "16",
            "--wq8",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let out = fixture.0.join("keyboard");
    let libs = runtime.to_str().unwrap();
    let runtime = Lrt::new(libs, libs, None).unwrap();
    let compile = |name| {
        runtime
            .compile(out.join(name).to_str().unwrap(), CPU)
            .unwrap()
    };
    let read = |name| fs::read(out.join(name)).unwrap();
    let config: Value = serde_json::from_slice(&read("config.json")).unwrap();
    assert_eq!(config, json!({"layers": 1, "ctx": 16}));
    let lexicon = read("lexicon.tsv");
    assert_eq!(lexicon, b"a\t100\nb\t100\nc\t100\n");
    let mut session = Session::open(
        compile("step.tflite"),
        compile("root.tflite"),
        ModelAssets {
            vocabulary: &read("vocab.json"),
            embeddings: &read("emb.bin"),
            lexicon: &lexicon,
            language: "en",
            layers: 1,
            context: 16,
        },
    )
    .unwrap();
    let keys = ['a', 'b', 'c']
        .into_iter()
        .enumerate()
        .map(|(i, code)| Key {
            code,
            cx: 10.0 + i as f32,
            cy: 10.0,
        })
        .collect();
    session
        .layout(keys, Default::default(), 100.0, None)
        .unwrap();
    let taps = [('x', 10, 10)];
    let suggestions = session.suggest("", &taps).unwrap().unwrap();
    assert_eq!(suggestions.cands, ["a", "b", "c"]);
    assert_eq!(session.commit("", &taps, ' ').unwrap(), "a");
    assert!(session.predict("a ", || true).unwrap().is_empty());
}
