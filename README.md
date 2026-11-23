# D3

D3 decodes keyboard taps into word suggestions and a word to commit.
The Tantal keyboard calls the Rust session through its JNI adapter. The
`decoder` CLI runs the same engine for research and exports keyboard assets.

```sh
cargo build --locked --release
cargo run --locked --release --bin decoder -- --help
```

The prediction command takes a trained model directory, keyboard layouts,
a language, a lexicon, and recorded taps. Pass the directory containing
`libLiteRt.so` as `--libs`. To run the inference test, set `TEST_LITERT_DIR`
to a directory containing the SDK's pinned Linux LiteRT library.

```sh
cargo test --locked
```

For Android arm64, install the `aarch64-linux-android` Rust target and set
`CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER` to the NDK's
`aarch64-linux-android36-clang` before building.

```sh
cargo build --locked --release --target aarch64-linux-android
```
