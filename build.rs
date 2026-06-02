fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_COHERE_MLX").is_ok()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        link_clang_runtime();
    }
}

fn link_clang_runtime() {
    let Ok(output) = std::process::Command::new("clang")
        .args(["--print-file-name", "libclang_rt.osx.a"])
        .output()
    else {
        return;
    };
    let Ok(path) = String::from_utf8(output.stdout) else {
        return;
    };
    let path = path.trim();
    if std::path::Path::new(path).is_file() {
        println!("cargo:rustc-link-arg={path}");
    }
}
