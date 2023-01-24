extern crate cc;

fn main() {
    cxx_build::bridge("src/main.rs")
        .cuda(true)
        .cudart("shared")
        .file("include/kernel.cu")
        .flag_if_supported("-std=c++17")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .compile("cuda_kernel");


    /* Link CUDA Runtime (libcudart.so) */
    
    // Add link directoryrank analysis
    // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
    // - This should be set by `$LIBRARY_PATH`
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-arg=-lcudart");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=include/kernel.cu");
    println!("cargo:rerun-if-changed=include/kernel.hpp");
}
