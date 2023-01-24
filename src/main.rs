

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("include/kernel.hpp");

        unsafe fn sum(array: *mut f32, len: usize) -> f32;
    }
}

fn main() {
    let mut array: Vec<f32> = (1..10001).map(|v| v as f32).collect();

    
    let expected_sum = array.len() as f32 * (array.len() + 1) as f32 / 2.0;

    let start = std::time::Instant::now();

    let computed: f32 = unsafe { ffi::sum(array.as_mut_ptr(), array.len()) };

    let end = std::time::Instant::now();

    println!("expected {}; got {}", expected_sum, computed);
    println!("elapsed: {}ms", end.duration_since(start).as_millis());
}
