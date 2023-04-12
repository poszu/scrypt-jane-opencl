use std::{io::Write, time};

fn main() {
    let label_count = 256 * 1024;

    let mut scrypter = scrypt_ocl::Scrypter::new(8192).unwrap();

    let now = time::Instant::now();
    let labels = scrypter.scrypt(label_count).unwrap();
    let elapsed = now.elapsed();
    println!(
        "Scrypting {} labels took {} seconds. Speed: {:.0} labels/sec ({:.2} MB/sec)",
        label_count,
        elapsed.as_secs(),
        label_count as f64 / elapsed.as_secs_f64(),
        label_count as f64 * 16.0 / elapsed.as_secs_f64() / 1024.0 / 1024.0
    );

    let mut file = std::fs::File::create("labels.bin").unwrap();
    file.write_all(&labels).unwrap();
}
