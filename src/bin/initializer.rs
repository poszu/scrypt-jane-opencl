use std::io::Write;

fn main() {
    let label_count = 256 * 1024;

    let mut scrypter = scrypt_ocl::Scrypter::new(8192).unwrap();

    let labels = scrypter.scrypt(label_count).unwrap();
    let mut file = std::fs::File::create("labels.bin").unwrap();
    file.write_all(&labels).unwrap();
}
