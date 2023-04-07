#![feature(test)]

extern crate test;
use test::Bencher;

#[bench]
fn scrypting_bench(b: &mut Bencher) {
    // Each iteration generates labels * 16B
    let labels = 256 * 16; // must be multiple of 128 for now

    let mut scrypter = scrypt_ocl::Scrypter::new(512);
    b.iter(|| {
        test::black_box(scrypter.scrypt(test::black_box(labels)).unwrap());
    });
}
