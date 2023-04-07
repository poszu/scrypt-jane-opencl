extern crate ocl;
use ocl::{enums::DeviceInfo, Buffer, Kernel, ProQue, SpatialDims};

pub struct Scrypter {
    kernel: Kernel,
    output: Buffer<u32>,
}

impl Scrypter {
    pub fn new(n: u32) -> Self {
        let src = include_str!("scrypt-jane.cl");
        let pro_que = ProQue::builder().src(src).dims(128).build().unwrap();

        println!("Device name: {:?}", pro_que.device().info(DeviceInfo::Name));
        // println!("Device: {}", pro_que.device().to_string());
        println!("Device max_wg_size: {:?}", pro_que.device().max_wg_size());

        let input = Buffer::<u32>::builder()
            .len(8)
            .queue(pro_que.queue().clone())
            .build()
            .unwrap();
        let output = Buffer::<u32>::builder()
            .len(128 * 4)
            .fill_val(0)
            .queue(pro_que.queue().clone())
            .build()
            .unwrap();

        // size_t ipt = (1024 / 1);
        // size_t bufsize = 128 * ipt * cgpu->thread_concurrency;
        let padcache = Buffer::<u8>::builder()
            .len(128 * 1024 * 256)
            .fill_val(0)
            .queue(pro_que.queue().clone())
            .build()
            .unwrap();

        let kernel = pro_que
            .kernel_builder("scrypt")
            .global_work_offset((0, 0, 0))
            .arg(n)
            .arg(&input)
            .arg(&output)
            .arg(&padcache)
            .build()
            .unwrap();

        Self { kernel, output }
    }

    pub fn scrypt(&mut self, labels: usize) -> ocl::Result<Vec<u8>> {
        let mut vec = vec![0u8; labels * 16];
        for id in (0..labels).step_by(128) {
            println!("Enquing with offset {id}");
            unsafe {
                self.kernel
                    .cmd()
                    .global_work_offset(SpatialDims::One(id))
                    .local_work_size(128)
                    .enq()
                    .unwrap();
            }

            let out_slice = bytemuck::cast_slice_mut::<u8, u32>(
                &mut vec.as_mut_slice()[id * 16..(id + 128) * 16],
            );
            self.output.read(out_slice).enq().unwrap();
        }
        Ok(vec)
    }
}

// EXPECTED result...
// [a8, 17, e7, a3, 2d, eb, ae, 46, c2, 93, 61, 0b, 9d, fa, 99, da]
// [dd, 61, 4b, cf, ae, f8, 70, 2a, 19, 12, 00, 45, 8c, e6, c9, df]
// [5f, a9, 29, 07, a2, 03, fa, ac, 15, f2, a3, aa, 97, 46, 63, 0b]
// [ae, e3, 0f, c7, 4c, c9, b2, 75, 48, 8f, e8, cc, a4, f3, 98, 9f]

#[test]
fn test_scrypt() {
    let mut scrypter = Scrypter::new(512);

    let labels = scrypter.scrypt(256).unwrap();
    for i in 0..256 {
        println!(
            "Output[{:04X}]: {:02x?}",
            i * 16,
            &labels[i * 16..(i + 1) * 16],
        );
    }
}
