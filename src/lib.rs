extern crate ocl;

use ocl::{
    enums::{DeviceInfo, KernelWorkGroupInfo},
    Buffer, Kernel, MemFlags, ProQue, SpatialDims,
};

#[derive(Debug)]
pub struct Scrypter {
    kernel: Kernel,
    output: Buffer<u8>,
    global_work_size: usize,
}

const LABEL_SIZE: usize = 16;

impl Scrypter {
    pub fn new(n: usize) -> ocl::Result<Self> {
        let src = include_str!("scrypt-jane.cl");
        let mut pro_que = ProQue::builder().src(src).build()?;

        println!("Device name: {}", pro_que.device().info(DeviceInfo::Name)?);
        println!(
            "Max compute size: {}",
            pro_que.device().info(DeviceInfo::MaxComputeUnits)?
        );
        let max_wg_size = pro_que.device().max_wg_size()?;
        println!("Device max_wg_size: {max_wg_size}");
        let global_work_size = max_wg_size * 4;

        pro_que.set_dims(SpatialDims::One(global_work_size));

        let input = Buffer::<u32>::builder()
            .len(8)
            .flags(MemFlags::new().read_only())
            .queue(pro_que.queue().clone())
            .build()?;

        let output = Buffer::<u8>::builder()
            .len(global_work_size * LABEL_SIZE)
            .flags(MemFlags::new().write_only())
            .queue(pro_que.queue().clone())
            .build()?;

        let lookup_gap = 2;
        let pad_size = global_work_size * 4 * 8 * (n / lookup_gap);

        let padcache = Buffer::<u32>::builder()
            .len(pad_size)
            .flags(MemFlags::new().host_no_access())
            .queue(pro_que.queue().clone())
            .build()?;

        let kernel = pro_que
            .kernel_builder("scrypt")
            .arg(n as u32)
            .arg(0)
            .arg(&input)
            .arg(&output)
            .arg(&padcache)
            .global_work_size(SpatialDims::One(global_work_size))
            .local_work_size(128)
            .build()?;

        let preferred_wg_size_mult = kernel.wg_info(
            pro_que.device(),
            KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple,
        )?;

        println!("Kernel PreferredWorkGroupSizeMultiple: {preferred_wg_size_mult}");

        Ok(Self {
            kernel,
            output,
            global_work_size,
        })
    }

    pub fn scrypt(&mut self, labels: usize) -> ocl::Result<Vec<u8>> {
        let mut vec = vec![0u8; labels * LABEL_SIZE];
        for (id, chunk) in vec
            .chunks_mut(self.global_work_size * LABEL_SIZE)
            .enumerate()
        {
            let start_index = self.global_work_size * id;
            self.kernel.set_arg(1, start_index as u32)?;
            unsafe {
                self.kernel.enq()?;
            }

            self.output.read(chunk).enq()?;
        }
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_scrypt() {
        let mut scrypter = Scrypter::new(8192).unwrap();

        let labels = scrypter.scrypt(2 * 1024).unwrap();
        let mut file = std::fs::File::create("labels.bin").unwrap();
        file.write_all(&labels).unwrap();

        for i in 0..4 {
            println!(
                "Output[{:04X}]: {:02x?}",
                i * 16,
                &labels[i * 16..(i + 1) * 16],
            );
        }

        for i in 1020..1030 {
            println!(
                "Output[{:04X}]: {:02x?}",
                i * 16,
                &labels[i * 16..(i + 1) * 16],
            );
        }
    }
}
