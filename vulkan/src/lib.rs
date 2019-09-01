#![crate_type = "lib"]
#![crate_name = "hetu_vulkan"]

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions,Queue};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;
use vulkano::device::Features;
use std::sync::Arc;


struct VulkanEnv {
    device: Arc<Device>,
    queue: Arc<Queue>,
    
}

impl VulkanEnv {
    pub fn new() -> Self {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .expect("failed to create instance");

        let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

        let queue_family = physical.queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a compute queue family");

        let (device, mut queues) = {
            Device::new(physical, &Features::none(), &DeviceExtensions::none(),
                        [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
        };

        let queue = queues.next().unwrap();
        

        VulkanEnv {
            device: device,
            queue: queue,
        }
    }
}

mod cs {
                vulkano_shaders::shader!{
                    ty: "compute",
                    src: "
    #version 450

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(set = 0, binding = 0) buffer Data {
        float data[];
    } data;

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        data.data[idx] = 1.f/(1.f+exp(-data.data[idx]));
    }"
                }
            }

struct SigmoidOperator {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader: cs::Shader,
    buffer:Arc<CpuAccessibleBuffer<[f32]>>,
}

impl SigmoidOperator {
    pub fn new(env:&VulkanEnv)->Self {
        let shader = cs::Shader::load(env.device.clone())
            .expect("failed to create shader module");
        let memory = vec![0f32;65536usize];
        let buffer:Arc<CpuAccessibleBuffer<[f32]>> = CpuAccessibleBuffer::from_iter(env.device.clone(), BufferUsage::all(),
                                                    memory.into_iter()).expect("failed to create buffer");
        SigmoidOperator {
            device:env.device.clone(),
            queue: env.queue.clone(),
            shader: shader,
            buffer: buffer,
        }
    }
    pub fn compute(&mut self, data : Vec<f32>) ->Vec<f32> {
        let size = data.len();
        let compute_pipeline = Arc::new(ComputePipeline::new(self.device.clone(), &self.shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"));
        
        {
            let mut content = self.buffer.write().unwrap();

            for (i, v) in data.into_iter().enumerate() {
                content[i] = v;
            }
        }
        let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_buffer(self.buffer.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family()).unwrap()
            .dispatch([size as u32, 1, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
            .build().unwrap();
                
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        let mut result:Vec<f32> = vec!();
        {
            let content = self.buffer.read().unwrap();
        
            for n in 0 .. size {
                result.push(content[n]);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use rand::Rng;
   

    #[test]
    pub fn test_sigmoid() {

            let env = VulkanEnv::new();
            let mut sigmoid = SigmoidOperator::new(&env);
            let mut rng = rand::thread_rng();
            let data:Vec<f32> = (0..1024).map(|_| rng.gen::<f32>()).collect(); 
            let data_buffer_content = sigmoid.compute(data.clone());

            for n in 0 .. 1024 {
                assert!(data_buffer_content[n]-1f32/(1f32+(-data[n]).exp())<0.00001||
                        data_buffer_content[n]-1f32/(1f32+(-data[n]).exp())>-0.00001);
            }

            let data1:Vec<f32> = (0..1024).map(|_| rng.gen::<f32>()).collect(); 
            let data_buffer_content = sigmoid.compute(data1.clone());

            for n in 0 .. 1024 {
                assert!(data_buffer_content[n]-1f32/(1f32+(-data1[n]).exp())<0.00001||
                        data_buffer_content[n]-1f32/(1f32+(-data1[n]).exp())>-0.00001);
            }
    }
}


