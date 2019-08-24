extern crate rand;
#[macro_use]
extern crate hetu;
extern crate structopt;
extern crate quicli;

use rand::seq::SliceRandom;
use rand::thread_rng;
use structopt::StructOpt;
use hetu::ann::*;


include!{"data.rs"}



fn conv2d_single_kernel(input:&Vec<f32>, h:usize, w:usize,weight:&Vec<f32>, kernel_size:usize, padding:usize, strides:usize) -> Vec<f32> {
     if input.len() != (h*w) as usize {
         panic!("input size error, w={}, h={}, but length is {}", w,h,input.len());
     } 
     if weight.len() != (kernel_size*kernel_size) as usize {
         panic!("kernel size error, size={}, but length is {}", kernel_size,input.len());
     }
     let mut rs = vec!();
     let h_max = h - kernel_size + 1 + 2*padding;
     let w_max = w - kernel_size + 1 + 2*padding;
     let new_w = w+2*padding;
     let new_h = h+2*padding;
     
     for hi in (0..h_max).step_by(strides as usize) {
         for wi in (0..w_max).step_by(strides as usize) {
             let base = wi+hi*new_w;
             let mut fp = 0.0f32;
             for kh in 0..kernel_size {
                for kw in 0..kernel_size {
                   let index = base+kw+(kh*new_w);
                   let py = hi+kh;
                   let px = wi+kw;
                   if py >= padding && 
                      px >= padding && 
                      px < w+padding && 
                      py < new_h-padding 
                   {
                       let input_v = input[index - w*padding - py*2*padding-padding]; 
                       fp = fp + input_v*weight[kw+kh*kernel_size];
                   }
                }
             }
             rs.push(fp);
         }
     }
     rs
}

fn conv2d(inputs:&Vec<Vec<f32>>,h:usize, w:usize,weights:&Vec<Vec<f32>>, kernel_size:usize, padding:usize, strides:usize) -> Vec<Vec<f32>> {
   let featurs_len = (h - kernel_size + 1 + 2*padding)/strides*(w - kernel_size + 1 + 2*padding)/strides;
   let mut rs:Vec<Vec<f32>> = Vec::new();
   for cell in weights.iter() {
       let mut fm:Vec<f32> = vec![0.0f32; featurs_len];
       for input in inputs.iter() {
           let channel = conv2d_single_kernel(input, h, w, cell, kernel_size,padding,strides);
           for (i, c) in channel.iter().enumerate() {
               fm[i] = fm[i] + c;
           }
       }
       rs.push(fm);
   }
   rs
}

fn main() {
    let input = vec![
                      10f32,10f32,10f32,0f32,0f32,0f32,
                      10f32,10f32,10f32,0f32,0f32,0f32,
                      10f32,10f32,10f32,0f32,0f32,0f32,
                      10f32,10f32,10f32,0f32,0f32,0f32,
                      10f32,10f32,10f32,0f32,0f32,0f32,
                      10f32,10f32,10f32,0f32,0f32,0f32,
                    ];
    let weight = vec![1f32,0f32,-1f32,1f32,0f32,-1f32,1f32,0f32,-1f32,];
    
    let rs = conv2d_single_kernel(&input, 6, 6, &weight, 3,1, 1);

    println!("{:?}", rs);

    #[derive(StructOpt)]
    struct Cli {
        #[structopt(parse(from_os_str))]
        path: std::path::PathBuf,
    };
    let args = Cli::from_args();
    
    let mut model = Model![
        Dense!(8,4),
        ReLu!(),
        Dense!(3,8),
        Softmax!()
    ];
    let (data, labels) = load_data(args.path.as_path());
    for p in 0..6000 {
        println!("epoch {} loss = {}", p, 1f32 - model.fit(&data, &labels, 0.01f32));
    }
}


