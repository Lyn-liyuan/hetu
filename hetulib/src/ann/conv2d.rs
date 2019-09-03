extern crate rand;
use rand::prelude::*;
use rand::Rng;

use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;

pub struct Conv2d {
    pub input: Mat,
    pub output: Mat,
    pub input_width: usize,
    pub input_height: usize,
    pub kernel_size: usize,
    pub kernal_num: usize,
    pub strides: usize,
    pub padding: usize,
    pub weights: Mat,
    pub grads: Mat,
}
impl Conv2d {
    pub fn new(
        input_width: usize,
        input_height: usize,
        kernel_size: usize,
        kernal_num: usize,
        strides: usize,
        padding: usize,
    ) -> Self {
        let grads: Mat = vec![vec![0.0f32; kernel_size * kernel_size]; kernal_num];
        let mut rng = rand::thread_rng();
        let weights = (0..kernal_num)
            .map(|_| {
                (0..kernel_size * kernel_size)
                    .map(|_| rng.gen::<f32>())
                    .collect()
            })
            .collect();
        Conv2d {
            input: vec![vec![]],
            output: vec![vec![]],
            input_width: input_width,
            input_height: input_height,
            kernel_size: kernel_size,
            kernal_num: kernal_num,
            strides: strides,
            padding: padding,
            weights: weights,
            grads: grads,
        }
    }
}

impl Layer for Conv2d {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.input = input.clone();
        let rs = conv2d(
            input,
            self.input_height,
            self.input_width,
            &self.weights,
            self.kernel_size,
            self.padding,
            self.strides,
        );
        self.output = rs.clone();
        rs
    }

    fn backward(&mut self, up_grads: &Mat) -> Mat {
        for (i, ug) in up_grads.iter().enumerate() {
            for (j, grads) in ug.iter().enumerate() {
                self.grads[i][j] = grads * self.input[i][j];
            }
        }
        self.grads.clone()
    }

    fn update_weights(&mut self, lamda: f32) {
        let z = &self.grads;
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] = self.weights[i][j] - lamda * z[i][j];
            }
        }
    }

    fn clear(&mut self) {
        self.grads = vec![vec![0.0f32; self.kernel_size * self.kernel_size]; self.kernal_num];
    }
}