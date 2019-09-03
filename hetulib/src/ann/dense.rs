extern crate rand;
use rand::prelude::*;
use rand::Rng;

use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;
use crate::base::Node;

pub struct Dense {
    pub node: Node,
    pub units: usize,
    pub input_size: usize,
    pub weights: Mat,
    pub grads: Mat,
    pub bias: Vector,
    pub grads_bias: Vector,
    pub use_bais: bool,
}

impl Dense {
    pub fn new(units: usize, input: usize, use_bais: bool) -> Dense {
        let mut rng = rand::thread_rng();
        let weights = (0..units)
            .map(|_| (0..input).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let mut bias = vec![];
        if use_bais {
            bias = (0..units).map(|_| rng.gen::<f32>()).collect();
        }
        Dense {
            units: units,
            input_size: input,
            node: Node {
                input: vec![],
                output: vec![],
            },
            weights: weights,
            grads: vec![vec![0.0f32; input]; units],
            bias: bias,
            grads_bias: vec![0.0f32; units],
            use_bais: use_bais,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.node.input = input.concat();
        let rs: Vector = (0..self.units)
            .map(|i| {
                if self.use_bais {
                    vec_sum_mutiplier(&self.weights[i], &self.node.input) + self.bias[i]
                } else {
                    vec_sum_mutiplier(&self.weights[i], &self.node.input)
                }
            })
            .collect();
        self.node.output = rs.clone();
        vec![rs]
    }
    fn backward(&mut self, up_grads: &Mat) -> Mat {
        for ug in up_grads.iter() {
            for (j, grads) in ug.iter().enumerate() {
                for (k, input) in self.node.input.iter().enumerate() {
                    self.grads[j][k] = grads * input;
                }
                self.grads_bias[j] = *grads;
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
        for bi in 0..self.bias.len() {
            self.bias[bi] = self.bias[bi] - lamda * self.grads_bias[bi]
        }
    }

    fn clear(&mut self) {
        self.grads = vec![vec![0.0f32; self.input_size]; self.units];
    }
}