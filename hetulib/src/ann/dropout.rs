extern crate rand;
use rand::prelude::*;
use rand::seq::SliceRandom;
use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;
use crate::base::Node;


pub struct Dropout {
    pub p: f32,
    mask: Vector,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p: p, mask: vec![] }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Mat, training: bool) -> Mat {
        if training {
            let vc = input.concat();
            let in_size = vc.len();
            self.mask = vec![0.0f32; in_size];
            for i in 0..((in_size as f32) * (1.0f32 - self.p)) as usize {
                self.mask[i] = 1.0f32
            }
            let mut rng = thread_rng();
            self.mask.shuffle(&mut rng);
            vec![vc
                .iter()
                .enumerate()
                .map(|(i, v)| self.mask[i] * v)
                .collect()]
        } else {
            input.clone()
        }
    }

    fn backward(&mut self, grads: &Mat) -> Mat {
        let mut rs: Mat = vec![vec![]];
        for cell in grads.iter() {
            rs.push(
                cell.iter()
                    .enumerate()
                    .map(|(i, v)| self.mask[i] * v)
                    .collect(),
            )
        }
        rs
    }
    fn update_weights(&mut self, _: f32) {}
    fn clear(&mut self) {}
}