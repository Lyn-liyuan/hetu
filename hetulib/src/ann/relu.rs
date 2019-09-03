use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;
use crate::base::Node;

pub struct ReLuLayer {
    pub node: Node,
}
impl ReLuLayer {
    pub fn new() -> Self {
        ReLuLayer {
            node: Node {
                input: vec![],
                output: vec![],
            },
        }
    }
}

impl Layer for ReLuLayer {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.node.input = input.concat();
        let rs = relu(&self.node.input);
        self.node.output = rs.clone();
        vec![rs]
    }

    fn backward(&mut self, grads: &Mat) -> Mat {
        let mut rs: Mat = vec![vec![0.0f32; 1usize]; self.node.input.len()];
        let diff = derivative_relu(&self.node.input);
        for (i, d) in diff.iter().enumerate() {
            for cell in grads.iter() {
                rs[i][0] = rs[i][0] + cell[i] * d;
            }
        }
        rs
    }
    fn update_weights(&mut self, _: f32) {}
    fn clear(&mut self) {}
}
