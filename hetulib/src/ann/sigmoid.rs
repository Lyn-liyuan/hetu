use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;
use crate::base::Node;

pub struct SigmoidLayer {
    pub node: Node,
}
impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer {
            node: Node {
                input: vec![],
                output: vec![],
            },
        }
    }
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.node.input = input.concat();
        let rs = vec_sigmoid(&self.node.input);
        self.node.output = rs.clone();
        vec![rs]
    }

    fn backward(&mut self, grads: &Mat) -> Mat {
        let mut rs: Mat = vec![vec![0.0f32; 1usize]; self.node.input.len()];
        for (i, out) in self.node.output.iter().enumerate() {
            for cell in grads.iter() {
                rs[i][0] = rs[i][0] + cell[i] * out * (1.0f32 - out);
            }
        }
        rs
    }
    fn update_weights(&mut self, _: f32) {}
    fn clear(&mut self) {}
}