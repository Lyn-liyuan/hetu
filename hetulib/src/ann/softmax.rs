use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;
use crate::base::Node;

pub struct SoftmaxLayer {
    pub node: Node,
}

impl SoftmaxLayer {
    pub fn new() -> Self {
        SoftmaxLayer {
            node: Node {
                input: vec![],
                output: vec![],
            },
        }
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.node.input = input.concat();
        let rs = softmax(&self.node.input);
        self.node.output = rs.clone();
        vec![rs]
    }
    fn backward(&mut self, grads: &Mat) -> Mat {
        grads
            .iter()
            .map(|v| derivative_softmax_cross_entropy(&self.node.output, &v))
            .collect()
    }
    fn update_weights(&mut self, _: f32) {}
    fn clear(&mut self) {}
}
