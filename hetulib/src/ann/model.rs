use crate::base::*;

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        &self.layers.push(layer);
    }
    fn forward(&mut self, input: &Mat, training: bool) -> Vector {
        let mut pre: Mat = self.layers[0].forward(input, training);
        for i in 1..self.layers.len() {
            pre = self.layers[i].forward(&pre, training);
        }
        pre.concat()
    }
    pub fn predict(&mut self, input: &Mat) -> Vector {
        self.forward(input, false)
    }

    pub fn fit(&mut self, data: &Vec<Mat>, label: &Mat, lamda: f32) -> f32 {
        let last_label = label;
        let mut out: Mat = Vec::new();
        for i in 0..data.len() {
            out.push(self.forward(&data[i], true));
            let true_value = &last_label[i];
            let mut g: Mat = vec![true_value.clone()];
            for j in (0..self.layers.len()).rev() {
                g = self.layers[j].backward(&g);
            }
            for layer in self.layers.iter_mut() {
                layer.update_weights(lamda)
            }
            for layer in self.layers.iter_mut() {
                layer.clear();
            }
        }
        out.clear();
        for row in data.iter() {
            out.push(self.predict(row));
        }
        softmax_acc(&out, label)
    }
}