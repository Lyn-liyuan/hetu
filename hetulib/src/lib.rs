#![crate_type = "lib"]
#![crate_name = "hetu"]

pub mod ann {
    extern crate rand;
    use rand::Rng;

    pub fn vec_sum_mutiplier(vc1: &Vec<f32>, vc2: &Vec<f32>) -> f32 {
        if vc1.len() != vc2.len() {
            panic!(
                "vector size is not equals!!! {} and {}",
                vc1.len(),
                vc2.len()
            )
        }
        let mut sum = 0.0f32;
        for i in 0..vc1.len() {
            sum += vc1[i] * vc2[i];
        }
        sum
    }

    fn relu(x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|v| v.max(0.0f32)).collect()
    }

    fn derivative_relu(x: &Vec<f32>) -> Vec<f32> {
        x.iter()
            .map(|i| if *i >= 0.0f32 { 1f32 } else { 0f32 })
            .collect()
    }

    fn sigmoid(x: &f32) -> f32 {
        1.0f32 / (1.0f32 + (-1.0f32 * x).exp())
    }

    fn vec_sigmoid(x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|v| sigmoid(v)).collect()
    }

    fn softmax(x: &Vec<f32>) -> Vec<f32> {
        let sum: f32 = x.iter().map(|v| v.exp()).sum();
        x.iter().map(|v| v.exp() / sum).collect()
    }

    fn softmax_acc(y: &Vec<Vec<f32>>, label: &Vec<Vec<f32>>) -> f32 {
        let mut acc_count = 0.0f32;
        for (i, yi) in y.iter().enumerate() {
            let mut p = 0usize;
            let mut mx = 0.0f32;
            for (j, v) in yi.iter().enumerate() {
                if *v > mx {
                    mx = *v;
                    p = j;
                }
            }
            if label[i][p] == 1.0f32 {
                acc_count = acc_count + 1.0f32;
            }
        }
        acc_count / (label.len() as f32)
    }

    fn derivative_softmax_cross_entropy(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
        if x.len() != y.len() {
            panic!("vector size is not equals!!!")
        }
        x.iter().enumerate().map(|(i, x)| x - y[i]).collect()
    }

    pub struct Node {
        pub input: Vec<f32>,
        pub output: Vec<f32>,
    }

    pub struct Dense {
        pub node: Node,
        pub units: usize,
        pub input_size: usize,
        pub weights: Vec<Vec<f32>>,
        pub grads: Vec<Vec<f32>>,
    }

    impl Dense {
        pub fn new(units: usize, input: usize) -> Dense {
            let grads: Vec<Vec<f32>> = vec![vec![0.0f32; input]; units];
            let mut rng = rand::thread_rng();
            let weights = (0..units)
                .map(|_| (0..input).map(|_| rng.gen::<f32>()).collect())
                .collect();
            Dense {
                units: units,
                input_size: input,
                node: Node {
                    input: vec![],
                    output: vec![],
                },
                weights: weights,
                grads: grads,
            }
        }
    }

    pub trait Layer {
        fn forward(&mut self, input: &Vec<f32>) -> Vec<f32>;
        fn backward(&mut self, grads: &Vec<Vec<f32>>) -> Vec<Vec<f32>>;
        fn update_weights(&mut self, lamda: f32);
        fn clear(&mut self);
    }

    impl Layer for Dense {
        fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
            self.node.input = input.clone();
            let rs: Vec<f32> = (0..self.units)
                .map(|i| vec_sum_mutiplier(&self.weights[i], &input))
                .collect();
            self.node.output = rs.clone();
            rs
        }
        fn backward(&mut self, up_grads: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            for ug in up_grads.iter() {
                for (j, grads) in ug.iter().enumerate() {
                    for (k, input) in self.node.input.iter().enumerate() {
                        self.grads[j][k] = grads * input;
                    }
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
            self.grads = vec![vec![0.0f32; self.input_size]; self.units];
        }
    }

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
        fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
            self.node.input = input.clone();
            let rs = softmax(input);
            self.node.output = rs.clone();
            rs
        }
        fn backward(&mut self, grads: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            grads
                .iter()
                .map(|v| derivative_softmax_cross_entropy(&self.node.output, &v))
                .collect()
        }
        fn update_weights(&mut self, _: f32) {}
        fn clear(&mut self) {}
    }

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
        fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
            self.node.input = input.clone();
            let rs = vec_sigmoid(&input);
            self.node.output = rs.clone();
            rs
        }

        fn backward(&mut self, grads: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            let mut rs: Vec<Vec<f32>> = vec![vec![0.0f32; 1usize]; self.node.input.len()];
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
        fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
            self.node.input = input.clone();
            let rs = relu(input.clone());
            self.node.output = rs.clone();
            rs
        }

        fn backward(&mut self, grads: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            let mut rs: Vec<Vec<f32>> = vec![vec![0.0f32; 1usize]; self.node.input.len()];
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

    pub struct Model {
        pub layers: Vec<Box<Layer>>,
    }

    pub trait NnModel {
        fn add(&mut self, layer: Box<Layer>);
        fn predict(&mut self, input: &Vec<f32>) -> Vec<f32>;
        fn fit(&mut self, data: &Vec<Vec<f32>>, label: &Vec<Vec<f32>>, lamda: f32) -> f32;
    }

    impl Model {
        pub fn new() -> Self {
            Model { layers: Vec::new() }
        }
    }

    impl NnModel for Model {
        fn add(&mut self, layer: Box<Layer>) {
            &self.layers.push(layer);
        }
        fn predict(&mut self, input: &Vec<f32>) -> Vec<f32> {
            let mut pre: Vec<f32> = self.layers[0].forward(input);
            for i in 1..self.layers.len() {
                pre = self.layers[i].forward(&pre);
            }
            pre
        }

        fn fit(&mut self, data: &Vec<Vec<f32>>, label: &Vec<Vec<f32>>, lamda: f32) -> f32 {
            let last_label = label;
            let mut out: Vec<Vec<f32>> = Vec::new();
            for i in 0..data.len() {
                out.push(self.predict(&data[i]));
                let true_value = &last_label[i];
                let mut g: Vec<Vec<f32>> = vec![true_value.clone()];
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
    #[macro_export]
    macro_rules! Dense {
        ($units:expr, $input:expr) => {
            Box::new(Dense::new($units, $input))
        };
    }

    #[macro_export]
    macro_rules! ReLu {
        () => {
            Box::new(ReLuLayer::new())
        };
    }

    #[macro_export]
    macro_rules! Sofitmax {
        () => {
            Box::new(SoftmaxLayer::new())
        };
    }

    #[macro_export]
    macro_rules! Model {
    ($($layer:expr),*) => {
      {
        let mut m = Model::new();
        $(m.add($layer);)*
        m
      }
    };
}
}
