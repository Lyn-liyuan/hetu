#![crate_type = "lib"]
#![crate_name = "hetu"]

pub mod ann {
    extern crate rand;
    use rand::Rng;
    type Mat = Vec<Vec<f32>>;

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

    fn relu(x: &Vec<f32>) -> Vec<f32> {
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

    pub fn softmax_acc(y: &Mat, label: &Mat) -> f32 {
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

    pub fn conv2d_single_kernel(
        input: &Vec<f32>,
        h: usize,
        w: usize,
        weight: &Vec<f32>,
        kernel_size: usize,
        padding: usize,
        strides: usize,
    ) -> Vec<f32> {
        if input.len() != (h * w) as usize {
            panic!(
                "input size error, w={}, h={}, but length is {}",
                w,
                h,
                input.len()
            );
        }
        if weight.len() != (kernel_size * kernel_size) as usize {
            panic!(
                "kernel size error, size={}, but length is {}",
                kernel_size,
                input.len()
            );
        }
        let mut rs = vec![];
        let h_max = h - kernel_size + 1 + 2 * padding;
        let w_max = w - kernel_size + 1 + 2 * padding;
        let new_w = w + 2 * padding;
        let new_h = h + 2 * padding;

        for hi in (0..h_max).step_by(strides as usize) {
            for wi in (0..w_max).step_by(strides as usize) {
                let base = wi + hi * new_w;
                let mut fp = 0.0f32;
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let index = base + kw + (kh * new_w);
                        let py = hi + kh;
                        let px = wi + kw;
                        if py >= padding
                            && px >= padding
                            && px < w + padding
                            && py < new_h - padding
                        {
                            let input_v = input[index - w * padding - py * 2 * padding - padding];
                            fp = fp + input_v * weight[kw + kh * kernel_size];
                        }
                    }
                }
                rs.push(fp);
            }
        }
        rs
    }

    pub fn conv2d(
        inputs: &Mat,
        h: usize,
        w: usize,
        weights: &Mat,
        kernel_size: usize,
        padding: usize,
        strides: usize,
    ) -> Mat {
        let featurs_len = (h - kernel_size + 1 + 2 * padding) / strides
            * (w - kernel_size + 1 + 2 * padding)
            / strides;
        let mut rs: Mat = Vec::new();
        for cell in weights.iter() {
            let mut fm: Vec<f32> = vec![0.0f32; featurs_len];
            for input in inputs.iter() {
                let channel =
                    conv2d_single_kernel(input, h, w, cell, kernel_size, padding, strides);
                for (i, c) in channel.iter().enumerate() {
                    fm[i] = fm[i] + c;
                }
            }
            rs.push(fm);
        }
        rs
    }

    pub struct Node {
        pub input: Vec<f32>,
        pub output: Vec<f32>,
    }
    pub struct Dense {
        pub node: Node,
        pub units: usize,
        pub input_size: usize,
        pub weights: Mat,
        pub grads: Mat,
    }

    impl Dense {
        pub fn new(units: usize, input: usize) -> Dense {
            let grads: Mat = vec![vec![0.0f32; input]; units];
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
        fn forward(&mut self, input: &Mat) -> Mat;
        fn backward(&mut self, grads: &Mat) -> Mat;
        fn update_weights(&mut self, lamda: f32);
        fn clear(&mut self);
    }

    impl Layer for Dense {
        fn forward(&mut self, input: &Mat) -> Mat {
            self.node.input = input.concat();
            let rs: Vec<f32> = (0..self.units)
                .map(|i| vec_sum_mutiplier(&self.weights[i], &self.node.input))
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
        fn forward(&mut self, input: &Mat) -> Mat {
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
        fn forward(&mut self, input: &Mat) -> Mat {
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
        fn forward(&mut self, input: &Mat) -> Mat {
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
        fn forward(&mut self, input: &Mat) -> Mat {
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

    pub struct Model {
        pub layers: Vec<Box<Layer>>,
    }

    impl Model {
        pub fn new() -> Self {
            Model { layers: Vec::new() }
        }
        pub fn add(&mut self, layer: Box<Layer>) {
            &self.layers.push(layer);
        }
        pub fn predict(&mut self, input: &Mat) -> Vec<f32> {
            let mut pre: Mat = self.layers[0].forward(input);
            for i in 1..self.layers.len() {
                pre = self.layers[i].forward(&pre);
            }
            pre.concat()
        }

        pub fn fit(&mut self, data: &Vec<Mat>, label: &Mat, lamda: f32) -> f32 {
            let last_label = label;
            let mut out: Mat = Vec::new();
            for i in 0..data.len() {
                out.push(self.predict(&data[i]));
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
    macro_rules! Softmax {
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
