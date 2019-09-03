pub type Mat = Vec<Vec<f32>>;
pub type Vector = Vec<f32>;
    
pub fn vec_sum_mutiplier(vc1: &Vector, vc2: &Vector) -> f32 {
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

pub fn relu(x: &Vector) -> Vector {
    x.iter().map(|v| v.max(0.0f32)).collect()
}

pub fn derivative_relu(x: &Vector) -> Vector {
    x.iter()
        .map(|i| if *i >= 0.0f32 { 1f32 } else { 0f32 })
        .collect()
}

pub fn sigmoid(x: &f32) -> f32 {
    1.0f32 / (1.0f32 + (-1.0f32 * x).exp())
}

pub fn vec_sigmoid(x: &Vector) -> Vector {
    x.iter().map(|v| sigmoid(v)).collect()
}

pub fn softmax(x: &Vector) -> Vector {
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

pub fn derivative_softmax_cross_entropy(x: &Vector, y: &Vector) -> Vector {
    if x.len() != y.len() {
        panic!("vector size is not equals!!!")
    }
    x.iter().enumerate().map(|(i, x)| x - y[i]).collect()
}

pub fn conv2d_single_kernel(
    input: &Vector,
    h: usize,
    w: usize,
    weight: &Vector,
    kernel_size: usize,
    padding: usize,
    strides: usize,
) -> Vector {
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
    let features_len = (h - kernel_size + 1 + 2 * padding) / strides
        * (w - kernel_size + 1 + 2 * padding)
        / strides;
    let mut rs: Mat = Vec::new();
    for cell in weights.iter() {
        let mut fm: Vector = vec![0.0f32; features_len];
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
    pub input: Vector,
    pub output: Vector,
}

pub trait Layer {
    fn forward(&mut self, input: &Mat, training: bool) -> Mat;
    fn backward(&mut self, grads: &Mat) -> Mat;
    fn update_weights(&mut self, lamda: f32);
    fn clear(&mut self);
}


