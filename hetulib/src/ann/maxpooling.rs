
use crate::base::Mat;
use crate::base::Vector;
use crate::base::Layer;

pub struct MaxPooling {
    pub input: Mat,
    pub output: Mat,
    pub input_width: usize,
    pub input_height: usize,
    pub pool_size: usize,
    pub strides: usize,
    pub padding: usize,
    pub max_index: Mat,
}
impl MaxPooling {
    pub fn new(
        input_width: usize,
        input_height: usize,
        pool_size: usize,
        strides: usize,
        padding: usize,
    ) -> Self {
        
        MaxPooling {
            input: vec![vec![]],
            output: vec![vec![]],
            input_width: input_width,
            input_height: input_height,
            pool_size: pool_size,
            strides: strides,
            padding: padding,
            max_index: vec![vec![]],
        }
    }
    fn pooling_single_channel(
        &mut self,
        input: &Vector,
        h: usize,
        w: usize,
        pooling_size: usize,
        padding: usize,
        strides: usize,
    ) -> (Vector, Vec<usize>) {
            if input.len() != (h * w) as usize {
                panic!(
                    "input size error, w={}, h={}, but length is {}",
                    w,
                    h,
                    input.len()
                );
            }
            
            let mut rs = vec![];
            let mut index_vec:Vec<usize> = vec![];
            let h_max = h - kernel_size + 1 + 2 * padding;
            let w_max = w - kernel_size + 1 + 2 * padding;
            let new_w = w + 2 * padding;
            let new_h = h + 2 * padding;

            for hi in (0..h_max).step_by(strides as usize) {
                for wi in (0..w_max).step_by(strides as usize) {
                    let base = wi + hi * new_w;
                    let mut maxv = 0.0f32;
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
                                let rindex = index - w * padding - py * 2 * padding - padding
                                let input_v = input[rindex];
                                if input_v > maxv {
                                    maxv = input_v;
                                }
                                index_vec.push(rindex);
                            }
                        }
                    }
                    rs.push(maxv);
                }
            }
            (rs,index_vec)
        }
    }

impl Layer for MaxPooling {
    fn forward(&mut self, input: &Mat, _: bool) -> Mat {
        self.input = input.clone();
        let mut result: Mat = vec![vec![]];
        for row in self.input.iter() {
            let (rs_row, indexs_row) = self.pooling_single_channel(row, self.input_height, self.input_width, self.pooling_size, self.padding, self.strides)
            result.push(rs_row);
            self.max_index.push(indexs_row);
        }
        result
    }

    fn backward(&mut self, up_grads: &Mat) -> Mat {
        let channel = self.input.len();
        let length = self.input_height*self.input_width;
        let mut back = vec![vec![0f32; length]; channel];
        for (i, ug) in up_grads.iter().enumerate() {
            for (j, grads) in ug.iter().enumerate() {
                back[i][self.max_index[i][j]] = grads;
            }
        }
        back.clone()
    }

    fn update_weights(&mut self, lamda: f32) {
    }

    fn clear(&mut self) {
    }
}