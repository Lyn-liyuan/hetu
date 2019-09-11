# Hetu
![HETU](https://github.com/Lyn-liyuan/hetu/raw/master/logo.jpg)
The Hetu is a mysterious pattern in the ancient Chinese mythology, and the gossip evolved from it can be used for divination and prediction. Its black dots and white dots resemble neurons.So we use the Hetu to name our neural network library.  

The Hetu is a tiny artificial neural network rust library. 
The project is a simple artificial neural network library and it supports Connected Layer, Convolution Layer, max pooling layer, ReLu, Sigmoid, Softmax activation functions. This project uses the RUST language to implement.  

## Directory structure
- exmaple : some examples
- hetulib: the source code of library
- vulkan: the compute operators is base on GPU. It is just an experiment now.

## Example
``` rust
    let mut model = Model![
        Dense!(8,4),
        ReLu!(),
        Dense!(3,8),
        Softmax!()
    ];
    let (data, labels) = load_data(args.path.as_path());
    for p in 0..6000 {
        println!("epoch {} loss = {}", p, 1f32 - model.fit(&data, &labels, 0.01f32));
    }
```
The example of convolution network for MNIST recognition  is coming soon.