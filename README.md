# Hetu
A tiny artificial neural network rust library. 
The project is a simple artificial neural network library that now supports Connected Layer, Convolution Layer, max pooling layer, ReLu, Sigmoid, Softmax activation functions.  This project uses RUST language development
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