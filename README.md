# Hetu
A tiny artificial neural network rust library. 
The project is a simple artificial neural network library that now supports full connectivity, ReLu, Sigmoid, Softmax activation functions.  This project uses RUST language development
# Can be used like this
``` rust
    let mut model = Model![
        Dense!(8,4),
        ReLu!(),
        Dense!(3,8),
        Sofitmax!()
    ];
    let (data, labels) = load_data(args.path.as_path());
    for p in 0..6000 {
        println!("epoch {} loss = {}", p, 1f32 - model.fit(&data, &labels, 0.01f32));
    }
```