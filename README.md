# Hetu
A tiny artificial neural network rust library. 
The project is a simple artificial neural network library that now supports full connectivity, ReLu, Sigmoid, Softmax activation functions. Subsequent will join the convolution layer and so on. This project uses RUST language development
# Can be used like this
fn main() {
    
    let mut model = Model::new();
    let input_layer = Dense::new(2,4);
    let relu = ReLuLayer::new();
    let dense = Dense::new(3, 2);
    let softmax = SoftmaxLayer::new();
    model.add(Box::new(input_layer));
    model.add(Box::new(relu));
    model.add(Box::new(dense));
    model.add(Box::new(softmax));
    let (data, labels) = load_data();
    for _ in 0..5000 {
        println!("acc ------------- {}",model.fit(&data, &labels, 0.05f64));
    }
}
