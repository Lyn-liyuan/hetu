extern crate rand;
extern crate hetu;

use rand::seq::SliceRandom;
use rand::thread_rng;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use hetu::ann::*;



fn load_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let path = Path::new("C:\\Users\\lyn\\rust_project\\hello\\iris.data");
    let mut file = match File::open(path) {
        Err(why) => panic!("couldn't open {}: {}", "iris.data", why.description()),
        Ok(file) => file,
    };

    let mut content = String::new();
    match file.read_to_string(&mut content) {
        Err(why) => panic!("couldn't read {}: {}", "iris.data", why.description()),
        Ok(_) => (),
    }

    let mut lines: Vec<&str> = content.split(|c| c == '\n').collect();
    let mut rng = thread_rng();
    lines.shuffle(&mut rng);
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<Vec<f64>> = Vec::new();
    for line in lines.iter() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() == 5 {
            let mut row: Vec<f64> = Vec::new();
            let mut mx: f64 = 0.000001f64;
            for i in 0..fields.len() - 1 {
                let value = fields[i].parse::<f64>().unwrap_or(0f64);
                if value > mx {
                    mx = value
                }
                row.push(value);
            }
            data.push(row.iter().map(|v| v / mx).collect());
            let setosa = vec![1f64, 0f64, 0f64];
            let versicolor = vec![0f64, 1f64, 0f64];
            let virginica = vec![0f64, 0f64, 1f64];
            if fields[4] == "Iris-setosa" {
                labels.push(setosa.clone());
            } else if fields[4] == "Iris-versicolor" {
                labels.push(versicolor.clone());
            } else {
                labels.push(virginica.clone());
            }
        }
    }
    (data, labels)
}

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


