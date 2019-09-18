use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn load_data(path:&Path) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    //let path = Path::new("C:\\Users\\lyn\\rust_project\\hello\\iris.data");
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
    let mut data: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();
    for line in lines.iter() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() == 5 {
            let mut row: Vec<f32> = Vec::new();
            let mut mx: f32 = 0.000001f32;
            for i in 0..fields.len() - 1 {
                let value = fields[i].parse::<f32>().unwrap_or(0f32);
                if value > mx {
                    mx = value
                }
                row.push(value);
            }
            data.push(vec![row.iter().map(|v| v / mx).collect()]);
            let setosa = vec![1f32, 0f32, 0f32];
            let versicolor = vec![0f32, 1f32, 0f32];
            let virginica = vec![0f32, 0f32, 1f32];
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