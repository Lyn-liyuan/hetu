extern crate rand;
#[macro_use]
extern crate hetu;
extern crate structopt;
extern crate quicli;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::prelude::*;
use structopt::StructOpt;
use hetu::ann::*;


include!{"data.rs"}


fn main() {
    #[derive(StructOpt)]
    struct Cli {
        #[structopt(parse(from_os_str))]
        path: std::path::PathBuf,
    };
    let args = Cli::from_args();
    
    let mut model = Model![
        Dense!(8,4,true),
        ReLu!(),
        Dense!(3,8,true),
        Softmax!()
    ];
    let (data, labels) = load_data(args.path.as_path());
    for p in 0..6000 {
        println!("epoch {} loss = {}", p, 1f32 - model.fit(&data, &labels, 0.01f32));
    }
}


