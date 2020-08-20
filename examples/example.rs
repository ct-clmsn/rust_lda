//  Copyright (c) 2020 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
extern crate regex;

use std::collections::HashMap;
use std::{env};

use rust_lda::{InvertedIndex, LatentDirichletAllocation, get_documents};

fn main() {
    let args : Vec<String> = env::args().collect();
    let directory = &args[1];
    println!("identifying files to process in {}", &directory);
    let filename_vector = get_documents(&directory);

    let mut index : InvertedIndex = InvertedIndex{ num_documents : 0, indices : HashMap::new() };
    println!("loading files into index");
    index.load_files(&filename_vector);

    println!("index converted to matrix");
    let matrix = index.to_matrix();

    let mut lda = LatentDirichletAllocation{ t : 4, iterations : 2000, alpha : 0.2, beta : 0.1, smoothing_mass : 0.0, optimize_interval : 50};

    println!("training lda");
    
    let ret_tuple = lda.train(matrix);
    println!("topics!");
    println!("{} {} {}", ret_tuple.0, ret_tuple.1, ret_tuple.2);
}
