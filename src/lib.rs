//  Copyright (c) 2020 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This file implements LDA w/Collapsed Gibbs Samping by:
//
//     D. Newman, A. Asuncion, P. Smyth, M. Welling. "Distributed Algorithms for Topic Models." JMLR 2009.
//
// https://www.ics.uci.edu/~asuncion/software/fast.htm
//
extern crate regex;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;

use std::collections::HashMap;
use std::ops::{Index, IndexMut};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use regex::Regex;

use crate::ndarray_rand::rand_distr::Distribution;
use ndarray::{Axis, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_stats::QuantileExt;

use std::{env, fs};
use std::path::Path;

//use rayon::prelude::*;

// InvertedIndex is...
//
// word -> { docid : word_freq } 
//
pub struct InvertedIndex {
   pub num_documents : usize,
   pub indices : HashMap< String, HashMap<u64, u64> >,
}

// Index method or 'trait' for InvertedIndex[String]
//
impl Index<String> for InvertedIndex {
    type Output = HashMap<u64, u64>;

    fn index(&self, token: String) -> &Self::Output {
        match self.indices.get(&token) {
            Some(hist) => hist,
            None => panic!("Index<String> Error: attempted to access an undefined Token")
        }
    }
}

// Index method or 'trait' for InvertedIndex[String]
//
impl IndexMut<String> for InvertedIndex {
    fn index_mut(&mut self, token: String ) -> &mut Self::Output {
        if let Some(hist) = self.indices.get_mut(&token) {
            return hist;
        } else {
            panic!("IndexMut<String> Error: attempted to access an undefined Token")
        }
    }
}

impl InvertedIndex {

    pub fn print( &self) {
        for (token, hist) in &self.indices {
            println!("token \"{}\"", token);
            for (k, v) in hist {
                println!("\t-> document {} = frequency {}", k, v);
            }
        }
    }

    pub fn to_matrix( &self ) -> ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 2]>> {
        let mut document_term_matrix : Array::<u64, _> = Array::zeros((self.num_documents, self.indices.len()));
        let mut term_idx : usize = 0;

        for (_, document_histogram) in &self.indices {
            for(key, value) in document_histogram {
                let k = *key as usize;
                document_term_matrix[[k,term_idx]] += value;
            }

            term_idx += 1;
        }

        return document_term_matrix;
    }

    pub fn load( &mut self, document_id : u64, document_text : String ) {
        let re:Regex = Regex::new("[\\p{L}\\p{M}]+").unwrap();

        for cap in re.captures_iter(&document_text) {
            let token = cap[0].to_string().to_lowercase();

            if self.indices.contains_key(&token) {
                let histogram = &mut self[token];

                if histogram.contains_key(&document_id) {
                    let value = histogram.get_mut(&document_id).expect("Missing value");
                    *value += 1;            
                }
                else {
                    histogram.insert(document_id, 1);
                }
            }
            else {
                let mut tmphist : HashMap<u64, u64> = HashMap::new();
                tmphist.insert(document_id, 1);
                self.indices.insert(token, tmphist);
            }
        }

        self.num_documents += 1;
    }

    pub fn load_files(&mut self, files : &Vec<String>) {
        let files_len = files.len();

        for i in 0..files_len {
            let path = Path::new(&files[i]);
            //println!("{}", &files[i]);
            let contents = fs::read_to_string(&files[i]).expect("read error");
            self.load(i as u64, contents); 
        }
    }

} // end InvertedIndex trait

struct PartitionedInvertedIndex {
    npartitions : usize,
    num_documents : usize,
    partitions : Vec< InvertedIndex >,
}

impl PartitionedInvertedIndex {

    pub fn to_matrix( self ) -> Vec< ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 2]>> > {
        type Matrix = ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 2]>>;

        let vocabulary_count = (0..self.npartitions).map(|x| self.partitions[x].indices.len()).sum();

        let mut matrices : Vec< Matrix > =
            (0..self.npartitions).map(|_x| Array::zeros((self.num_documents, vocabulary_count))).collect();

        let mut accumulated_vocabulary_count : Vec< usize > =
            (0..self.npartitions).scan(0 as usize, |acc, x| {*acc += self.partitions[x].indices.len(); Some(*acc)}).collect();

        accumulated_vocabulary_count.insert(0, 0);

        //let partitions : Vec<usize> = (0..self.npartitions).map(|x| x).collect();

        for p in 0..self.npartitions {
            let mut term_idx : usize = accumulated_vocabulary_count[p];

            for (_, document_histogram) in &self.partitions[p].indices {
                for(key, value) in document_histogram {
                    let k = *key as usize;
                    matrices[p][[k,term_idx]] += value;
                }

                term_idx += 1;
            }
        }

        return matrices;
    }

    pub fn print(&self) {
        let mut partition_counter : usize = 0;

        for p in self.partitions.iter() {
            println!("partition -> {}", partition_counter);
            p.print();
            partition_counter +=1;
        }
    }

    pub fn load( &mut self, document_id : u64, document_text : String ) {
        let re : Regex = Regex::new("[\\p{L}\\p{M}]+").unwrap();
        let mut hasher = DefaultHasher::new();

        if self.partitions.len() < 1 {
            self.partitions = (0..self.npartitions).map(|_x| InvertedIndex{ num_documents : 0, indices : HashMap::new() }).collect();
        }

        for cap in re.captures_iter(&document_text) {
            let token = cap[0].to_string().to_lowercase();
            token.hash(&mut hasher);
            let token_partition = (hasher.finish() as usize) % self.npartitions;

            if self.partitions[token_partition].indices.contains_key(&token) {
                let histogram = &mut self.partitions[token_partition][token];

                if histogram.contains_key(&document_id) {
                    let value = histogram.get_mut(&document_id).expect("Missing value");
                    *value += 1;            
                }
                else {
                    histogram.insert(document_id, 1);
                }
            }
            else {
                let mut tmphist : HashMap<u64, u64> = HashMap::new();
                tmphist.insert(document_id, 1);
                self.partitions[token_partition].indices.insert(token, tmphist);
            }
        }

        self.num_documents += 1;
        (0..self.npartitions).map(|x| self.partitions[x].num_documents = self.num_documents ).collect()
    }

} // end PartitionedInvertedIndex trait

pub struct LatentDirichletAllocation {
    pub t : usize,
    pub iterations : usize,
    pub alpha : f64,
    pub beta : f64,
    pub smoothing_mass : f64,
    pub optimize_interval : usize,
}

impl LatentDirichletAllocation {

    fn gibbs(&mut self,
             document_word_matrix : & ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 2]>>,
             z : &mut ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>,
             wp0 : & ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
             dp : &mut ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
             ztot0 : &mut ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>)
    {
        type MatrixF = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>;
        type VectorF = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;
 
        let d : usize = document_word_matrix.nrows();
        let w : usize = document_word_matrix.ncols();
        let wbeta : f64 = ( w as f64 ) * self.beta; 

        let mut wp : MatrixF = Array::zeros((w, self.t));

        wp.assign( &wp0 );

        let mut rng = rand::thread_rng();
        let mut n : usize = 0;

        for di in 0..d {
            for wi in 0..w {

                for _f in 0..document_word_matrix[ [di, wi] ] as usize {

                    let mut t : usize = z[n];

                    ztot0[t] -= 1.0;
                    wp[ [wi, t] ] -= 1.0;
                    dp[ [di, t] ] -= 1.0;

                    let probs : VectorF = wp.slice(s![wi, ..]).iter().zip(dp.slice(s![di, ..]).iter()).zip(ztot0.iter()).map(|(x, y)| ((*x.0 as f64) * self.beta * (*x.1 as f64) * self.alpha) / ((*y as f64) * wbeta) ).collect();
                    let die = rand::distributions::Uniform::new(0.0, 1.0);
                    let mut max_prob : f64 = probs.sum() * die.sample(&mut rng) * 2.0;
                    let mut cur_prob = probs[0];

                    t = 0;
                    while cur_prob < max_prob {
                        t = (t+1) % self.t;
                        cur_prob += probs[t].abs();
                    }

                    z[n] = t;

                    ztot0[t] += 1.0;
                    wp[[wi, t]] += 1.0;
                    dp[[di, t]] += 1.0;

                    n += 1;
                }
            }
        }
    }

    pub fn train(&mut self, document_word_matrix : ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 2]>> )
        -> (ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>,
            ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
            ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>) {

        type MatrixF = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>;
        type VectorF = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;
        type VectorU = ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>;
        
        let n : usize = document_word_matrix.sum() as usize;
        let d : usize = document_word_matrix.nrows();
        let w : usize = document_word_matrix.ncols();
        let wbeta : f64 = w as f64 * self.t as f64;

        let mut wp : MatrixF = Array::zeros((w, self.t));
        let mut dp : MatrixF = Array::zeros((d, self.t));
        let mut z : VectorU = Array::random(n, Uniform::from(0..self.t));

        //println!("building wp/dp matrices");
        {
            let mut k : usize = 0;
            for i in 0..d {
                for j in 0..w {
                    for _h in 0..document_word_matrix[[i,j]] as usize {
                        dp[[ i, z[k] ]] += 1.0;
                        wp[[ j, z[k] ]] += 1.0;
                        k += 1;
                    }
                }
            }
        }

        //println!("running gibbs sampler");
        let mut wp0 : MatrixF = Array::zeros((w, self.t));
        let mut ztot0 : VectorF = Array::zeros(self.t);

        for i in 0..self.iterations {

            if i > 0 && i % 10 == 0 {
                println!("\titeration {}", i);
            }

            &wp0.assign(&wp);
            ztot0.assign( &wp0.sum_axis(Axis(0)) );
            self.gibbs(&document_word_matrix, &mut z, &wp, &mut dp, &mut ztot0);

            let wp_wp0 = &wp - &wp0;
            wp.assign( &( &wp0 + &wp_wp0 ) );
        }

        (z, wp, dp)
    }

} // end LatentDirichletAllocation trait

pub fn get_documents(directory_str : &String) -> Vec<String> {
    let mut files_vector : Vec<String> = Vec::new();

    let dir_path = Path::new(&directory_str);

    if !dir_path.exists() || !dir_path.is_dir() {
        return files_vector;
    }

    let dir = fs::read_dir(&dir_path);

    for d in dir.unwrap() {
        let dentry = d.unwrap();
        let filetype = &dentry.file_type().unwrap();
        if filetype.is_file() {
            let mut filename : String = directory_str.to_string();
            filename.push_str(&dentry.file_name().into_string().unwrap());
            files_vector.push(filename);
        }
    }

    files_vector
}
