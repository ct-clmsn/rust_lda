<!-- Copyright (c) 2020 Christopher Taylor                                          -->
<!--                                                                                -->
<!--   Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!--   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->

# rust_lda - A Rust Latent Dirichlet Allocation Library

Project details [here](http://www.github.com/ct-clmsn/rust_lda/).

This project implements latent dirichlet allocation using rust.

### License

Boost Software License

### Features
* InvertedIndex implementation
* Latent Dirichlet Allocation implementation 

### Demo
`cargo run --example example sample-data/web/en/`


### TODO
* Add parallelization support
* Test different sampling wheel techniques
* Human-readable print outs

### Author
Christopher Taylor

### Dependencies
[Rust](https://www.rust-lang.org)

### Thank you
* [MALLET](https://github.com/mimno/Mallet) - for providing the data in this projects 'sample-data' directory on the [MALLET site](http://mallet.cs.umass.edu/index.php).
