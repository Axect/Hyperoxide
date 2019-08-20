extern crate hyperoxide;
use hyperoxide::prelude::*;

fn main() {
    let v = vec![Vector::from_vec(vec![0f64; 1000]); 1000];
    let mut m = Matrix::new(v.clone(), Row);
    let mut n = Matrix::new(v.clone(), Col);
    println!("{:?}", m.clone() * n.clone());
}
