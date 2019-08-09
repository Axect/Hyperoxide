extern crate hyperoxide;
use hyperoxide::*;

fn main() {
    let v = vec![Vector::from_vec(vec![0f64; 1000]); 1000];
    let mut m = Matrix::new(v, Row);
    println!("{:?}", m.clone() * m.clone());
}