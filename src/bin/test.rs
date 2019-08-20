extern crate hyperoxide;
use hyperoxide::prelude::*;

fn main() {
    let v = Vector::from_vec(vec![1f64, 2f64]);
    let w = Vector::from_vec(vec![3f64, 4f64]);
    let v2 = Vector::from_vec(vec![1f64, 3f64]);
    let w2 = Vector::from_vec(vec![2f64, 4f64]);

    let m1 = Matrix::new(vec![v, w], Row);
    let m2 = Matrix::new(vec![v2, w2], Col);

    let m = m1.change_shape() * m2.clone();
    let n = m1.transpose() * m2.clone();
    println!("{:?}", m);
    println!("{:?}", n);
}
