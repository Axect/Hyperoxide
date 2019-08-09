pub use Norm::*;

/// Norm Enum
#[derive(Debug, Copy, Clone)]
pub enum Norm {
    Frobenius,
    PQ(usize, usize),
    One,
    Infinity,
}

pub type Perms = Vec<(usize, usize)>;

pub trait LinearAlgebra: Sized {
    fn transpose(&self) -> Self;
    fn t(&self) -> Self;
    fn norm(&self, norm: Norm) -> f64;
    fn det(&self) -> f64;
    fn lu(&self) -> Option<(Perms, Perms, Self, Self)>;
    fn block(&self) -> (Self, Self, Self, Self);
    fn inv(&self) -> Option<Self>;
    fn pseudo_inv(&self) -> Option<Self>;
}