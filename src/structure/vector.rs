use crate::operation::fp::FP;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[derive(Debug, Clone)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn from_vec(v: Vec<f64>) -> Self {
        Vector { data: v }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// =============================================================================
// Operator Overloading
// =============================================================================

impl Index<usize> for Vector {
    type Output = f64;
    fn index(&self, key: usize) -> &Self::Output {
        &self.data[key]
    }
}

impl Add<Vector> for Vector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());

        let mut data = vec![0f64; self.len()];
        for i in 0..self.len() {
            data[i] = self.data[i] + other.data[i];
        }
        Vector::from_vec(data)
    }
}

impl Sub<Vector> for Vector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());

        let mut data = vec![0f64; self.len()];
        for i in 0..self.len() {
            data[i] = self.data[i] - other.data[i];
        }
        Vector::from_vec(data)
    }
}

/// Dot product
impl Mul<Vector> for Vector {
    type Output = f64;

    fn mul(self, other: Self) -> f64 {
        assert_eq!(self.len(), other.len());

        let mut s = 0f64;
        for i in 0..self.len() {
            s += self.data[i] * other.data[i];
        }
        s
    }
}

impl<'a, 'b> Add<&'b Vector> for &'a Vector {
    type Output = Vector;

    fn add(self, other: &'b Vector) -> Self::Output {
        let l = self.len();
        assert_eq!(l, other.len());

        let mut data = vec![0f64; l];
        for i in 0..l {
            data[i] = self.data[i] + other.data[i];
        }
        Vector::from_vec(data)
    }
}

impl<'a, 'b> Mul<&'b Vector> for &'a Vector {
    type Output = f64;

    fn mul(self, rhs: &Vector) -> Self::Output {
        let l = self.len();
        assert_eq!(l, rhs.len());

        let mut s = 0f64;
        for i in 0..l {
            s += self.data[i] * rhs.data[i];
        }
        s
    }
}

impl FP for Vector {
    type Element = f64;

    fn fmap<F>(&self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        let mut data = vec![0f64; self.len()];
        for i in 0..self.len() {
            data[i] = f(self[i]);
        }
        Vector { data }
    }

    fn zip_with<G>(&self, g: G, other: &Self) -> Self
    where
        G: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let l = self.len();
        assert_eq!(l, other.len());

        let mut data = vec![0f64; l];
        for i in 0..l {
            data[i] = g(self[i], other[i]);
        }
        Vector { data }
    }

    fn reduce<G>(&self, g: G, default: Self::Element) -> Self::Element where
        G: Fn(Self::Element, Self::Element) -> Self::Element {
        self.data.clone().into_iter().fold(default, g)
    }
}
