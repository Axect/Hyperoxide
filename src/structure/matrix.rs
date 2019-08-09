use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::{Vector, FP, LinearAlgebra, Norm, Frobenius, PQ, One, Infinity};
pub use Shape::{Col, Row};
use std::mem::swap;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Shape {
    Row,
    Col,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Vector>,
    row: usize,
    col: usize,
    shape: Shape,
}

impl Matrix {
    pub fn new(data: Vec<Vector>, shape: Shape) -> Self {
        let row: usize;
        let col: usize;
        match shape {
            Row => {
                row = data.len();
                col = data[0].len();
            }
            Col => {
                col = data.len();
                row = data[0].len();
            }
        }

        Matrix {
            data,
            row,
            col,
            shape,
        }
    }

    pub fn change_shape(&self) -> Matrix {
        let r = self.row;
        let c = self.col;

        match self.shape {
            Row => {
                let mut data: Vec<Vector> = Vec::new();
                for i in 0 .. self.col {
                    let mut v = vec![0f64; r];
                    for j in 0 .. self.row {
                        v[j] = self[j][i];
                    }
                    data.push(Vector::from_vec(v));
                }
                Matrix::new(data, Col)
            }
            Col => {
                let mut data: Vec<Vector> = Vec::new();
                for i in 0 .. self.row {
                    let mut v = vec![0f64; c];
                    for j in 0 .. self.col {
                        v[j] = self[j][i];
                    }
                    data.push(Vector::from_vec(v));
                }
                Matrix::new(data, Row)
            }
        }
    }
}

// =============================================================================
// Matrix
// =============================================================================
impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, pair: (usize, usize)) -> &Self::Output {
        let (i, j) = pair;
        &self.data[i][j]
    }
}

impl Index<usize> for Matrix {
    type Output = Vector;

    fn index(&self, key: usize) -> &Self::Output {
        &self.data[key]
    }
}

impl Add<Matrix> for Matrix {
    type Output = Self;

    fn add(self, rhs: Matrix) -> Self::Output {
        self.zip_with(|x, y| x + y, &rhs)
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Self;

    fn sub(self, rhs: Matrix) -> Self::Output {
        self.zip_with(|x, y| x - y, &rhs)
    }
}

/// Matrix multiplication
impl Mul<Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: Matrix) -> Self::Output {
        match (self.shape, rhs.shape) {
            (Row, Col) => {
                let mut data: Vec<Vector> = Vec::new();
                for i in 0 .. self.row {
                    let v = &self[i];
                    let mut w = vec![0f64; rhs.col];
                    for j in 0 .. rhs.col {
                        w[j] = v * &rhs[j];
                    }
                    data.push(Vector::from_vec(w));
                }
                Matrix::new(data, self.shape)
            },
            (Row, Row) => {
                self.mul(rhs.change_shape())
            },
            (Col, Col) => {
                self.change_shape().mul(rhs)
            },
            _ => {
                self.change_shape().mul(rhs.change_shape())
            }
        }
    }
}

// =============================================================================
// Functional Programming
// =============================================================================
impl FP for Matrix {
    type Element = Vector;

    fn fmap<F>(&self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        Matrix::new(
            self.data.clone().into_iter().map(|v| f(v)).collect(),
            self.shape,
        )
    }

    fn zip_with<G>(&self, g: G, other: &Self) -> Self
    where
        G: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Matrix::new(
            self.data
                .clone()
                .into_iter()
                .zip(other.data.clone())
                .map(|(x, y)| g(x, y))
                .collect(),
            self.shape,
        )
    }

    fn reduce<G>(&self, g: G, default: Self::Element) -> Self::Element where
        G: Fn(Self::Element, Self::Element) -> Self::Element {
        self.data.clone().into_iter().fold(default, g)
    }
}

// =============================================================================
// Linear Algebra
// =============================================================================
impl LinearAlgebra for Matrix {
    fn transpose(&self) -> Self {
        let mut m = self.change_shape();
        swap(&mut m.row, &mut m.col);
        m.shape = self.shape;
        m
    }

    fn t(&self) -> Self {
        self.transpose()
    }

    /// Matrix norm
    ///
    /// # Kinds
    /// * `Frobenius` : Frobenius norm
    /// * `PQ(usize, usize)` : L_pq norm
    /// * `One` : 1-norm
    /// * `Infinity` : Infinity norm
    fn norm(&self, norm: Norm) -> f64 {
        match norm {
            Frobenius => {
                let mut s = 0f64;
                match self.shape {
                    Row => {
                        for i in 0 .. self.row {
                            let v_ref = &self[i];
                            for j in 0 .. self.col {
                                s += v_ref[j].powi(2);
                            }
                        }
                    }
                    Col => {
                        for i in 0 .. self.col {
                            let v_ref = &self[i];
                            for j in 0 .. self.row {
                                s += v_ref[j].powi(2);
                            }
                        }
                    }
                }
                s.sqrt()
            }
            PQ(p, q) => {
                let mut s = 0f64;
                for j in 0..self.col {
                    let mut s_row = 0f64;
                    for i in 0..self.row {
                        s_row += self[(i, j)].powi(p as i32);
                    }
                    s += s_row.powf(q as f64 / (p as f64));
                }
                s.powf(1f64 / (q as f64))
            }
            One => {
                let mut m = std::f64::MIN;
                let a = match self.shape {
                    Row => self.change_shape(),
                    Col => self.clone(),
                };
                for c in 0..a.col {
                    let s = a[c].reduce(|x, y| x + y, 0f64);
                    if s > m {
                        m = s;
                    }
                }
                m
            }
            Infinity => {
                let mut m = std::f64::MIN;
                let a = match self.shape {
                    Row => self.clone(),
                    Col => self.change_shape(),
                };
                for r in 0..a.row {
                    let s = a[r].reduce( |x, y| x + y, 0f64);
                    if s > m {
                        m = s;
                    }
                }
                m
            }
        }
    }

    fn det(&self) -> f64 {
        unimplemented!()
    }

    fn lu(&self) -> Option<(Vec<(usize, usize)>, Vec<(usize, usize)>, Self, Self)> {
        unimplemented!()
    }

    fn block(&self) -> (Self, Self, Self, Self) {
        unimplemented!()
    }

    fn inv(&self) -> Option<Self> {
        unimplemented!()
    }

    fn pseudo_inv(&self) -> Option<Self> {
        unimplemented!()
    }
}