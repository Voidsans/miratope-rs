//! Contains methods to generate many symmetry groups.

// Circumvents rust-analyzer bug.
#[allow(unused_imports)]
use crate::cox;

use super::cox::CoxMatrix;
use crate::EPS;
use approx::abs_diff_ne;
use dyn_clone::DynClone;
use itertools::iproduct;
use nalgebra::{DMatrix as Matrix, DVector as Vector, Quaternion};
use std::{
    collections::{BTreeMap, VecDeque},
    f64::consts::PI,
    iter,
    ops::{Deref, DerefMut},
};

/// Converts a 3D rotation matrix into a quaternion. Uses the code from
/// [Day (2015)](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf).
fn mat_to_quat(mat: Matrix<f64>) -> Quaternion<f64> {
    debug_assert!(mat.determinant() > 0.0);

    let t;
    let q;

    if mat[(2, 2)] < 0.0 {
        if mat[(0, 0)] > mat[(1, 1)] {
            t = 1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)];
            q = Quaternion::new(
                t,
                mat[(0, 1)] + mat[(1, 0)],
                mat[(2, 0)] + mat[(0, 2)],
                mat[(1, 2)] - mat[(2, 1)],
            );
        } else {
            t = 1.0 - mat[(0, 0)] + mat[(1, 1)] - mat[(2, 2)];
            q = Quaternion::new(
                mat[(0, 1)] + mat[(1, 0)],
                t,
                mat[(1, 2)] + mat[(2, 1)],
                mat[(2, 0)] - mat[(0, 2)],
            );
        }
    } else if mat[(0, 0)] < -mat[(1, 1)] {
        t = 1.0 - mat[(0, 0)] - mat[(1, 1)] + mat[(2, 2)];
        q = Quaternion::new(
            mat[(2, 0)] + mat[(0, 2)],
            mat[(1, 2)] + mat[(2, 1)],
            t,
            mat[(0, 1)] - mat[(1, 0)],
        );
    } else {
        t = 1.0 + mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];
        q = Quaternion::new(
            mat[(1, 2)] - mat[(2, 1)],
            mat[(2, 0)] - mat[(0, 2)],
            mat[(0, 1)] - mat[(1, 0)],
            t,
        );
    }

    q * 0.5 / t.sqrt()
}

/// Converts a quaternion into a matrix, depending on whether it's a left or
/// right quaternion multiplication.
fn quat_to_mat(q: Quaternion<f64>, left: bool) -> Matrix<f64> {
    let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let k = Quaternion::new(0.0, 0.0, 0.0, 1.0);

    let qi = if left { q * i } else { i * q };
    let qj = if left { q * j } else { j * q };
    let qk = if left { q * k } else { k * q };

    let c1 = q.coords.into_iter().copied();
    let c2 = qi.coords.into_iter().copied();
    let c3 = qj.coords.into_iter().copied();
    let c4 = qk.coords.into_iter().copied();

    // Compensates for the wack storage order.
    let mut mat = Matrix::from_iterator(4, 4, c1.chain(c2).chain(c3).chain(c4));
    mat.swap_rows(2, 3);
    mat.swap_rows(1, 2);
    mat.swap_rows(0, 1);

    mat
}

/// An iterator such that `dyn` objects using it can be cloned. Used to get
/// around orphan rules.
trait GroupIter: Iterator<Item = Matrix<f64>> + DynClone {}
impl<T: Iterator<Item = Matrix<f64>> + DynClone> GroupIter for T {}
dyn_clone::clone_trait_object!(GroupIter);

/// A [group](https://en.wikipedia.org/wiki/Group_(mathematics)) of matrices,
/// acting on a space of a certain dimension.
#[derive(Clone)]
pub struct Group {
    dim: usize,
    iter: Box<dyn GroupIter>,
}

impl Group {
    /// Gets all of the elements of the group. Consumes the iterator.
    fn elements(self) -> Vec<Matrix<f64>> {
        self.iter.collect()
    }

    /// Gets the number of elements of the group. Consumes the iterators.
    pub fn order(self) -> usize {
        self.iter.count()
    }

    /// Buils the rotation subgroup of a group.
    pub fn rotations(self) -> Self {
        // The determinant might not be exactly 1, so we're extra lenient and
        // just test for positive determinants.
        Self {
            dim: self.dim,
            iter: Box::new(self.iter.filter(|el| el.determinant() > 0.0)),
        }
    }

    fn quaternions(self, left: bool) -> Option<Self> {
        if self.dim != 3 {
            return None;
        }

        Some(Self {
            dim: 4,
            iter: Box::new(
                self.rotations()
                    .iter
                    .map(move |el| {
                        let q = mat_to_quat(el);
                        let m = quat_to_mat(q, left);
                        iter::once(m.clone()).chain(iter::once(-m))
                    })
                    .flatten(),
            ),
        })
    }

    pub fn left_quaternions(self) -> Option<Self> {
        self.quaternions(true)
    }

    pub fn right_quaternions(self) -> Option<Self> {
        self.quaternions(false)
    }

    /// Returns a new `Group` whose elements have all been generated already,
    /// so that they can be used multiple times quickly.
    pub fn cache(self) -> Self {
        self.elements().into()
    }

    /// Generates the trivial group of a certain dimension.
    fn trivial(dim: usize) -> Self {
        Self {
            dim,
            iter: Box::new(iter::once(Matrix::identity(dim, dim))),
        }
    }

    /// Generates the trivial group of a certain dimension.
    fn central_inv(dim: usize) -> Self {
        Self {
            dim,
            iter: Box::new(
                vec![Matrix::identity(dim, dim), -Matrix::identity(dim, dim)].into_iter(),
            ),
        }
    }

    /// Generates a Coxeter group from its [`CoxMatrix`], or returns `None` if
    /// the group doesn't fit as a matrix group in spherical space.
    fn cox_group(cox: CoxMatrix) -> Option<Self> {
        const EPS: f64 = 1e-6;

        let dim = cox.ncols();
        let mut generators = Vec::with_capacity(dim);

        // Builds each generator from the top down as a triangular matrix, so
        // that the dot products match the values in the Coxeter matrix.
        for i in 0..dim {
            let mut gen_i = Vector::from_element(dim, 0.0);

            for (j, gen_j) in generators.iter().enumerate() {
                let dot = gen_i.dot(gen_j);
                gen_i[j] = ((PI / cox[(i, j)] as f64).cos() - dot) / gen_j[j];
            }

            // The vector doesn't fit in spherical space.
            let norm_sq = gen_i.norm_squared();
            if norm_sq >= 1.0 - EPS {
                return None;
            } else {
                gen_i[i] = (1.0 - norm_sq).sqrt();
            }

            generators.push(gen_i);
        }

        Some(Self {
            dim,
            iter: Box::new(GenIter::new(
                dim,
                generators.into_iter().map(refl_mat).collect(),
            )),
        })
    }

    /// Generates the direct product of two groups. Uses the specified function
    /// to uniquely map the ordered pairs of matrices into other matrices.
    fn direct_product(
        g: Self,
        h: Self,
        dim: usize,
        product: &'static dyn Fn((Matrix<f64>, Matrix<f64>)) -> Matrix<f64>,
    ) -> Self {
        Self {
            dim,
            iter: Box::new(iproduct!(g.iter, h.iter).map(product)),
        }
    }
}

impl From<Vec<Matrix<f64>>> for Group {
    fn from(elements: Vec<Matrix<f64>>) -> Self {
        Self {
            dim: elements
                .get(0)
                .expect("Group must have at least one element.")
                .ncols(),
            iter: Box::new(elements.into_iter()),
        }
    }
}

mod product {
    use nalgebra::DMatrix as Matrix;

    /// Takes the normal matrix product of a pair of matrices.
    pub fn matrix((mat1, mat2): (Matrix<f64>, Matrix<f64>)) -> Matrix<f64> {
        mat1 * mat2
    }

    /// Takes the [direct sum](https://en.wikipedia.org/wiki/Block_matrix#Direct_sum)
    /// of a pair of matrices.
    pub fn direct_sum((mat1, mat2): (Matrix<f64>, Matrix<f64>)) -> Matrix<f64> {
        let dim1 = mat1.ncols();
        let dim2 = mat2.ncols();
        let dim = dim1 + dim2;

        let mut mat = Matrix::zeros(dim, dim);

        for i in 0..dim1 {
            for j in 0..dim1 {
                mat[(i, j)] = mat1[(i, j)];
            }
        }

        for i in 0..dim2 {
            for j in 0..dim2 {
                mat[(i + dim1, j + dim1)] = mat2[(i, j)];
            }
        }

        mat
    }
}

/// The result of trying to get the next element in a group.
pub enum GroupNext {
    /// We've already found all elements of the group.
    None,

    /// We found an element we had found previously.
    Repeat,

    /// We found a new element.
    New(Matrix<f64>),
}

#[derive(Clone, Debug)]
/// A matrix ordered by fuzzy lexicographic ordering. Used to quickly determine
/// whether an element in a [`GenGroup`] is a duplicate.
struct OrdMatrix(Matrix<f64>);

impl Deref for OrdMatrix {
    type Target = Matrix<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OrdMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PartialEq for OrdMatrix {
    fn eq(&self, other: &Self) -> bool {
        let mut other = other.iter();

        for x in self.iter() {
            let y = other.next().unwrap();

            if abs_diff_ne!(x, y, epsilon = EPS) {
                return false;
            }
        }

        true
    }
}

impl Eq for OrdMatrix {}

impl PartialOrd for OrdMatrix {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut other = other.iter();

        for x in self.iter() {
            let y = other.next().unwrap();

            if abs_diff_ne!(x, y, epsilon = EPS) {
                return x.partial_cmp(y);
            }
        }

        Some(std::cmp::Ordering::Equal)
    }
}

impl Ord for OrdMatrix {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// An iterator for a `Group` [generated](https://en.wikipedia.org/wiki/Generator_(mathematics))
/// by a set of floating point matrices. Its elements are built in a BFS order.
/// It contains a lookup table, used to figure out whether an element has
/// already been found or not.
///
/// # Todo
/// Currently, to figure out whether an element has been found or not, we do a
/// linear search on the entire set of elements that we've found so far. This
/// means that generating a group with *n* elements has O(*n*²) asymptotic
/// complexity, which will be really bad if we ever want to implement big groups
/// like E6, E7, or God forbid E8.
///
/// If all of our matrices had integer entries, which is the case for a lot of
/// Coxeter groups, we could instead use a `HashSet` to reduce the complexity
/// to O(*n* log(*n*)). For floating point entries, where we'll rather want to
/// find the "closest" element to another one (to account for imprecision), a
/// [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) would achieve the same
/// complexity, but it would be much harder to implement.
#[derive(Clone)]
pub struct GenIter {
    /// The number of dimensions the group acts on.
    dim: usize,

    /// The generators for the group.
    generators: Vec<Matrix<f64>>,

    /// The elements that have been generated. Will be put into a more clever
    /// structure that's asymptotically more efficient and doesn't need storing
    /// everything at once eventually.
    elements: BTreeMap<OrdMatrix, usize>,

    queue: VecDeque<OrdMatrix>,

    /// Stores the index in (`generators`)[GenGroup.generators] of the generator
    /// that's being checked. All previous once will have already been
    /// multiplied to the right of the current element. Quirk of the current
    /// data structure, subject to change.
    gen_idx: usize,
}

impl Iterator for GenIter {
    type Item = Matrix<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.try_next() {
                GroupNext::None => return None,
                GroupNext::Repeat => {}
                GroupNext::New(el) => return Some(el),
            };
        }
    }
}

/// Determines whether two matrices are "approximately equal" elementwise.
fn matrix_approx(mat1: &Matrix<f64>, mat2: &Matrix<f64>) -> bool {
    const EPS: f64 = 1e-4;

    let mat1 = mat1.iter();
    let mut mat2 = mat2.iter();

    for x in mat1 {
        let y = mat2.next().unwrap();

        if (x - y).abs() > EPS {
            return false;
        }
    }

    true
}

/// Builds a reflection matrix from a given vector.
pub fn refl_mat(n: Vector<f64>) -> Matrix<f64> {
    let dim = n.nrows();
    let nn = n.norm_squared();

    // Reflects every basis vector, builds a matrix from all of their images.
    Matrix::from_columns(
        &Matrix::identity(dim, dim)
            .column_iter()
            .map(|v| v - (2.0 * v.dot(&n) / nn) * &n)
            .collect::<Vec<_>>(),
    )
}

impl GenIter {
    /// Builds a new group from a set of generators.
    fn new(dim: usize, generators: Vec<Matrix<f64>>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(OrdMatrix(Matrix::identity(dim, dim)));

        Self {
            dim,
            generators,
            elements: BTreeMap::new(),
            queue,
            gen_idx: 0,
        }
    }

    /// Inserts a new element into the group. Returns whether the element is new.
    fn insert(&mut self, el: Matrix<f64>) -> bool {
        let el = OrdMatrix(el);

        // If the element is a repeat.
        if let Some(value) = self.elements.insert(el.clone(), 1) {
            // Bumps the value by 1, or removes the element if this is the last
            // time we'll find the element.
            if value != self.generators.len() - 1 {
                self.elements.insert(el, value + 1);
            } else {
                // self.elements.remove(&el);
            }

            false
        }
        // If the element is new, we add it to the queue as well.
        else {
            &self.elements;
            self.queue.push_back(el);

            true
        }
    }

    /// Gets the next element and the next generator to attempt to multiply.
    /// Advances the iterator.
    fn next_el_gen(&mut self) -> Option<[Matrix<f64>; 2]> {
        let el = self.queue.front()?.0.clone();
        let gen = self.generators[self.gen_idx].clone();

        // Advances the indices.
        self.gen_idx += 1;
        if self.gen_idx == self.generators.len() {
            self.gen_idx = 0;
            self.queue.pop_front();
        }

        Some([el, gen])
    }

    /// Multiplies the current element times the current generator, determines
    /// if it is a new element. Advances the iterator.
    fn try_next(&mut self) -> GroupNext {
        // If there's a next element and generator.
        if let Some([el, gen]) = self.next_el_gen() {
            let new_el = el * gen;

            // If the group element is new.
            if self.insert(new_el.clone()) {
                GroupNext::New(new_el)
            }
            // If we found a repeat.
            else {
                GroupNext::Repeat
            }
        }
        // If we already went through the entire group.
        else {
            GroupNext::None
        }
    }
}

#[cfg(test)]
mod tests {
    use gcd::Gcd;

    use super::*;

    /// Tests a given symmetry group.
    fn test(group: Group, order: usize, rot_order: usize, name: &str) {
        let group = group.cache();

        assert_eq!(
            group.clone().order(),
            order,
            "{} does not have the expected order.",
            name
        );

        assert_eq!(
            group.rotations().order(),
            rot_order,
            "The rotational group of {} does not have the expected order.",
            name
        );
    }

    /// Tests the trivial group in various dimensions.
    #[test]
    fn i() {
        for n in 1..=10 {
            test(Group::trivial(n), 1, 1, &format!("I^{}", n))
        }
    }

    /// Tests the group consisting of the identity and a central inversion in
    /// various dimensions.
    #[test]
    fn c2() {
        for n in 1..=10 {
            test(
                Group::central_inv(n),
                2,
                (n + 1) % 2 + 1,
                &format!("<-I^{}>", n),
            )
        }
    }

    /// Tests the I2(*n*) symmetries, which correspond to the symmetries of a
    /// regular *n*-gon.
    #[test]
    fn i2() {
        for n in 2..=10 {
            for d in 1..n {
                if n.gcd(d) != 1 {
                    continue;
                }

                test(cox!(n as f64 / d as f64), 2 * n, n, &format!("I2({})", n));
            }
        }
    }

    /// Tests the A*n* symmetries, which correspond to the symmetries of the
    /// regular simplices.
    #[test]
    fn a() {
        let mut order = 2;

        for n in 2..=5 {
            order *= n + 1;

            test(cox!(3.0; n - 1), order, order / 2, &format!("A{}", n))
        }
    }

    #[test]
    fn a_quat() {
        test(
            cox!(3.0, 3.0).left_quaternions().unwrap(),
            24,
            24,
            &"Quaternion A3",
        )
    }

    #[test]
    fn double_an() {
        let mut order = 4;

        for n in 2..=5 {
            order *= n + 1;

            test(
                Group::direct_product(cox!(3.0; n - 1), Group::central_inv(n), n, &product::matrix),
                order,
                order / 2,
                &format!("A{}", n),
            )
        }
    }

    /// Tests the B*n* symmetries, which correspond to the symmetries of the
    /// regular hypercube and orthoplex.
    #[test]
    fn b() {
        let mut order = 2;

        for n in 2..=5 {
            // A better cox! macro would make this unnecessary.
            let mut cox = vec![3.0; n - 1];
            cox[0] = 4.0;
            let cox = CoxMatrix::from_lin_diagram(cox);

            order *= n * 2;

            test(
                Group::cox_group(cox).unwrap(),
                order,
                order / 2,
                &format!("B{}", n),
            )
        }
    }

    /// Tests the H*n* symmetries, which correspond to the symmetries of a
    /// regular dodecahedron and a regular hecatonicosachoron.
    #[test]
    fn h() {
        test(cox!(5.0, 3.0), 120, 60, &"H3");
        test(cox!(5.0, 3.0, 3.0), 14400, 7200, &"H4");
    }

    /// Tests the E*n* symmetries... or rather, tests one of them. E7 and E8 are
    /// currently out of reach.
    #[test]
    fn e() {
        // In the future, we'll have better code for this, I hope.
        let e6 = Group::cox_group(CoxMatrix(Matrix::from_iterator(
            6,
            6,
            vec![
                1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0,
                2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                3.0, 2.0, 2.0, 1.0,
            ]
            .into_iter(),
        )))
        .unwrap();

        test(e6, 51840, 25920, &"E6");
    }

    #[test]
    /// Tests the direct product of A3 with itself.
    fn a3xa3() {
        let a3 = cox!(3.0, 3.0);
        let g = Group::direct_product(a3.clone(), a3.clone(), 6, &product::direct_sum);
        test(g, 576, 288, &"A3×A3");
    }
}
