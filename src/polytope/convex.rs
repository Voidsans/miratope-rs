
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use super::{
    geometry::{Hyperplane, Point},
    Abstract, Concrete, Element, ElementList,
};

use nalgebra::DMatrix;
use scapegoat::SGSet as BTreeSet;
use std::fmt::Debug;

#[derive(PartialEq, Debug)]
enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Represents a vertex set in a convex hull.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Debug)]
struct VertexSet {
    /// The sorted indices of the vertices on the ridge.
    vertices: Vec<usize>,

    /// The vertices in each ridge are oriented. The fact that we need to order
    /// them makes this inconvenient.
    ///
    /// This variable stores whether two vertices have to be swapped to have the
    /// correct orientation.
    orientation: bool,
}

/// Determines whether two permutations of a vector have different parities.
/// Uses cycle sort to sort a vector into the other, and records the parity of
/// the number of swaps.
///
/// Has O(n) complexity.
fn parity(mut p0: Vec<usize>, p1: &[usize]) -> bool {
    let mut parity = false;
    let mut hash = HashMap::new();

    // Maps each value in p0 to its index in the array.
    for (i, &val) in p1.iter().enumerate() {
        hash.insert(val, i);
    }

    // Cycle sort, but we use the hash table to find the correct position of
    // every element in constant time.
    for i in 0..p0.len() {
        loop {
            let j = *hash.get(&p0[i]).unwrap();

            if i == j {
                break;
            }

            p0.swap(i, j);
            parity = !parity;
        }
    }

    parity
}

impl VertexSet {
    /// Sorts the vertices and updates the orientation accordingly.
    fn sort(&mut self) {
        let old_vertices = self.vertices.clone();
        self.vertices.sort_unstable();

        self.orientation ^= parity(old_vertices, &self.vertices);
    }
}

/// Returns the sign of a number, up to some imprecision.
fn sign(x: f64) -> Sign {
    const EPS: f64 = 1e-9;

    if x > EPS {
        Sign::Positive
    } else if x < -EPS {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

/// Returns the sign of the hypervolume of a simplex specified by a set of
/// n-dimensional points.
fn sign_hypervolume(simplex: &[&Point], orientation: bool) -> Sign {
    let dim = simplex.len() - 1;
    let mut m = DMatrix::from_element(dim + 1, dim + 1, 1.0);

    debug_assert_eq!(dim, simplex[0].len());

    for (j, v) in simplex.iter().enumerate() {
        for (i, &c) in v.iter().enumerate() {
            m[(i, j)] = c;
        }
    }

    let flip = if orientation { 1.0 } else { -1.0 };
    sign(flip * m.determinant())
}

/// Finds a single ridge on the convex hull. As a side effect, sorts the
/// vertices of the polytope lexicographically.
fn get_hull_ridge(vertices: &mut [Point]) -> VertexSet {
    // Sorts the vertices lexicographically, by turning them into vecs and
    // applying their ordering.
    vertices.sort_unstable_by(|a, b| {
        a.iter()
            .cloned()
            .collect::<Vec<_>>()
            .partial_cmp(&b.iter().cloned().collect::<Vec<_>>())
            .expect("NaN vertex found!")
    });

    let mut vertices = vertices.iter().enumerate();
    let (_, v0) = vertices.next().unwrap();
    let dim = v0.len();
    let mut h = Hyperplane::new(v0.clone());

    let mut ridge = vec![0];

    // Takes the first `dim - 1` points in lexicographic order, so that no three
    // are collinear, no four are coplanar... these ought to create a ridge.
    while h.rank != dim - 2 {
        let (i, v) = vertices
            .next()
            .expect("Polytope has higher dimension than rank!");

        if h.add(v.clone()).is_some() {
            ridge.push(i);
        }
    }

    // The starting orientation is irrelevant.
    VertexSet {
        vertices: ridge,
        orientation: true,
    }
}

/// Finds the index of the closest leftmost vertex relative to a ridge.
/// "Leftmost" will depend on the orientation of the ridge.
fn leftmost_vertex(vertices: &[Point], ridge: &VertexSet) -> Vec<usize> {
    debug_assert!(is_sorted(&ridge.vertices));

    let mut leftmost_vertices = Vec::new();
    let mut facet: Vec<_> = ridge.vertices.iter().map(|&idx| &vertices[idx]).collect();

    let mut vertex_iter = vertices.iter().enumerate();

    // We find a starting vertex not on the ridge.
    let mut h = Hyperplane::from_points(facet.iter().cloned().cloned().collect());
    loop {
        let (i, v0) = vertex_iter.next().unwrap();

        // The ridge should be sorted, so we can optimize this.
        if h.add(v0.clone()).is_some() {
            leftmost_vertices.push(i);
            facet.push(v0);

            break;
        }
    }

    // The previous to last vertex on the facet will always be one of the
    // leftmost vertices found so far.

    // We compare with all of the other vertices.
    for (i, v) in vertex_iter {
        facet.push(v);

        // If the new vertex is to the left of the previous leftmost one:
        if sign_hypervolume(&facet, ridge.orientation) == Sign::Positive {
            // Resets leftmost vertices.
            leftmost_vertices.clear();
            leftmost_vertices.push(i);

            // Adds new leftmost to the facet.
            let len = facet.len();
            facet.swap(len - 2, len - 1);
        }

        facet.pop();
    }

    debug_assert!(!leftmost_vertices.is_empty());
    leftmost_vertices
}

/// Gets the new ridges that have to be searched, in the correct orientation.
fn get_new_ridges(old_ridge: &VertexSet, new_vertices: &[usize]) -> Vec<VertexSet> {
    // THIS CURRENTLY ONLY DEALS WITH SIMPLICIAL FACETS.
    let new_vertex = new_vertices[0];

    let len = old_ridge.vertices.len();
    let mut new_ridges = Vec::with_capacity(len);

    // We simply add the new vertex in each position of the ridge.
    // These ridges should have the correct orientation.
    for k in 0..len {
        let mut new_ridge = old_ridge.clone();
        new_ridge.vertices[k] = new_vertex;
        new_ridge.sort();

        new_ridges.push(new_ridge);
    }

    new_ridges
}

/// Checks whether an array is sorted in increasing order. Used only for debugs.
fn is_sorted(el: &[usize]) -> bool {
    for i in 0..(el.len() - 1) {
        if el[i] >= el[i + 1] {
            return false;
        }
    }

    true
}

/// Finds the common elements of two arrays.
/// # Assumptions:
/// * Both arrays must be sorted in increasing order.
fn common(el0: &[usize], el1: &[usize]) -> Vec<usize> {
    // Nightly Rust has el.is_sorted().
    debug_assert!(is_sorted(el0));
    debug_assert!(is_sorted(el1));

    let mut common = Vec::new();
    let (mut i, mut j) = (0, 0);

    // Again, nightly rust would allow us to compress this.
    while let Some(&sub0) = el0.get(i) {
        if let Some(sub1) = el1.get(j) {
            match sub0.cmp(sub1) {
                Ordering::Equal => {
                    common.push(sub0);
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Less => i += 1,
            };
        } else {
            break;
        }
    }

    common
}

/// Checks whether a given vertex set actually generates a valid d-polytope.
fn check_subelement(vertices: &[Point], el: &[usize], rank: isize) -> bool {
    let rank = rank as usize;

    // A d-element must have at least d + 1 vertices.
    if el.len() < rank + 1 {
        return false;
    }

    // It is possible for two d-elements to share more than d
    // elements without them being a common (d - 1)-element, but
    // only when d >= 4.
    if rank >= 4 {
        // The hyperplane of the intersection of the elements.
        let h = Hyperplane::from_points(
            el.iter()
                .map(|&sub| vertices[sub].clone())
                .collect::<Vec<Point>>(),
        );

        // If this hyperplane does not have the correct dimension, it
        // can't actually be a subelement.
        if h.rank != rank - 1 {
            return false;
        }
    }

    true
}

/// Gift wrapping is only able to find the vertices of the facets of the polytope.
/// This function retrieves all other elements from them.
fn get_polytope_from_facets(vertices: Vec<Point>, facets: ElementList) -> Concrete {
    let dim = vertices[0].len();

    // Initializes the abstract polytope.
    let mut abs = Abstract::with_rank(dim as isize);
    abs.push_min();
    for _ in 0..(dim as isize) {
        abs.push(ElementList::new());
    }

    // Adds everything else.
    let mut els_verts = facets;

    for r in (2..(dim as isize)).rev() {
        let mut subs_map = HashMap::new();
        let len = els_verts.len();

        // Each element of `els_verts` contains the indices of the vertices,
        // and not the subelements of the element. This vector fixes that.
        let mut els_subs = ElementList::with_capacity(len);

        for _ in 0..len {
            els_subs.push(Element::new());
        }

        // Checks every pair of d-elements to see if their intersection forms
        // a (d - 1)-element.
        for i in 0..(len - 1) {
            for j in (i + 1)..len {
                // The intersection of the two elements.
                let el = common(&els_verts[i].subs, &els_verts[j].subs);

                // Checks that el actually has the correct rank.
                if !check_subelement(&vertices, &el, r - 1) {
                    continue;
                }

                match subs_map.get(&el) {
                    Some(&idx) => {
                        if !els_subs[i].subs.contains(&idx) {
                            els_subs[i].subs.push(idx);
                        }

                        if !els_subs[j].subs.contains(&idx) {
                            els_subs[j].subs.push(idx);
                        }
                    }
                    None => {
                        let idx = subs_map.len();

                        els_subs[i].subs.push(idx);
                        els_subs[j].subs.push(idx);

                        subs_map.insert(el, idx);
                    }
                }
            }
        }

        els_verts = ElementList::new();
        els_verts.resize(subs_map.len(), Element { subs: vec![] });

        for (subs, idx) in subs_map {
            els_verts[idx] = Element { subs };
        }

        abs[r] = els_subs;
    }

    // At this point, els_verts contains, for each edge, the indices of its
    // vertices, which are precisely the rank 1 elements of the polytope.
    abs[1] = els_verts;

    // Adds the vertices and the maximal element.
    abs[0] = ElementList::vertices(vertices.len());
    abs.push_max();

    Concrete::new(vertices, abs)
}

/// Builds the convex hull of a set of vertices. Uses the gift wrapping algorithm.
pub fn convex_hull(mut vertices: Vec<Point>) -> Concrete {
    let mut facets = HashSet::new();
    let mut ridges = BTreeSet::new();

    // Gets first ridge, reorders elements in the process.
    ridges.insert(get_hull_ridge(&mut vertices));

    // While there's still a ridge we need to check...
    while let Some(old_ridge) = ridges.pop_first() {
        let mut new_vertices = leftmost_vertex(&vertices, &old_ridge);
        let new_ridges = get_new_ridges(&old_ridge, &new_vertices);

        let mut facet = Element {
            subs: old_ridge.vertices.clone(),
        };
        facet.subs.append(&mut new_vertices);
        facet.subs.sort_unstable();

        // We skip the facet if it isn't new.
        if facets.contains(&facet) {
            continue;
        }

        // For each new ridge:
        for new_ridge in new_ridges {
            // If this is the second time we find this ridge, it means we've
            // already added both of the facets corresponding to it.
            if ridges.contains(&new_ridge) {
                ridges.remove(&new_ridge);
            }
            // Else, we still need to find the other facet.
            else {
                ridges.insert(new_ridge);
            }
        }

        facets.insert(facet);
    }

    get_polytope_from_facets(vertices, ElementList(facets.into_iter().collect()))
}

impl Concrete {
    pub fn convex_hull(&self) -> Concrete {
        convex_hull(self.vertices.clone())
    }
}
