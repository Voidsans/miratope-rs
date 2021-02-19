use nalgebra::*;

use super::{ElementList, Polytope};

type Matrix = nalgebra::DMatrix<f64>;

pub fn dual(p: &Polytope) -> Polytope {
    let el_counts = p.el_counts();
    let r = p.rank();

    let vertices = &p.vertices;
    let elements = &p.elements[..r - 1];
    let components = &p.elements[r];

    let du_vertices = vertices.clone();
    let mut du_elements: Vec<ElementList> = Vec::with_capacity(el_counts[r - 1]);
    let mut du_components = Vec::with_capacity(components.len());

    // Builds the dual incidence graph.
    for (d, els) in elements.iter().enumerate() {
        let c = el_counts[r - 1 - d];
        let mut du_els = Vec::with_capacity(c);

        for _ in 0..c {
            du_els.push(vec![]);
        }

        for (i, el) in els.iter().enumerate() {
            for &sub in el {
                let mut du_el = &mut du_els[sub];
                du_el.push(i);
            }
        }

        du_elements.push(du_els);
    }

    du_elements.push(du_components);
    Polytope::new(du_vertices, du_elements)
}
