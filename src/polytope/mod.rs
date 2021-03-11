//! Contains all of the methods used to operate on
//! (polytopes)[https://polytope.miraheze.org/wiki/Polytope].

use bevy::prelude::Mesh;
use bevy::render::mesh::Indices;
use bevy::render::pipeline::PrimitiveTopology;
use petgraph::{graph::Graph, prelude::NodeIndex, Undirected};
use serde::{Deserialize, Serialize};

use geometry::Point;

pub mod convex;
pub mod geometry;
pub mod off;
pub mod shapes;

pub type Element = Vec<usize>;
pub type ElementList = Vec<Element>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolytopeSerde {
    pub vertices: Vec<Point>,
    pub elements: Vec<ElementList>,
}

pub struct ElementType {
    multiplicity: usize,
    facet_count: usize,
    volume: Option<f64>,
}

impl From<Polytope> for PolytopeSerde {
    fn from(p: Polytope) -> Self {
        PolytopeSerde {
            vertices: p.vertices,
            elements: p.elements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polytope {
    /// The concrete vertices of the polytope.
    ///
    /// Here, "dimension" is the number being encoded by the [`Dim`][nalgebra::Dim]
    /// used in [`Point`]s. This is not to be confused by the rank, which is defined
    /// in terms of the length of the element list.
    ///
    /// # Assumptions
    ///
    /// * All points have the same dimension.
    /// * There are neither `NaN` nor `Infinity` values.
    pub vertices: Vec<Point>,

    /// A compact representation of the incidences of the polytope. This vector
    /// stores the edges, faces, ..., of the polytope, all the way up
    /// to the components.
    ///
    /// The (d – 1)-th entry of this vector corresponds to the indices of the
    /// (d – 1)-elements that form a given d-element.
    pub elements: Vec<ElementList>,

    extra_vertices: Vec<Point>,

    triangles: Vec<[usize; 3]>,
}

impl Polytope {
    /// Builds a new [Polytope] with the given vertices and elements.
    pub fn new(vertices: Vec<Point>, elements: Vec<ElementList>) -> Self {
        let (extra_vertices, triangles) = if elements.len() >= 2 {
            Self::triangulate(&vertices, &elements[0], &elements[1])
        } else {
            (vec![], vec![])
        };

        Polytope {
            vertices,
            elements,
            extra_vertices,
            triangles,
        }
    }

    /// Builds a polytope and auto-generates its connected components.
    pub fn new_wo_comps(vertices: Vec<Point>, mut elements: Vec<ElementList>) -> Self {
        let rank = elements.len() + 1;
        assert!(rank >= 2, "new_wo_comps can only work on 2D and higher!");

        let num_ridges = if rank >= 3 {
            elements[rank - 3].len()
        } else {
            vertices.len()
        };

        let facets = &elements[rank - 2];
        let num_facets = facets.len();

        // g is the incidence graph of ridges and facets.
        // The ith ridge is stored at position i.
        // The ith facet is stored at position num_ridges + i.
        let mut graph: Graph<(), (), Undirected> = Graph::new_undirected();

        // Is there not any sort of extend function we can use for syntactic sugar?
        for _ in 0..(num_ridges + num_facets) {
            graph.add_node(());
        }

        // We add an edge for each adjacent facet and ridge.
        for (i, f) in facets.iter().enumerate() {
            for r in f.iter() {
                graph.add_edge(NodeIndex::new(*r), NodeIndex::new(num_ridges + i), ());
            }
        }

        // Converts the connected components of our facet + ridge graph
        // into just the lists of facets in each component.
        let g_comps = petgraph::algo::kosaraju_scc(&graph);
        let mut comps = Vec::with_capacity(g_comps.len());

        for g_comp in g_comps.iter() {
            let mut comp = Vec::new();

            for idx in g_comp.iter() {
                let idx: usize = idx.index();

                if idx < num_ridges {
                    comp.push(idx);
                }
            }

            comps.push(comp);
        }

        elements.push(comps);

        Polytope::new(vertices, elements)
    }

    fn triangulate(
        _vertices: &[Point],
        edges: &[Element],
        faces: &[Element],
    ) -> (Vec<Point>, Vec<[usize; 3]>) {
        let extra_vertices = Vec::new();
        let mut triangles = Vec::new();

        for face in faces {
            let edge_i = *face.first().expect("no indices in face");
            let vert_i = edges
                .get(edge_i)
                .expect("Index out of bounds: you probably screwed up the polytope's indices.")[0];

            for verts in face[1..].iter().map(|&i| {
                let edge = &edges[i];
                assert_eq!(edge.len(), 2, "edges has more than two elements");
                [edge[0], edge[1]]
            }) {
                let [vert_j, vert_k]: [usize; 2] = verts;
                if vert_i != vert_j && vert_i != vert_k {
                    triangles.push([vert_i, vert_j, vert_k]);
                }
            }
        }

        (extra_vertices, triangles)
    }

    /// Returns the rank of the polytope.
    pub fn rank(&self) -> usize {
        self.elements.len()
    }

    pub fn dimension(&self) -> usize {
        let vertex = self.vertices.get(0);

        if let Some(vertex) = vertex {
            vertex.len()
        } else {
            0
        }
    }

    /// Gets the element counts of a polytope.
    /// The n-th entry corresponds to the amount of n-elements.
    pub fn el_nums(&self) -> Vec<usize> {
        let mut nums = Vec::with_capacity(self.elements.len() + 1);
        nums.push(self.vertices.len());

        for e in self.elements.iter() {
            nums.push(e.len());
        }

        nums
    }

    /// Gets the coordinates of the vertices of the polytope, including the
    /// extra vertices needed for the mesh.
    fn get_vertex_coords(&self) -> Vec<[f32; 3]> {
        self.vertices
            .iter()
            .chain(self.extra_vertices.iter())
            .map(|point| {
                let mut iter = point.iter().copied().take(3);
                let x = iter.next().unwrap_or(0.0);
                let y = iter.next().unwrap_or(0.0);
                let z = iter.next().unwrap_or(0.0);
                [x as f32, y as f32, z as f32]
            })
            .collect()
    }

    /// Builds a mesh from a polytope.
    pub fn get_mesh(&self) -> Mesh {
        let vertices = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(self.triangles.len() * 3);
        for &[i, j, k] in &self.triangles {
            indices.push(i as u16);
            indices.push(j as u16);
            indices.push(k as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
    }

    /// Builds a wireframe from a polytope.
    pub fn get_wireframe(&self) -> Mesh {
        let edges = &self.elements[0];
        let vertices: Vec<_> = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(edges.len() * 2);
        for edge in edges {
            indices.push(edge[0] as u16);
            indices.push(edge[1] as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));
        mesh
    }

    /// Gets the gravicenter of a polytope.
    pub fn gravicenter(&self) -> Point {
        let dim = self.dimension();
        let mut g: Point = vec![0.0; dim].into();
        let vertices = &self.vertices;

        for v in vertices {
            g += v;
        }

        g / (vertices.len() as f64)
    }

    /// Gets the edge lengths of all edges in the polytope, in order.
    pub fn edge_lengths(&self) -> Vec<f64> {
        let vertices = &self.vertices;
        let mut edge_lengths = Vec::new();

        // If there are no edges, we just return the empty vector.
        if let Some(edges) = self.elements.get(0) {
            edge_lengths.reserve_exact(edges.len());

            for edge in edges {
                let (sub1, sub2) = (edge[0], edge[1]);

                edge_lengths.push((&vertices[sub1] - &vertices[sub2]).norm());
            }
        }

        edge_lengths
    }

    /// Checks whether all of the edge lengths of the polytope are approximately
    /// equal to a specified one.
    pub fn is_equilateral_with_len(&self, len: f64) -> bool {
        const EPS: f64 = 1e-9;
        let edge_lengths = self.edge_lengths().into_iter();

        // Checks that every other edge length is equal to the first.
        for edge_len in edge_lengths {
            if (edge_len - len).abs() > EPS {
                return false;
            }
        }

        true
    }

    /// Checks whether a polytope is equilateral to a fixed precision.
    pub fn is_equilateral(&self) -> bool {
        if let Some(edges) = self.elements.get(0) {
            if let Some(edge) = edges.get(0) {
                let vertices = edge.iter().map(|&v| &self.vertices[v]).collect::<Vec<_>>();
                let (v0, v1) = (vertices[0], vertices[1]);

                return self.is_equilateral_with_len((v0 - v1).norm());
            }
        }

        true
    }

    /// I haven't actually implemented this in the general case.
    pub fn midradius(&self) -> f64 {
        let vertices = &self.vertices;
        let edges = &self.elements[0];
        let edge = &edges[0];
        let (sub1, sub2) = (edge[0], edge[1]);

        (&vertices[sub1] + &vertices[sub2]).norm() / 2.0
    }

    /// Gets the element "types" of a polytope.
    pub fn get_element_types(&self) -> Vec<Vec<ElementType>> {
        const EPS: f64 = 1e-9;

        let rank = self.rank();
        let mut element_types = Vec::with_capacity(rank + 1);
        let el_nums = self.el_nums();

        for &el_num in &el_nums {
            element_types.push(Vec::with_capacity(el_num));
        }

        let mut types = Vec::with_capacity(rank);
        let elements = &self.elements;
        let mut output = Vec::with_capacity(rank + 1);

        // There's only one type of point.
        types.push(Vec::new());
        output.push(vec![ElementType {
            multiplicity: el_nums[0],
            facet_count: 1,
            volume: Some(1.0),
        }]);

        // The edge types are the edges of each different length.
        let mut edge_types = Vec::new();
        for length in &self.edge_lengths() {
            if let Some(index) = edge_types
                .iter()
                .position(|&x: &f64| (x - length).abs() < EPS)
            {
                element_types[1].push(index);
            } else {
                element_types[1].push(edge_types.len());
                edge_types.push(*length);
            }
        }
        types.push(edge_types);

        let mut edge_types = Vec::with_capacity(types[1].len());
        for (idx, &edge_type) in types[1].iter().enumerate() {
            edge_types.push(ElementType {
                multiplicity: element_types[1].iter().filter(|&x| *x == idx).count(),
                facet_count: 2,
                volume: Some(edge_type),
            })
        }
        output.push(edge_types);

        for d in 2..=rank {
            types.push(Vec::new());
            for el in &elements[d - 1] {
                let count = el.len();
                if let Some(index) = types[d].iter().position(|&x| x == count as f64) {
                    element_types[d].push(index);
                } else {
                    element_types[d].push(types[d].len());
                    types[d].push(count as f64);
                }
            }
            let mut types_in_rank = Vec::with_capacity(types[d].len());
            for i in 0..types[d].len() {
                types_in_rank.push(ElementType {
                    multiplicity: element_types[d].iter().filter(|&x| *x == i).count(),
                    facet_count: types[d][i] as usize,
                    volume: None,
                })
            }
            output.push(types_in_rank);
        }

        output
    }

    pub fn print_element_types(&self) -> String {
        let types = self.get_element_types();
        let mut output = String::new();
        let el_names = vec![
            "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta",
            "Xenna", "Daka", "Henda",
        ];
        let el_suffixes = vec![
            "", "", "gon", "hedron", "choron", "teron", "peton", "exon", "zetton", "yotton",
            "xennon", "dakon",
        ];

        output.push_str(&format!("{}:\n", el_names[0]));
        for t in &types[0] {
            output.push_str(&format!("{}\n", t.multiplicity));
        }
        output.push_str("\n");

        output.push_str(&format!("{}:\n", el_names[1]));
        for t in &types[1] {
            output.push_str(&format!(
                "{} of length {}\n",
                t.multiplicity,
                t.volume.unwrap_or(0.0)
            ));
        }
        output.push_str("\n");

        for d in 2..self.rank() {
            output.push_str(&format!("{}:\n", el_names[d]));
            for t in &types[d] {
                output.push_str(&format!(
                    "{} × {}-{}\n",
                    t.multiplicity, t.facet_count, el_suffixes[d]
                ));
            }
            output.push_str("\n");
        }

        output.push_str("Components:\n");
        for t in &types[self.rank()] {
            output.push_str(&format!(
                "{} × {}-{}\n",
                t.multiplicity,
                t.facet_count,
                el_suffixes[self.rank()]
            ));
        }

        return output;
    }
}

impl From<PolytopeSerde> for Polytope {
    fn from(ps: PolytopeSerde) -> Self {
        Polytope::new(ps.vertices, ps.elements)
    }
}
