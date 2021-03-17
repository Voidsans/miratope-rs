use std::io::Result;
use std::path::Path;
use std::{collections::HashMap, fs::read};

use super::*;

<<<<<<< Updated upstream
fn strip_comments(src: &mut String) {
    while let Some(com_start) = src.chars().position(|c| c == '#') {
        let line_end = com_start
            + src[com_start..]
                .chars()
                .position(|c| c == '\n')
                .unwrap_or_else(|| src.len() - com_start - 1);
        src.drain(com_start..=line_end);
    }
}

fn homogenize_whitespace(src: &mut String) {
    *src = src.trim().to_string();
    let mut new_src = String::with_capacity(src.capacity());
    let mut was_ws = false;

    for c in src.chars() {
        if c.is_whitespace() && !was_ws {
            new_src.push(' ');
            was_ws = true;
        } else {
            new_src.push(c);
            was_ws = false;
        }
=======
use super::{super::*, Abstract, Concrete, Element, ElementList, Polytope, RankVec};

/// Gets the name for an element with a given rank.
fn element_name(rank: isize) -> String {
    match super::ELEMENT_NAMES.get(rank as usize) {
        Some(&name) => String::from(name),
        None => rank.to_string() + "-elements",
>>>>>>> Stashed changes
    }

    *src = new_src;
}

fn read_usize(chars: &mut impl Iterator<Item = char>) -> usize {
    let mut n = 0;

    while let Some(c @ '0'..='9') = chars.next() {
        n *= 10;
        n += (c as usize) - ('0' as usize);
    }

    n
}

<<<<<<< Updated upstream
fn read_f64(chars: &mut impl Iterator<Item = char>) -> f64 {
    //let mut lookahead = chars.enumerate().find_map(|(i, c)| c);
    todo!()
}

fn get_elem_nums(chars: &mut impl Iterator<Item = char>, dim: usize) -> Vec<usize> {
    let mut num_elems = Vec::with_capacity(dim);

    for _ in 0..dim {
        chars.next();
        num_elems.push(read_usize(chars) as usize);
    }

    // 2-elements go before 1-elements, we're undoing that.
    if dim >= 3 {
        num_elems.swap(1, 2);
    }

    num_elems
}

pub fn polytope_from_off_src(mut src: String) -> Polytope {
    strip_comments(&mut src);
    homogenize_whitespace(&mut src);

    let mut chars = src.chars();
    let mut dim = read_usize(&mut chars);

    // This assumes our OFF file isn't 0D.
    if dim == 0 {
        dim = 3;
    }
=======
/// Gets the number of elements from the OFF file.
/// This includes components iff dim ≤ 2, as this makes things easier down the line.
fn get_el_nums<'a>(rank: isize, toks: &mut impl Iterator<Item = &'a str>) -> Vec<usize> {
    let rank = rank as usize;
    let mut el_nums = Vec::with_capacity(rank);

    // Reads entries one by one.
    for _ in 0..rank {
        let num_elem = toks.next().expect("OFF file ended unexpectedly.");
        el_nums.push(num_elem.parse().expect("could not parse as integer"));
    }

    // A point has a single component (itself)
    if rank == 0 {
        el_nums.push(1);
    }
    // A dyad has twice as many vertices as components.
    else if rank == 1 {
        let comps = el_nums[0] / 2;
        el_nums.push(comps);
    } else {
        // A polygon always has as many vertices as edges.
        if rank == 2 {
            el_nums.push(el_nums[0]);
        }
>>>>>>> Stashed changes

    // Checks for our magic word.
    if [chars.next(), chars.next(), chars.next()] != [Some('O'), Some('F'), Some('F')] {
        panic!("ayo this file's not an OFF")
    }

    let num_elems = get_elem_nums(&mut chars, dim);

    // Reads all vertices.
    let num_verts = num_elems[0];
    let mut verts = Vec::with_capacity(num_verts);

    // Add each vertex to the vector.
    for _ in 0..num_verts {
        let mut vert = Vec::with_capacity(dim);

        for _ in 0..dim {
            chars.next();
            vert.push(read_f64(&mut chars));
        }

        verts.push(vert.into());
    }

<<<<<<< Updated upstream
    // Reads edges and faces.
    let mut elements = Vec::with_capacity(dim as usize);
    let (num_edges, num_faces) = (num_elems[1], num_elems[2]);
    let (mut edges, mut faces) = (Vec::with_capacity(num_edges), Vec::with_capacity(num_faces));
=======
    vertices
}

/// Reads the faces from the OFF file and gets the edges and faces from them.
/// Since the OFF file doesn't store edges explicitly, this is harder than reading
/// general elements.
fn parse_edges_and_faces<'a>(
    num_edges: usize,
    num_faces: usize,
    toks: &mut impl Iterator<Item = &'a str>,
) -> (ElementList, ElementList) {
    let mut edges = ElementList::with_capacity(num_edges);
    let mut faces = ElementList::with_capacity(num_faces);

>>>>>>> Stashed changes
    let mut hash_edges = HashMap::new();

    // Add each face to the element list.
    for _ in 0..num_faces {
        chars.next();
        let face_sub_count = read_usize(&mut chars);

<<<<<<< Updated upstream
        let mut face = Vec::with_capacity(face_sub_count);
        let mut verts = Vec::with_capacity(face_sub_count);

        // Reads all vertices of the face.
        for _ in 0..face_sub_count {
            chars.next();
            verts.push(read_usize(&mut chars));
        }

        // Gets all edges of the face.
        for i in 0..face_sub_count {
            let edge = vec![verts[i], verts[(i + 1) % face_sub_count]]; // Missing sort!
            let edge_idx = hash_edges.get(&edge);

            match edge_idx {
                None => {
                    // Is the clone really necessary?
                    hash_edges.insert(edge.clone(), edges.len());
                    face.push(edges.len());
                    edges.push(edge);
                }
                Some(idx) => {
                    face.push(*idx);
                }
=======
        let mut face = Element {
            subs: Vec::with_capacity(face_sub_num),
        };
        let mut face_verts = Vec::with_capacity(face_sub_num);

        // Reads all vertices of the face.
        for _ in 0..face_sub_num {
            face_verts.push(
                toks.next()
                    .expect("OFF file ended unexpectedly.")
                    .parse()
                    .expect("Integer parsing failed!"),
            );
        }

        // Gets all edges of the face.
        for i in 0..face_sub_num {
            let mut edge = Element {
                subs: vec![face_verts[i], face_verts[(i + 1) % face_sub_num]],
            };
            edge.subs.sort_unstable();

            if let Some(idx) = hash_edges.get(&edge) {
                face.subs.push(*idx);
            } else {
                hash_edges.insert(edge.clone(), edges.len());
                face.subs.push(edges.len());
                edges.push(edge);
>>>>>>> Stashed changes
            }
        }

        faces.push(face);
    }

    // The number of edges in the file should match the number of read edges, though this isn't obligatory.
    if edges.len() != num_edges {
<<<<<<< Updated upstream
        println!("Edge count doesn't match expected edge count!");
    }

    elements.push(edges);
    elements.push(faces);

    // Adds all higher elements.
    for d in 3..(dim - 1) {
        let num_els = num_elems[d];
        let mut els = Vec::with_capacity(num_els);

        // Adds every d-element to the element list.
        for _ in 0..num_els {
            chars.next();
            let el_sub_count = read_usize(&mut chars);
            let mut el = Vec::with_capacity(el_sub_count);

            // Reads all sub-elements of the d-element.
            for _ in 0..el_sub_count {
                chars.next();
                el.push(read_usize(&mut chars));
            }

            els.push(el);
        }
=======
        println!("Edge num doesn't match expected edge num!");
    }

    (edges, faces)
}

/// Reads the next set of elements from the OFF file, starting from cells.
pub fn parse_els<'a>(num_el: usize, toks: &mut impl Iterator<Item = &'a str>) -> ElementList {
    let mut els_subs = ElementList::with_capacity(num_el);

    // Adds every d-element to the element list.
    for _ in 0..num_el {
        let el_sub_num = toks
            .next()
            .expect("OFF file ended unexpectedly.")
            .parse()
            .expect("Integer parsing failed!");
        let mut subs = Vec::with_capacity(el_sub_num);

        // Reads all sub-elements of the d-element.
        for _ in 0..el_sub_num {
            let el_sub = toks.next().expect("OFF file ended unexpectedly.");
            subs.push(el_sub.parse().expect("Integer parsing failed!"));
        }

        els_subs.push(Element { subs });
    }

    els_subs
}

/// Builds a [`Polytope`] from the string representation of an OFF file.
pub fn from_src(src: String) -> Concrete {
    let mut toks = data_tokens(&src);
    let rank = {
        let first = toks.next().expect("OFF file empty");
        let rank = first.strip_suffix("OFF").expect("no \"OFF\" detected");

        if rank.is_empty() {
            3
        } else {
            rank.parse()
                .expect("could not parse dimension as an integer")
        }
    };

    // Deals with dumb degenerate cases.
    if rank == -1 {
        return Concrete::nullitope();
    } else if rank == 0 {
        return Concrete::point();
    }

    let num_elems = get_el_nums(rank, &mut toks);
    let vertices = parse_vertices(num_elems[0], rank as usize, &mut toks);
    let mut abs = Abstract::with_rank(rank);

    // Adds nullitope and vertices.
    abs.push_min();
    abs.push_vertices(vertices.len());

    // Reads edges and faces.
    if rank >= 2 {
        let (edges, faces) = parse_edges_and_faces(num_elems[1], num_elems[2], &mut toks);
        abs.push(edges);
        abs.push(faces);
    }

    // Adds all higher elements.
    for &num_el in num_elems.iter().take(rank as usize).skip(3) {
        abs.push(parse_els(num_el, &mut toks));
    }

    // Caps the abstract polytope, returns the concrete one.
    abs.push_max();
    Concrete { vertices, abs }
}

/// Loads a polytope from a file path.
pub fn from_path(fp: &impl AsRef<Path>) -> IoResult<Concrete> {
    Ok(from_src(String::from_utf8(std::fs::read(fp)?).unwrap()))
}

/// A set of options to be used when saving the OFF file.
#[derive(Clone, Copy)]
pub struct OFFOptions {
    /// Whether the OFF file should have comments specifying each face type.
    pub comments: bool,
}

impl Default for OFFOptions {
    fn default() -> Self {
        OFFOptions { comments: true }
    }
}

fn write_el_counts(off: &mut String, opt: &OFFOptions, mut el_counts: RankVec<usize>) {
    let rank = el_counts.rank();

    // # Vertices, Faces, Edges, ...
    if opt.comments {
        off.push_str("\n# Vertices");

        let mut element_names = Vec::with_capacity((rank - 1) as usize);

        for r in 1..rank {
            element_names.push(element_name(r));
        }

        if element_names.len() >= 2 {
            element_names.swap(0, 1);
        }

        for element_name in element_names {
            off.push_str(", ");
            off.push_str(&element_name);
        }

        off.push('\n');
    }

    // Swaps edges and faces, because OFF format bad.
    if rank >= 3 {
        el_counts.swap(1, 2);
    }

    for r in 0..rank {
        off.push_str(&el_counts[r].to_string());
        off.push(' ');
    }

    off.push('\n');
}

/// Writes the vertices of a polytope into an OFF file.
fn write_vertices(off: &mut String, opt: &OFFOptions, vertices: &[Point]) {
    // # Vertices
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(0));
        off.push('\n');
    }

    // Adds the coordinates.
    for v in vertices {
        for c in v.into_iter() {
            off.push_str(&c.to_string());
            off.push(' ');
        }
        off.push('\n');
    }
}

/// Gets and writes the faces of a polytope into an OFF file.
fn write_faces(
    off: &mut String,
    opt: &OFFOptions,
    dim: usize,
    edges: &ElementList,
    faces: &ElementList,
) {
    // # Faces
    if opt.comments {
        let el_name = if dim > 2 {
            element_name(2)
        } else {
            super::COMPONENTS.to_string()
        };
>>>>>>> Stashed changes

        elements.push(els);
    }

<<<<<<< Updated upstream
    Polytope::new(verts, elements)
}

pub fn open_off(fp: &Path) -> Result<Polytope> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
=======
    // Retrieves the faces, writes them line by line.
    for face in faces.iter() {
        off.push_str(&face.subs.len().to_string());

        // Maps an OFF index into a graph index.
        let mut hash_edges = HashMap::new();
        let mut graph = Graph::new_undirected();

        // Maps the vertex indices to consecutive integers from 0.
        for &edge_idx in &face.subs {
            let edge = &edges[edge_idx];
            let mut hash_edge = Vec::with_capacity(2);

            for &vertex_idx in &edge.subs {
                match hash_edges.get(&vertex_idx) {
                    Some(&idx) => hash_edge.push(idx),
                    None => {
                        let idx = hash_edges.len();
                        hash_edges.insert(vertex_idx, idx);
                        hash_edge.push(idx);

                        graph.add_node(vertex_idx);
                    }
                }
            }
        }

        // There should be as many graph indices as edges on the face.
        // Otherwise, something went wrong.
        debug_assert_eq!(
            hash_edges.len(),
            face.subs.len(),
            "Faces don't have the same number of edges as there are in the polytope!"
        );

        // Adds the edges to the graph.
        for &edge_idx in &face.subs {
            let edge = &edges[edge_idx];
            graph.add_edge(
                NodeIndex::new(*hash_edges.get(&edge.subs[0]).unwrap()),
                NodeIndex::new(*hash_edges.get(&edge.subs[1]).unwrap()),
                (),
            );
        }

        // Retrieves the cycle of vertices.
        let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
        while let Some(nx) = dfs.next(&graph) {
            off.push(' ');
            off.push_str(&graph[nx].to_string());
        }
        off.push('\n');
    }
}

/// Writes the n-elements of a polytope into an OFF file.
fn write_els(off: &mut String, opt: &OFFOptions, rank: isize, els: &[Element]) {
    // # n-elements
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(rank));
        off.push('\n');
    }

    // Adds the elements' indices.
    for el in els {
        off.push_str(&el.subs.len().to_string());

        for &sub in &el.subs {
            off.push(' ');
            off.push_str(&sub.to_string());
        }

        off.push('\n');
    }
}

/// Converts a polytope into an OFF file.
pub fn to_src(p: &Concrete, opt: OFFOptions) -> String {
    let rank = p.rank();
    let vertices = &p.vertices;
    let abs = &p.abs;
    let mut off = String::new();

    // Blatant advertising.
    if opt.comments {
        off += &format!(
            "# Generated using Miratope v{} (https://github.com/OfficialURL/miratope-rs)\n",
            env!("CARGO_PKG_VERSION")
        );
    }

    // Writes header.
    if rank != 3 {
        off += &rank.to_string();
    }
    off += "OFF\n";

    // If we have a nullitope or point on our hands, that is all.
    if rank < 1 {
        return off;
    }

    // Adds the element counts.
    write_el_counts(&mut off, &opt, p.el_counts());

    // Adds vertex coordinates.
    write_vertices(&mut off, &opt, vertices);

    // Adds faces.
    if rank >= 2 {
        write_faces(&mut off, &opt, rank as usize, &abs[1], &abs[2]);
    }

    // Adds the rest of the elements.
    for r in 3..rank {
        write_els(&mut off, &opt, r, &abs[r]);
    }

    off
}

/// Writes a polytope's OFF file in a specified file path.
pub fn to_path(fp: &impl AsRef<Path>, p: &Concrete, opt: OFFOptions) -> IoResult<()> {
    std::fs::write(fp, to_src(p, opt))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Used to test a particular polytope.
    fn test_shape(mut p: Concrete, el_nums: Vec<usize>) {
        // Checks that element counts match up.
        for r in -1..=p.rank() {
            assert_eq!(p.el_count(r), el_nums[(r + 1) as usize]);
        }

        // Checks that the polytope can be reloaded correctly.
        p = from_src(to_src(&p, Default::default()));
        for r in -1..=p.rank() {
            assert_eq!(p.el_count(r), el_nums[(r + 1) as usize]);
        }
    }

    #[test]
    /// Checks that a point has the correct amount of elements.
    fn point_nums() {
        let point: Concrete = from_src("0OFF".to_string()).into();

        test_shape(point, vec![1])
    }

    #[test]
    /// Checks that a dyad has the correct amount of elements.
    fn dyad_nums() {
        let dyad: Concrete = from_src("1OFF 2 -1 1 0 1".to_string()).into();

        test_shape(dyad, vec![2, 1])
    }

    #[test]
    /// Checks that a hexagon has the correct amount of elements.
    fn hig_nums() {
        let hig: Concrete = from_src(
            "2OFF 6 1 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 6 0 1 2 3 4 5".to_string()
        ).into();

        test_shape(hig, vec![6, 6, 1])
    }

    #[test]
    /// Checks that a hexagram has the correct amount of elements.
    fn shig_nums() {
        let shig: Concrete = from_src(
            "2OFF 6 2 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 3 0 2 4 3 1 3 5".to_string()
        ).into();

        test_shape(shig, vec![6, 6, 2])
    }

    #[test]
    /// Checks that a tetrahedron has the correct amount of elements.
    fn tet_nums() {
        let tet: Concrete = from_src(
            "OFF 4 4 6 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2".to_string(),
        )
        .into();

        test_shape(tet, vec![4, 6, 4, 1])
    }

    #[test]
    /// Checks that a 2-tetrahedron compund has the correct amount of elements.
    fn so_nums() {
        let so: Concrete = from_src(
            "OFF 8 8 12 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 -1 1 1 1 -1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2 3 4 5 6 3 7 4 6 3 4 5 7 3 7 5 6 ".to_string(),
        )
        .into();

        test_shape(so, vec![8, 12, 8, 2])
    }

    #[test]
    /// Checks that a pentachoron has the correct amount of elements.
    fn pen_nums() {
        let pen: Concrete = from_src(
            "4OFF 5 10 10 5 0.158113883008419 0.204124145231932 0.288675134594813 0.5 0.158113883008419 0.204124145231932 0.288675134594813 -0.5 0.158113883008419 0.204124145231932 -0.577350269189626 0 0.158113883008419 -0.612372435695794 0 0 -0.632455532033676 0 0 0 3 0 3 4 3 0 2 4 3 2 3 4 3 0 2 3 3 0 1 4 3 1 3 4 3 0 1 3 3 1 2 4 3 0 1 2 3 1 2 3 4 0 1 2 3 4 0 4 5 6 4 1 4 7 8 4 2 5 7 9 4 3 6 8 9"
                .to_string(),
        )
        .into();

        test_shape(pen, vec![5, 10, 10, 5, 1])
    }

    #[test]
    /// Checks that comments are correctly parsed.
    fn comments() {
        let tet: Concrete = from_src(
            "# So
            OFF # this
            4 4 6 # is
            # a # test # of
            1 1 1 # the 1234 5678
            1 -1 -1 # comment 987
            -1 1 -1 # removal 654
            -1 -1 1 # system 321
            3 0 1 2 #let #us #see
            3 3 0 2# if
            3 0 1 3#it
            3 3 1 2#works!#"
                .to_string(),
        )
        .into();

        test_shape(tet, vec![4, 6, 4, 1])
    }

    #[test]
    #[should_panic(expected = "OFF file empty")]
    fn empty() {
        Concrete::from(from_src("".to_string()));
    }

    #[test]
    #[should_panic(expected = "no \"OFF\" detected")]
    fn magic_num() {
        Concrete::from(from_src("foo bar".to_string()));
    }
>>>>>>> Stashed changes
}
