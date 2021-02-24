use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::Path;
use petgraph::{graph::NodeIndex, visit::Dfs, Graph};
use super::*;

fn get_dimension(reader: &mut impl Iterator<Item=u8>) -> usize {
    reader.next().expect("Could not read dimension") as usize
}

enum ElemCounts {
    Dim0,
    /// Does contain edge info
    Dim1(usize, usize),
    /// Does not contain edge info
    DimN(usize, Vec<usize>),
}

fn get_dimension_counts_d1_aux(reader: &mut impl Iterator<Item=u8>) -> Option<ElemCounts> {
    let mut buffer = [0; std::mem::size_of::<usize>()];
    buffer[0] = reader.next()?;
    buffer[1] = reader.next()?;
    buffer[2] = reader.next()?;
    let c0 = usize::from_le_bytes(buffer);
    buffer[0] = reader.next()?;
    buffer[1] = reader.next()?;
    buffer[2] = reader.next()?;
    let c1 = usize::from_le_bytes(buffer);
    Some(ElemCounts::Dim1(c0, c1))
}

fn get_dimension_counts(reader: &mut impl Iterator<Item=u8>, mut dim: usize) -> Option<ElemCounts> {
    if dim == 0 {
        return Some(ElemCounts::Dim0);
    } else if dim == 1 {
        return get_dimension_counts_d1_aux(reader);
    }

    dim -= 2;
    let mut buffer = [0; std::mem::size_of::<usize>()];
    buffer[0] = reader.next()?;
    buffer[1] = reader.next()?;
    buffer[2] = reader.next()?;
    let c0 = usize::from_le_bytes(buffer);
    let mut counts = Vec::with_capacity(dim);

    for _ in 0..=dim {
        buffer[0] = reader.next()?;
        buffer[1] = reader.next()?;
        buffer[2] = reader.next()?;
        counts.push(usize::from_le_bytes(buffer));
    }

    Some(ElemCounts::DimN(c0, counts))
}

fn get_vertices(
    reader: &mut impl Iterator<Item=u8>,
    dim: usize,
    count: usize,
) -> Option<Vec<Point>> {
    debug_assert_ne!(dim, 0);
    let mut verts = Vec::with_capacity(count);
    let mut buffer = [0; std::mem::size_of::<f32>()];
    let mut point: Point = vec![0.0; dim].into();

    for _ in 0..count {
        for i in 0..dim {
            buffer[0] = reader.next()?;
            buffer[1] = reader.next()?;
            buffer[2] = reader.next()?;
            buffer[3] = reader.next()?;
            point[i] = f32::from_le_bytes(buffer) as f64;
        }

        verts.push(point.clone())
    }

    Some(verts)
}

fn get_indices_d1(
    reader: &mut impl Iterator<Item=u8>,
    dim: usize,
    edges: usize,
) -> Option<Vec<Vec<Vec<usize>>>> {
    debug_assert_ne!(dim, 0);
    let mut buffer = [0; std::mem::size_of::<usize>()];
    let mut indices = Vec::with_capacity(edges);

    for _ in 0..edges {
        buffer[0] = reader.next()?;
        buffer[1] = reader.next()?;
        buffer[2] = reader.next()?;
        let i0 = usize::from_le_bytes(buffer);
        buffer[0] = reader.next()?;
        buffer[1] = reader.next()?;
        buffer[2] = reader.next()?;
        let i1 = usize::from_le_bytes(buffer);
        indices.push(vec![i0, i1]);
    }

    Some(vec![indices])
}

fn get_indices_dn(
    reader: &mut impl Iterator<Item=u8>,
    dim: usize,
    count: usize,
) -> Option<Vec<Vec<Vec<usize>>>> {
    debug_assert_ne!(dim, 0);
    
}
