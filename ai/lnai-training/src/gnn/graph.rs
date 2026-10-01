
use lnai_models::{GraphBatch, compute_sparse_knn_graph};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone)]
pub struct SpatialIndex3D {
    cell_size: f32,
    inv_cell_size: f32,
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    coords: Vec<[f32; 3]>,
}

impl SpatialIndex3D {
    pub fn build(coords: Vec<[f32; 3]>, cell_size: f32) -> Self {
        let cell_size = if cell_size <= 1e-4 { 10.0 } else { cell_size };
        let inv_cell_size = 1.0 / cell_size;
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (idx, &[x, y, z]) in coords.iter().enumerate() {
            let cx = (x * inv_cell_size).floor() as i32;
            let cy = (y * inv_cell_size).floor() as i32;
            let cz = (z * inv_cell_size).floor() as i32;
            grid.entry((cx, cy, cz)).or_default().push(idx);
        }

        Self {
            cell_size,
            inv_cell_size,
            grid,
            coords,
        }
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn query_knn(&self, point: [f32; 3], k: usize) -> Vec<(usize, f32)> {
        let n = self.coords.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let cx = (point[0] * self.inv_cell_size).floor() as i32;
        let cy = (point[1] * self.inv_cell_size).floor() as i32;
        let cz = (point[2] * self.inv_cell_size).floor() as i32;

        let mut candidates = Vec::new();
        let mut radius = 1;

        while candidates.len() < k && radius <= 5 {
            candidates.clear();
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if let Some(indices) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            candidates.extend_from_slice(indices);
                        }
                    }
                }
            }
            radius += 1;
        }

        if candidates.len() < k {
            candidates = (0..n).collect();
        }

        let mut dists: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|idx| {
                let c = self.coords[idx];
                let dx = point[0] - c[0];
                let dy = point[1] - c[1];
                let dz = point[2] - c[2];
                (idx, dx * dx + dy * dy + dz * dz)
            })
            .collect();

        dists.sort_unstable_by_key(|&(idx, _)| idx);
        dists.dedup_by_key(|&mut (idx, _)| idx);

        let effective_k = k.min(dists.len());
        if effective_k > 0 && effective_k < dists.len() {
            dists.select_nth_unstable_by(effective_k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists[..effective_k]
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        dists.truncate(effective_k);
        dists
    }
}

pub fn hash_coords(coords: &[[f32; 3]], k: usize) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325 ^ (k as u64);
    for &[x, y, z] in coords {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001B3);
        h ^= y.to_bits() as u64;
        h = h.wrapping_mul(0x100000001B3);
        h ^= z.to_bits() as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

static GLOBAL_GRAPH_CACHE: OnceLock<Mutex<HashMap<u64, Arc<GraphBatch>>>> = OnceLock::new();

pub struct GraphCache;

impl GraphCache {
    pub fn get_or_build(coords: &[[f32; 3]], k: usize) -> Arc<GraphBatch> {
        let key = hash_coords(coords, k);
        let cache = GLOBAL_GRAPH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache.lock().expect("graph cache lock");
            if let Some(hit) = guard.get(&key) {
                return Arc::clone(hit);
            }
        }

        let graph = Arc::new(compute_sparse_knn_graph(coords, k));
        let mut guard = cache.lock().expect("graph cache lock");
        guard.insert(key, Arc::clone(&graph));
        graph
    }

    pub fn clear() {
        if let Some(cache) = GLOBAL_GRAPH_CACHE.get() {
            cache.lock().expect("graph cache lock").clear();
        }
    }

    pub fn len() -> usize {
        GLOBAL_GRAPH_CACHE
            .get()
            .map(|c| c.lock().expect("graph cache lock").len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_index_finds_nearest_point() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
        ];
        let index = SpatialIndex3D::build(coords, 2.0);
        let neighbors = index.query_knn([0.1, 0.0, 0.0], 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 0);

        assert_eq!(neighbors[1].0, 1);

    }

    #[test]
    fn graph_cache_returns_identical_instances() {
        let coords = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let g1 = GraphCache::get_or_build(&coords, 2);
        let g2 = GraphCache::get_or_build(&coords, 2);
        assert!(Arc::ptr_eq(&g1, &g2), "Cache must return the same Arc");
        assert_eq!(g1.num_nodes, 3);
    }
}
