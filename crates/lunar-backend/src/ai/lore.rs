use lunar_utils::env::get_lunar_models_dir;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::rng::SimpleRng;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoreEntry {
    pub id: u32,
    pub star_type: String,
    pub spectral_class: String,
    pub designated_name: String,
    pub visual_profile: String,
    pub description: String,
    pub system_lore: String,
}

pub struct LoreCache {
    entries: Vec<LoreEntry>,
}

pub(crate) static LORE_CACHE: RwLock<Option<Arc<LoreCache>>> = RwLock::const_new(None);

pub async fn get_lore_cache() -> Option<Arc<LoreCache>> {
    if let Some(cached) = LORE_CACHE.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_lore_cache().await;
    *LORE_CACHE.write().await = loaded.clone();
    loaded
}

pub(crate) async fn load_lore_cache() -> Option<Arc<LoreCache>> {
    let models_dir = get_lunar_models_dir();
    let path = models_dir.join("stellar_lore_cache.json");

    if !path.exists() {
        println!("  Lore cache not found ({})", path.display());
        return None;
    }

    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return None,
    };

    let entries: Vec<LoreEntry> = match serde_json::from_str(&json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  Failed to parse lore cache: {e}");
            return None;
        }
    };

    println!(
        "  Lore cache loaded: {} entries from {}",
        entries.len(),
        path.display()
    );
    Some(Arc::new(LoreCache { entries }))
}

impl LoreCache {
    pub fn pick(&self, seed: u64) -> Option<&LoreEntry> {
        if self.entries.is_empty() {
            return None;
        }
        let mut rng = SimpleRng::new(seed);
        let idx = (rng.next_u64() as usize) % self.entries.len();
        Some(&self.entries[idx])
    }

    pub fn pick_by_class(&self, seed: u64, spectral_class: &str) -> Option<&LoreEntry> {
        let matching: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.spectral_class == spectral_class)
            .map(|(i, _)| i)
            .collect();

        if matching.is_empty() {
            return self.pick(seed);
        }

        let mut rng = SimpleRng::new(seed);
        let idx = (rng.next_u64() as usize) % matching.len();
        Some(&self.entries[matching[idx]])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
