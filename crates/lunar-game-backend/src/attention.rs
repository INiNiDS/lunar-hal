use std::collections::HashMap;

use lunar_structures::ResponseStar;

/// Per-star attention data tracked by the Attention Map.
///
/// These two values drive utility curves that let the AI choose
/// the moment when the player is least expecting an attack.
#[derive(Clone, Debug, Default)]
pub struct AttentionEntry {
    /// Time in seconds since the player last looked at this star.
    pub t_neglect: f32,
    /// World-space distance (parsecs) from the mouse cursor to this star.
    pub d_mouse: f32,
}

/// Карта внимания игрока — динамическая карта того, на какие
/// планеты/звёзды игрок смотрит, а какие игнорирует.
///
/// Для каждой звезды хранится:
/// - `t_neglect` — время, в течение которого игрок не смотрел на
///   звезду / не наводил курсор.
/// - `d_mouse` — расстояние от текущего положения мыши до звезды в
///   мировых координатах.
///
/// Используется врагом (Enemy) для выбора оптимального момента атаки.
#[derive(Clone, Debug, Default)]
pub struct AttentionMap {
    entries: HashMap<u32, AttentionEntry>,
}

impl AttentionMap {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Продвигает таймеры невнимания и пересчитывает расстояния до
    /// мыши для переданного набора звёзд.
    ///
    /// - `dt` — дельта времени в секундах с прошлого тика.
    /// - `mouse_world` — положение мыши в мировых координатах
    ///   (парсеки), либо `None` если позиция неизвестна.
    /// - `stars` — звёзды, для которых обновляется внимание.
    pub fn tick(&mut self, dt: f32, mouse_world: Option<(f32, f32)>, stars: &[ResponseStar]) {
        for star in stars {
            let entry = self
                .entries
                .entry(star.id)
                .or_insert_with(AttentionEntry::default);
            entry.t_neglect += dt;

            if let Some((mx, my)) = mouse_world {
                let dx = star.x - mx;
                let dy = star.y - my;
                entry.d_mouse = (dx * dx + dy * dy).sqrt();
            }
        }
    }

    /// Сбрасывает `t_neglect` для конкретной звезды — игрок посмотрел
    /// на неё или навёл курсор.
    pub fn on_player_look(&mut self, star_id: u32) {
        if let Some(entry) = self.entries.get_mut(&star_id) {
            entry.t_neglect = 0.0;
        }
    }

    /// Сбрасывает `t_neglect` для всех звёзд в радиусе `radius` от
    /// позиции мыши.
    pub fn reset_nearby(&mut self, mouse_world: (f32, f32), radius: f32, stars: &[ResponseStar]) {
        let r2 = radius * radius;
        for star in stars {
            let dx = star.x - mouse_world.0;
            let dy = star.y - mouse_world.1;
            if dx * dx + dy * dy <= r2 {
                self.on_player_look(star.id);
            }
        }
    }

    pub fn get(&self, star_id: u32) -> Option<&AttentionEntry> {
        self.entries.get(&star_id)
    }

    /// Полный снапшот для передачи во врага или в UI.
    pub fn snapshot(&self) -> HashMap<u32, AttentionEntry> {
        self.entries.clone()
    }
}
