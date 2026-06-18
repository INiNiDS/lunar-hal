use crate::{Camera, SectorKey, UpdatePayload};
use lunar_structures::ResponseStar;
use rand::RngExt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnemyAction {
    Nothing,
    AttackingStar(Star),
    AttackingEnemy(EnemyId),
    Escaping { speed: f32, direction: (f32, f32) },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnemyType {
    Aggressive,
    Defensive,
     Opportunistic,
     Random,
}

impl From<u32> for EnemyType {
    fn from(value: u32) -> Self {
        match value {
            0 => EnemyType::Aggressive,
            1 => EnemyType::Defensive,
            2 => EnemyType::Opportunistic,
            _ => EnemyType::Random
        }
    }
}

pub type Star = (f32, f32);
pub type EnemyId = u32;

#[derive(Clone, Debug, Copy)]
pub struct Enemy {
    id: usize,
    coordinates: (f32, f32),
    radius: f32,
    action: EnemyAction,
    /// The sector the enemy last found itself in.
    current_sector: Option<SectorKey>,
    enemy_type: EnemyType
}

impl Enemy {
    pub fn new(radius: f32, id: usize) -> Self {
        Enemy {
            id,
            coordinates: (0.0, 0.0),
            radius,
            action: EnemyAction::Nothing,
            current_sector: None,
            enemy_type: Self::generate_random_type(radius),
        }
    }

    pub fn reset(&mut self) {
        self.coordinates = (0.0, 0.0);
        self.action = EnemyAction::Nothing;
        self.current_sector = None;
        self.enemy_type = Self::generate_random_type(self.radius);
    }

    #[inline]
    pub fn update(
         &mut self,
        enemies: &[Enemy],
        payload: &UpdatePayload,
        camera: &Camera,
    ) {
        self.decide_what_to_do(enemies, payload, camera);
    }

    /// Decide what to do this tick. Receives the full update payload
    /// so that the enemy can inspect the current sector, its stars,
    /// camera movement (accumulated over the rolling window), and
    /// player actions without reaching into the game state directly.
    #[inline]
    fn decide_what_to_do(
        &mut self,
        enemies: &[Enemy],
        payload: &UpdatePayload,
        camera: &Camera,
    ) {
        self.current_sector = payload.current_sector;

        match self.action {
            EnemyAction::Nothing => {
                self.on_idle(&payload.sector_stars, enemies, payload, camera);
            }
            EnemyAction::AttackingEnemy(_target_id) => {todo!()}
            EnemyAction::AttackingStar(_target_star) => {todo!()}
            EnemyAction::Escaping { .. } => {todo!()}
        }
    }

    #[inline]
    fn on_idle(
        &mut self,
        sector_stars: &[ResponseStar],
        enemies: &[Enemy],
        payload: &UpdatePayload,
        camera: &Camera,
    ) {
        let delta: (f32, f32) = payload.camera_movement.offset_delta;
        let current_chunk = self.current_sector;

        let star_count = sector_stars.len();
        if star_count == 0 {
            return;
        }

        todo!()
    }

    fn generate_random_type(radius: f32) -> EnemyType {
        let mut rng = rand::rng();
        let x = rng.random_range(0..radius as u32);
        EnemyType::from(x)
    }
}


pub struct EnemyInstance {
    enemies: Vec<Enemy>,
}


impl EnemyInstance {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let enemies_count = rng.random_range(0..=5);
        let enemies: Vec<Enemy> = (0..enemies_count)
        .map(|id| Enemy::new(rng.random_range(5.0..20.0), id))
        .collect();
        EnemyInstance {
            enemies,
        }
    }

    pub fn update(&mut self, payload: &UpdatePayload, camera: Camera) {
        for i in 0..self.enemies.len() {
            let (left, right) = self.enemies.split_at_mut(i);
            let (enemy, rest) = right.split_first_mut().unwrap();
            enemy.update(&[left, rest].concat(), payload, &camera); // TODO: Maybe we can avoid the allocations here...
        }
    }

    pub fn reset_enemy(&mut self, id: usize) {
        if let Some(enemy) = self.enemies.get_mut(id) {
            enemy.reset();
        }
    }

    pub fn reset_enemies(&mut self) {
        for id in 0..self.enemies.len() {
            self.reset_enemy(id);
        }
    }
}