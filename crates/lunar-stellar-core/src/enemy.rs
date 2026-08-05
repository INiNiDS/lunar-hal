use crate::{SectorKey, UpdatePayload};
use lunar_structures::ResponseStar;
use rand::RngExt;

pub const STAR_MAX_HP: f32 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnemyAction {
    Nothing,
    AttackingStar(Star),
    AttackingEnemy(EnemyId),
    Flying(f32, f32),
    Escaping { direction: (f32, f32) },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnemyType {
    Tank,
    Thief,
    Invisible,
    Scavenger,
    Backstabber,
    Coward,
}

impl EnemyType {
    pub fn hp(&self) -> f32 {
        match self {
            EnemyType::Tank => 150.0,
            EnemyType::Thief => 50.0,
            EnemyType::Invisible => 100.0,
            EnemyType::Scavenger => 70.0,
            EnemyType::Backstabber => 90.0,
            EnemyType::Coward => 40.0,
        }
    }

    pub fn damage_per_hit(&self) -> f32 {
        match self {
            EnemyType::Tank => 12.0,
            EnemyType::Thief => 4.0,
            EnemyType::Invisible => 7.0,
            EnemyType::Scavenger => 5.0,
            EnemyType::Backstabber => 8.0,
            EnemyType::Coward => 3.0,
        }
    }

    pub fn attack_cooldown(&self) -> f32 {
        match self {
            EnemyType::Tank => 2.0,
            EnemyType::Thief => 0.4,
            EnemyType::Invisible => 1.0,
            EnemyType::Scavenger => 0.8,
            EnemyType::Backstabber => 1.2,
            EnemyType::Coward => 1.0,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            EnemyType::Tank => "Tank",
            EnemyType::Thief => "Thief",
            EnemyType::Invisible => "Invisible",
            EnemyType::Scavenger => "Scavenger",
            EnemyType::Backstabber => "Backstabber",
            EnemyType::Coward => "Coward",
        }
    }

    pub fn visibility(&self) -> f32 {
        match self {
            EnemyType::Invisible => 0.05,
            _ => 1.0,
        }
    }

    pub fn speed(&self) -> f32 {
        match self {
            EnemyType::Thief => 20.0,
            EnemyType::Scavenger | EnemyType::Coward => 15.0,
            EnemyType::Backstabber | EnemyType::Invisible => 10.0,
            EnemyType::Tank => 5.0,
        }
    }
}

impl From<u32> for EnemyType {
    fn from(value: u32) -> Self {
        match value % 6 {
            0 => EnemyType::Tank,
            1 => EnemyType::Thief,
            2 => EnemyType::Invisible,
            3 => EnemyType::Scavenger,
            4 => EnemyType::Backstabber,
            _ => EnemyType::Coward,
        }
    }
}

pub type Star = (f32, f32);
pub type EnemyId = u32;

#[derive(Clone, Debug)]
pub struct StarDamage {
    pub star_id: u32,
    pub position: Star,
    pub damage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnemyDamage {
    pub target_id: usize,
    pub damage: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Projectile {
    pub id: usize,
    pub coordinates: (f32, f32),
    pub target_star_id: u32,
    pub target_coordinates: (f32, f32),
    pub speed: f32,
    pub damage: f32,
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Enemy {
    pub id: usize,
    pub coordinates: (f32, f32),
    pub radius: f32,
    pub action: EnemyAction,
    pub current_sector: Option<SectorKey>,
    pub enemy_type: EnemyType,
    pub hp: f32,
    pub visibility: f32,
    pub attack_timer: f32,
}

impl Enemy {
    pub fn new(radius: f32, id: usize) -> Self {
        let enemy_type = Self::generate_random_type(radius);
        let hp = enemy_type.hp();
        let visibility = enemy_type.visibility();
        Enemy {
            id,
            coordinates: (0.0, 0.0),
            radius,
            action: EnemyAction::Nothing,
            current_sector: None,
            enemy_type,
            hp,
            visibility,
            attack_timer: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.coordinates = (0.0, 0.0);
        self.action = EnemyAction::Nothing;
        self.current_sector = None;
        self.enemy_type = Self::generate_random_type(self.radius);
        self.hp = self.enemy_type.hp();
        self.visibility = self.enemy_type.visibility();
        self.attack_timer = 0.0;
    }

    #[inline]
    pub fn update(
        &mut self,
        dt: f32,
        mouse_world: Option<(f32, f32)>,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
        enemy_damage: &mut Vec<EnemyDamage>,
        projectiles: &mut Vec<Projectile>,
    ) {
        if self.attack_timer > 0.0 {
            self.attack_timer = (self.attack_timer - dt).max(0.0);
        }
        self.update_stealth_and_timers(mouse_world, payload);
        self.decide_what_to_do(dt, left, right, payload, enemy_damage, projectiles);
    }

    pub fn make_visible(&mut self) {
        self.visibility = 1.0;
    }

    pub fn is_visible(&self) -> bool {
        self.visibility > 0.0
    }

    pub fn take_damage(&mut self, amount: f32) {
        self.hp -= amount;
    }

    pub fn take_damage_from_player(&mut self, amount: f32) {
        self.hp -= amount;
        if self.is_dead() {
            return;
        }

        match self.enemy_type {
            EnemyType::Thief | EnemyType::Invisible | EnemyType::Scavenger | EnemyType::Coward => {
                let mut rng = rand::rng();
                let angle = rng.random_range(0.0..std::f32::consts::TAU);
                self.action = EnemyAction::Escaping {
                    direction: (angle.cos(), angle.sin()),
                };
            }
            EnemyType::Backstabber => {
                if self.hp / self.enemy_type.hp() < 0.5 {
                    let mut rng = rand::rng();
                    let angle = rng.random_range(0.0..std::f32::consts::TAU);
                    self.action = EnemyAction::Escaping {
                        direction: (angle.cos(), angle.sin()),
                    };
                }
            }
            EnemyType::Tank => {}
        }
    }

    pub fn is_dead(&self) -> bool {
        self.hp <= 0.0
    }

    fn generate_random_type(radius: f32) -> EnemyType {
        let mut rng = rand::rng();
        let x = rng.random_range(0..radius as u32);
        EnemyType::from(x)
    }

    #[inline]
    fn decide_what_to_do(
        &mut self,
        dt: f32,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
        enemy_damage: &mut Vec<EnemyDamage>,
        projectiles: &mut Vec<Projectile>,
    ) {
        self.current_sector = payload.current_sector;

        self.react_to_attackers(left, right);

        match self.action {
            EnemyAction::Nothing => {
                self.on_idle(&payload.sector_stars, left, right, payload);
            }
            EnemyAction::AttackingEnemy(target_id) => {
                self.attack_enemy(dt, left, right, target_id, enemy_damage);
            }
            EnemyAction::Flying(target_x, target_y) => {
                self.flying(dt, target_x, target_y, left, right, payload);
            }
            EnemyAction::AttackingStar(target_star) => {
                self.attack_star(dt, target_star, payload, left, right, projectiles);
            }
            EnemyAction::Escaping { direction } => {
                self.escaping(dt, direction);
            }
        }
    }

    fn react_to_attackers(&mut self, left: &[Enemy], right: &[Enemy]) {
        if let Some(attacker) = self.find_attacker(left, right) {
            self.respond_to_attacker(attacker);
        }
    }

    fn find_attacker<'a>(&self, left: &'a [Enemy], right: &'a [Enemy]) -> Option<&'a Enemy> {
        left.iter().chain(right.iter()).find(|other| {
            if let EnemyAction::AttackingEnemy(target_id) = other.action {
                target_id as usize == self.id
            } else {
                false
            }
        })
    }

    fn respond_to_attacker(&mut self, attacker: &Enemy) {
        match self.enemy_type {
            EnemyType::Tank => {
                if !matches!(self.action, EnemyAction::AttackingEnemy(_)) {
                    self.action = EnemyAction::AttackingEnemy(attacker.id as u32);
                }
            }
            EnemyType::Backstabber => {
                let attacker_max_hp = attacker.enemy_type.hp();
                let attacker_hp_ratio = attacker.hp / attacker_max_hp;

                if attacker_hp_ratio < 0.4 {
                    if !matches!(self.action, EnemyAction::AttackingEnemy(_)) {
                        self.action = EnemyAction::AttackingEnemy(attacker.id as u32);
                    }
                } else {
                    let escape_dir = self.get_escape_direction(attacker.coordinates);
                    self.action = EnemyAction::Escaping {
                        direction: escape_dir,
                    };
                }
            }
            EnemyType::Thief | EnemyType::Invisible | EnemyType::Scavenger | EnemyType::Coward => {
                let escape_dir = self.get_escape_direction(attacker.coordinates);
                self.action = EnemyAction::Escaping {
                    direction: escape_dir,
                };
            }
        }
    }

    #[inline]
    fn update_stealth_and_timers(
        &mut self,
        mouse_world: Option<(f32, f32)>,
        payload: &UpdatePayload,
    ) {
        if self.enemy_type == EnemyType::Invisible {
            let is_hovered = if let Some(mw) = mouse_world {
                let dx = self.coordinates.0 - mw.0;
                let dy = self.coordinates.1 - mw.1;
                (dx * dx + dy * dy).sqrt() < 30.0
            } else {
                false
            };

            if is_hovered {
                self.visibility = 1.0;
            } else {
                let sd = payload.session_duration;
                if (sd.as_secs_f32() % 10.0) < 1.0 {
                    self.visibility = 0.5;
                } else {
                    self.visibility = 0.05;
                }
            }
        } else {
            self.visibility = 1.0;
        }
    }

    #[inline]
    fn on_idle(
        &mut self,
        sector_stars: &[ResponseStar],
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
    ) {
        let star_count = sector_stars.len();
        if star_count == 0 {
            return;
        }
        let mut best_action = EnemyAction::Nothing;
        let mut highest_utility = 0.1;

        for star in sector_stars {
            let utility = self.evaluate_star_utility(star, left, right, payload);
            if utility > highest_utility {
                highest_utility = utility;
                best_action = EnemyAction::AttackingStar((star.x, star.y));
            }
        }

        for enemy in left.iter().chain(right.iter()) {
            let utility = self.evaluate_enemy_utility(enemy);
            if utility > highest_utility {
                highest_utility = utility;
                best_action = EnemyAction::AttackingEnemy(enemy.id as u32);
            }
        }

        if self.enemy_type == EnemyType::Coward {
            if let Some(rival) = self.find_strong_rival_nearby(left, right, 150.0) {
                let escape_dir = self.get_escape_direction(rival.coordinates);
                self.action = EnemyAction::Escaping {
                    direction: escape_dir,
                };
                return;
            }
        }

        if highest_utility <= 0.1 && rand::rng().random_bool(0.03) {
            let mut rng = rand::rng();
            let patrol_x = self.coordinates.0 + rng.random_range(-120.0..120.0);
            let patrol_y = self.coordinates.1 + rng.random_range(-120.0..120.0);
            best_action = EnemyAction::Flying(patrol_x, patrol_y);
        }

        self.action = best_action;
    }

    fn evaluate_star_utility(
        &self,
        star: &ResponseStar,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
    ) -> f32 {
        if star.hp <= 0.0 {
            return 0.0;
        }

        let star_pos = (star.x, star.y);
        let dist = self.distance_to(star_pos);

        let attention = payload.attention_map.get(&star.id);
        let neglect_time = attention.map(|a| a.t_neglect).unwrap_or(60.0);
        let d_mouse = attention.map(|a| a.d_mouse).unwrap_or(100.0);
        let star_hp_ratio = (star.hp / STAR_MAX_HP).clamp(0.0, 1.0);

        match self.enemy_type {
            EnemyType::Tank => self.tank_star_utility(dist),
            EnemyType::Thief => self.thief_star_utility(dist, neglect_time, d_mouse),
            EnemyType::Invisible => self.invisible_star_utility(dist, neglect_time, d_mouse),
            EnemyType::Scavenger => {
                self.scavenger_star_utility(dist, star_pos, star_hp_ratio, left, right)
            }
            EnemyType::Coward => self.coward_star_utility(dist, star_pos, d_mouse, left, right),
            EnemyType::Backstabber => self.backstabber_star_utility(dist),
        }
    }

    #[inline]
    fn tank_star_utility(&self, dist: f32) -> f32 {
        let max_hp = self.enemy_type.hp();
        let hp_ratio = (max_hp - self.hp) / max_hp;
        let base_utility = 120.0 / (dist + 1.0);
        base_utility * (1.0 + hp_ratio)
    }

    #[inline]
    fn thief_star_utility(&self, dist: f32, neglect_time: f32, d_mouse: f32) -> f32 {
        if neglect_time > 15.0 && d_mouse > 15.0 {
            (neglect_time * 6.0) / (dist + 1.0)
        } else {
            0.0
        }
    }

    #[inline]
    fn invisible_star_utility(&self, dist: f32, neglect_time: f32, d_mouse: f32) -> f32 {
        let safety_factor = d_mouse.min(50.0);
        let neglect_factor = neglect_time.min(60.0);
        (safety_factor * neglect_factor) / (dist + 1.0)
    }

    #[inline]
    fn scavenger_star_utility(
        &self,
        dist: f32,
        star_pos: Star,
        star_hp_ratio: f32,
        left: &[Enemy],
        right: &[Enemy],
    ) -> f32 {
        if self.is_star_under_attack(star_pos, left, right) {
            let loot_greed = 1.0 - star_hp_ratio;
            (200.0 * loot_greed) / (dist + 1.0)
        } else {
            20.0 / (dist + 1.0)
        }
    }

    #[inline]
    fn coward_star_utility(
        &self,
        dist: f32,
        star_pos: Star,
        d_mouse: f32,
        left: &[Enemy],
        right: &[Enemy],
    ) -> f32 {
        let strong_rival_near_star = left.iter().chain(right.iter()).any(|other| {
            other.enemy_type == EnemyType::Tank && other.distance_to(star_pos) < 120.0
        });

        if strong_rival_near_star {
            0.0
        } else {
            let safety = d_mouse.min(30.0);
            (safety * 50.0) / (dist + 1.0)
        }
    }

    #[inline]
    fn backstabber_star_utility(&self, dist: f32) -> f32 {
        15.0 / (dist + 1.0)
    }

    fn evaluate_enemy_utility(&self, enemy: &Enemy) -> f32 {
        let dist = self.distance_to(enemy.coordinates);

        match self.enemy_type {
            EnemyType::Tank => {
                if dist < 80.0 {
                    20.0 / (dist + 1.0)
                } else {
                    0.0
                }
            }
            EnemyType::Backstabber => {
                let target_max_hp = enemy.enemy_type.hp();
                let target_hp_ratio = enemy.hp / target_max_hp;

                if target_hp_ratio < 0.35 && dist < 200.0 {
                    let execution_desire = 1.0 - target_hp_ratio;
                    (250.0 * execution_desire) / (dist + 1.0)
                } else {
                    0.0
                }
            }
            EnemyType::Scavenger => {
                let target_max_hp = enemy.enemy_type.hp();
                let target_hp_ratio = enemy.hp / target_max_hp;
                if enemy.enemy_type == EnemyType::Tank && target_hp_ratio < 0.4 && dist < 120.0 {
                    (150.0 * (1.0 - target_hp_ratio)) / (dist + 1.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    #[inline]
    fn flying(
        &mut self,
        dt: f32,
        target_x: f32,
        target_y: f32,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
    ) {
        self.escape_if_needed((target_x, target_y), left, right, payload);
        if let EnemyAction::Flying(target_x, target_y) = self.action {
            let dx = target_x - self.coordinates.0;
            let dy = target_y - self.coordinates.1;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < 5.0 {
                self.action = EnemyAction::Nothing;
            } else {
                let speed = self.enemy_type.speed();
                self.coordinates.0 += dx / distance * speed * dt;
                self.coordinates.1 += dy / distance * speed * dt;
            }
        }
    }

    #[inline]
    fn attack_star(
        &mut self,
        dt: f32,
        target_star: Star,
        payload: &UpdatePayload,
        left: &[Enemy],
        right: &[Enemy],
        projectiles: &mut Vec<Projectile>,
    ) {
        self.escape_if_needed(target_star, left, right, payload);

        let EnemyAction::AttackingStar(target_star) = self.action else {
            return;
        };

        let dx = target_star.0 - self.coordinates.0;
        let dy = target_star.1 - self.coordinates.1;
        let distance = (dx * dx + dy * dy).sqrt();

        let attack_range = 15.0;

        if distance > attack_range {
            let speed = self.enemy_type.speed();
            self.coordinates.0 += dx / distance * speed * dt;
            self.coordinates.1 += dy / distance * speed * dt;
            return;
        }

        let Some(s) = find_star_at(&payload.sector_stars, target_star) else {
            self.action = EnemyAction::Nothing;
            return;
        };

        if s.hp > 0.0 && self.attack_timer <= 0.0 {
            let bullet_id = rand::rng().random_range(1..1_000_000);
            projectiles.push(Projectile {
                id: bullet_id,
                coordinates: self.coordinates,
                target_star_id: s.id,
                target_coordinates: target_star,
                speed: 20.0,
                radius: 8.0,
                damage: self.enemy_type.damage_per_hit(),
            });
            self.attack_timer = self.enemy_type.attack_cooldown();
        }

        if s.hp <= 0.0 {
            self.action = EnemyAction::Nothing;
        }
    }

    #[inline]
    fn attack_enemy(
        &mut self,
        dt: f32,
        left: &[Enemy],
        right: &[Enemy],
        target_id: EnemyId,
        enemy_damage: &mut Vec<EnemyDamage>,
    ) {
        let target = left
            .iter()
            .chain(right.iter())
            .find(|e| e.id == target_id as usize);
        if let Some(target) = target {
            let dx = target.coordinates.0 - self.coordinates.0;
            let dy = target.coordinates.1 - self.coordinates.1;
            let distance = (dx * dx + dy * dy).sqrt();

            let attack_range = 2.0;

            if distance > attack_range {
                let speed = self.enemy_type.speed();
                self.coordinates.0 += dx / distance * speed * dt;
                self.coordinates.1 += dy / distance * speed * dt;
            } else {
                if self.attack_timer <= 0.0 {
                    enemy_damage.push(EnemyDamage {
                        target_id: target.id,
                        damage: self.enemy_type.damage_per_hit(),
                    });
                    self.attack_timer = self.enemy_type.attack_cooldown();
                }

                if target.hp <= 0.0 {
                    self.action = EnemyAction::Nothing;
                }
            }
        } else {
            self.action = EnemyAction::Nothing;
        }
    }

    #[inline]
    fn escaping(&mut self, dt: f32, direction: (f32, f32)) {
        let speed = self.enemy_type.speed();
        self.coordinates.0 += direction.0 * speed * dt;
        self.coordinates.1 += direction.1 * speed * dt;

        if rand::rng().random_bool(0.04) {
            self.action = EnemyAction::Nothing;
        }
    }

    fn escape_if_needed(
        &mut self,
        target_star: Star,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
    ) {
        if self.should_interrupt_attack(target_star, left, right, payload) {
            let direction = self.get_escape_direction(target_star);
            self.action = EnemyAction::Escaping { direction };
        }
    }

    fn should_interrupt_attack(
        &self,
        target_star: Star,
        left: &[Enemy],
        right: &[Enemy],
        payload: &UpdatePayload,
    ) -> bool {
        let Some(star) = find_star_at(&payload.sector_stars, target_star) else {
            return true;
        };

        if let Some(attention) = payload.attention_map.get(&star.id) {
            self.evaluate_interrupt(attention.t_neglect, attention.d_mouse, left, right)
        } else {
            false
        }
    }

    fn evaluate_interrupt(
        &self,
        t_neglect: f32,
        d_mouse: f32,
        left: &[Enemy],
        right: &[Enemy],
    ) -> bool {
        match self.enemy_type {
            EnemyType::Tank => false,
            EnemyType::Thief => t_neglect < 1.0 || d_mouse < 4.0,
            EnemyType::Invisible => t_neglect < 0.3 || d_mouse < 2.5,
            EnemyType::Scavenger => {
                let low_hp = self.hp / self.enemy_type.hp() < 0.3;
                t_neglect < 0.8 || d_mouse < 3.5 || low_hp
            }
            EnemyType::Backstabber => {
                let low_hp_colleague_nearby = left.iter().chain(right.iter()).any(|other| {
                    let max_hp = other.enemy_type.hp();
                    let hp_ratio = other.hp / max_hp;
                    hp_ratio < 0.35 && self.distance_to(other.coordinates) < 100.0
                });
                t_neglect < 1.0 || d_mouse < 3.0 || low_hp_colleague_nearby
            }
            EnemyType::Coward => t_neglect < 1.5 || d_mouse < 6.0,
        }
    }

    fn get_escape_direction(&self, threat_pos: (f32, f32)) -> (f32, f32) {
        let dx = self.coordinates.0 - threat_pos.0;
        let dy = self.coordinates.1 - threat_pos.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 0.0 {
            (dx / len, dy / len)
        } else {
            (1.0, 0.0)
        }
    }

    fn distance_to(&self, other_pos: (f32, f32)) -> f32 {
        ((self.coordinates.0 - other_pos.0).powi(2) + (self.coordinates.1 - other_pos.1).powi(2))
            .sqrt()
    }

    fn is_star_under_attack(&self, star_pos: Star, left: &[Enemy], right: &[Enemy]) -> bool {
        left.iter().chain(right.iter()).any(|other| {
            if let EnemyAction::AttackingStar(pos) = other.action {
                (pos.0 - star_pos.0).powi(2) + (pos.1 - star_pos.1).powi(2) < 0.01
            } else {
                false
            }
        })
    }

    fn find_strong_rival_nearby(
        &self,
        left: &[Enemy],
        right: &[Enemy],
        range: f32,
    ) -> Option<Enemy> {
        left.iter()
            .chain(right.iter())
            .filter(|other| {
                other.enemy_type == EnemyType::Tank && self.distance_to(other.coordinates) < range
            })
            .cloned()
            .next()
    }
}

fn find_star_at(stars: &[ResponseStar], target_star: Star) -> Option<&ResponseStar> {
    stars.iter().find(|s| {
        let sdx = s.x - target_star.0;
        let sdy = s.y - target_star.1;
        (sdx * sdx + sdy * sdy) < 1.0
    })
}

#[derive(Default)]
pub struct EnemyInstance {
    pub enemies: Vec<Enemy>,
    pub projectiles: Vec<Projectile>,
}

impl EnemyInstance {
    pub fn new() -> Self {
        EnemyInstance {
            enemies: Vec::new(),
            projectiles: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        mouse_world: Option<(f32, f32)>,
        payload: &UpdatePayload,
    ) -> Vec<StarDamage> {
        let mut star_damage = Vec::new();
        let mut enemy_damage: Vec<EnemyDamage> = Vec::new();

        let existing_projectiles_count = self.projectiles.len();

        for i in 0..self.enemies.len() {
            let (left, right) = self.enemies.split_at_mut(i);
            let (enemy, rest) = right.split_first_mut().unwrap();
            enemy.update(
                dt,
                mouse_world,
                left,
                rest,
                payload,
                &mut enemy_damage,
                &mut self.projectiles,
            );
        }

        for dmg in enemy_damage {
            if let Some(target) = self.enemies.iter_mut().find(|e| e.id == dmg.target_id) {
                target.take_damage(dmg.damage);
            }
        }

        let mut exploded_bullets = Vec::new();
        for idx in 0..existing_projectiles_count {
            let proj = &mut self.projectiles[idx];
            let dx = proj.target_coordinates.0 - proj.coordinates.0;
            let dy = proj.target_coordinates.1 - proj.coordinates.1;
            let dist = (dx * dx + dy * dy).sqrt();
            let step = proj.speed * dt;

            if dist <= step {
                exploded_bullets.push(idx);
                star_damage.push(StarDamage {
                    star_id: proj.target_star_id,
                    position: proj.target_coordinates,
                    damage: proj.damage,
                });
            } else {
                proj.coordinates.0 += (dx / dist) * step;
                proj.coordinates.1 += (dy / dist) * step;
            }
        }

        for idx in exploded_bullets.into_iter().rev() {
            if idx < self.projectiles.len() {
                self.projectiles.remove(idx);
            }
        }

        star_damage
    }

    pub fn enemies(&self) -> &[Enemy] {
        &self.enemies
    }

    pub fn next_id(&self) -> usize {
        self.enemies.iter().map(|e| e.id).max().unwrap_or(0) + 1
    }

    pub fn spawn(&mut self, id: usize, et: EnemyType, position: (f32, f32)) {
        let mut enemy = Enemy::new(14.0, id);
        enemy.enemy_type = et;
        enemy.hp = et.hp();
        enemy.visibility = et.visibility();
        enemy.coordinates = position;
        self.enemies.push(enemy);
    }

    pub fn push(&mut self, enemy: Enemy) {
        self.enemies.push(enemy);
    }

    pub fn damage_enemy(&mut self, id: usize, amount: f32) -> bool {
        if let Some(enemy) = self.enemies.iter_mut().find(|e| e.id == id) {
            enemy.take_damage_from_player(amount);
            return enemy.is_dead();
        }
        false
    }

    pub fn remove_dead(&mut self) -> usize {
        let before = self.enemies.len();
        self.enemies.retain(|e| !e.is_dead());
        before - self.enemies.len()
    }

    pub fn remove(&mut self, id: usize) -> bool {
        let before = self.enemies.len();
        self.enemies.retain(|e| e.id != id);
        self.enemies.len() != before
    }

    pub fn clear(&mut self) {
        self.enemies.clear();
        self.projectiles.clear();
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
