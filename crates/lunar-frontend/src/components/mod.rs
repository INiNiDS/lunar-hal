// Top-level component groups
pub mod layout;
pub mod pages;
pub mod ui;
pub mod editor;

// Re-exports (convenience)
pub use layout::header::Header;
pub use layout::hero_section::HeroSection;
pub use pages::about::AboutPage;
pub use pages::contact::ContactPage;
pub use ui::divider::Divider;
pub use ui::glowing_button::GlowingButton;
pub use ui::glowing_subtitle::GlowingSubtitle;
pub use ui::glowing_title::GlowingTitle;
pub use editor::sidebar::EntropySlider;
