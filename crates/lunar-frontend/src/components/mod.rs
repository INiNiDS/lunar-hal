// Top-level component groups
pub mod pages;
pub mod layout;
pub mod ui;

// Re-exports (convenience)
pub use pages::about::AboutPage;
pub use pages::contact::ContactPage;
pub use layout::header::Header;
pub use layout::hero_section::HeroSection;
pub use ui::divider::Divider;
pub use ui::glowing_button::GlowingButton;
pub use ui::glowing_subtitle::GlowingSubtitle;
pub use ui::glowing_title::GlowingTitle;
