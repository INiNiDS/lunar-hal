mod lore_gen;

#[tokio::main]
async fn main() {
    let output = std::env::args()
        .nth(1)
       .unwrap_or_else(|| "models/stellar_lore_cache.json".to_string());

    if let Err(e) = lore_gen::generate_all(&output).await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
