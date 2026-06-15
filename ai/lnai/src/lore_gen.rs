use anyhow::Result;
use serde::{Deserialize, Serialize};

const LLM_URL: &str = "http://localhost:8000/v1/chat/completions";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

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

const SYSTEM_PROMPT: &str = r#"You are a sci-fi lore generator for a space exploration game called LunarSim. You create rich, atmospheric descriptions of stars and stellar anomalies. Your writing style is evocative, scientific yet poetic — think of the lore in games like Elite Dangerous, No Man's Sky, and Mass Effect combined with hard sci-fi sensibility.

Rules:
- Each entry must be UNIQUE and MEMORABLE
- Use real astrophysics concepts creatively (magnetohydrodynamics, Hawking radiation, CNO cycle, etc.)
- Vary tone: some entries mysterious, some ominous, some wondrous, some clinical
- Include specific details: distances, frequencies, element names, historical dates
- Never repeat phrases or structures across entries
- Keep descriptions between 2-4 sentences
- Keep system_lore between 3-6 sentences with concrete worldbuilding details
- visual_profile should be 1-2 vivid sentences about what the star LOOKS like
- designated_name should sound like a catalog designation with character (e.g. "UVS-7701 Thresher", "AX-Δ4 Morozko", "KX-1192 Pale Virtue")"#;

fn build_user_prompt(batch_idx: u32, star_type: &str, spectral_class: &str) -> String {
    format!(
        r#"Generate exactly 5 unique star lore entries for the following category. Each entry must be completely different from others.

Category: {star_type} ({spectral_class} class stars)
Batch: {batch_idx}

For each entry, output a JSON object with these fields:
- "designated_name": A unique catalog-style name with character
- "visual_profile": What the star looks like (1-2 vivid sentences)
- "description": A short atmospheric description (2-4 sentences)
- "system_lore": Extended lore about the star system (3-6 sentences with concrete worldbuilding)

Output ONLY a JSON array of 5 objects, nothing else. No markdown, no explanation."#,
    )
}

pub async fn generate_batch(
    client: &reqwest::Client,
    batch_idx: u32,
    star_type: &str,
    spectral_class: &str,
) -> Result<Vec<LoreEntry>> {
    let request = ChatRequest {
        model: "default".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: build_user_prompt(batch_idx, star_type, spectral_class),
            },
        ],
        temperature: 1.2,
        max_tokens: 2000,
    };

    let resp = client
        .post(LLM_URL)
        .json(&request)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("LLM request failed: {status} - {body}");
    }

    let chat_resp: ChatResponse = resp.json().await?;
    let content = chat_resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let raw_entries: Vec<serde_json::Value> = match serde_json::from_str(cleaned) {
        Ok(v) => v,
        Err(_) => {
            let start = cleaned.find('[').unwrap_or(0);
            let end = cleaned.rfind(']').unwrap_or(cleaned.len());
            serde_json::from_str(&cleaned[start..=end]).unwrap_or_default()
        }
    };

    let mut entries = Vec::new();
    for (i, raw) in raw_entries.iter().enumerate() {
        let entry = LoreEntry {
            id: batch_idx * 5 + i as u32,
            star_type: star_type.to_string(),
            spectral_class: spectral_class.to_string(),
            designated_name: raw["designated_name"].as_str().unwrap_or("Unknown").to_string(),
            visual_profile: raw["visual_profile"].as_str().unwrap_or("").to_string(),
            description: raw["description"].as_str().unwrap_or("").to_string(),
            system_lore: raw["system_lore"].as_str().unwrap_or("").to_string(),
        };
        entries.push(entry);
    }

    Ok(entries)
}

pub async fn generate_all(output_path: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let mut all_entries: Vec<LoreEntry> = Vec::new();
    let mut id_counter = 0u32;

    let categories = [
        ("Black Hole", "X"),
        ("Pulsar", "P"),
        ("Neutron Star", "N"),
        ("Magnetar", "Q"),
        ("Wolf-Rayet", "W"),
        ("O-type Supergiant", "O"),
        ("B-type Supergiant", "B"),
        ("A-type Giant", "A"),
        ("F-type Subgiant", "F"),
        ("G-type Main Sequence", "G"),
        ("K-type Dwarf", "K"),
        ("M-type Red Dwarf", "M"),
        ("White Dwarf", "D"),
        ("Binary System", "BIN"),
        ("Dyson Anomaly", "DA"),
        ("T Tauri Protostar", "T"),
        ("Carbon Star", "C"),
        ("Dwarf Nova", "DN"),
        ("L-type Brown Dwarf", "L"),
        ("Y-type Brown Dwarf", "Y"),
    ];

    let batches_per_category = 6;

    for (star_type, spectral_class) in &categories {
        println!("Generating lore for {star_type}...");
        for batch in 0..batches_per_category {
            match generate_batch(&client, batch, star_type, spectral_class).await {
                Ok(mut entries) => {
                    for entry in &mut entries {
                        entry.id = id_counter;
                        id_counter += 1;
                    }
                    println!("  Batch {}: got {} entries", batch, entries.len());
                    all_entries.extend(entries);
                }
                Err(e) => {
                    eprintln!("  Batch {} FAILED: {e}", batch);
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    let json = serde_json::to_string_pretty(&all_entries)?;
    std::fs::write(output_path, json)?;
    println!("Wrote {} lore entries to {output_path}", all_entries.len());

    Ok(())
}
