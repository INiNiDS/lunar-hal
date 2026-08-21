//! Localization decode microbench: candidate post-processing cost per
//! neighborhood (pure CPU math, no model weights involved).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::LocalizationOutput;

/// Builds a synthetic localization output: `anchors * slots` candidates.
fn synthetic_output(anchors: usize, slots: usize, seed: u64) -> Vec<LocalizationOutput> {
    let mut lcg = seed;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0
    };
    (0..anchors)
        .map(|_| LocalizationOutput {
            candidates: (0..slots)
                .map(|_| lnai_models::StarCandidate {
                    existence_prob: next(),
                    relative_position: [next() * 20.0 - 10.0; 3],
                    covariance: [next() * 0.1; 6],
                })
                .collect(),
        })
        .collect()
}

/// Full decode pipeline: filter by existence threshold, extract variances,
/// accumulate the strongest candidate per anchor.
fn decode_all(outputs: &[LocalizationOutput], threshold: f32) -> (usize, f64) {
    let mut kept = 0_usize;
    let mut best_sum_sq = 0.0_f64;
    for output in outputs {
        if let Some(best) = output
            .candidates
            .iter()
            .filter(|c| c.existence_prob >= threshold)
            .max_by(|a, b| a.existence_prob.total_cmp(&b.existence_prob))
        {
            kept += 1;
            let var = best.positional_variances();
            best_sum_sq += (var[0] + var[1] + var[2]) as f64;
        }
    }
    (kept, best_sum_sq)
}

fn bench_localization_decode(c: &mut Criterion) {
    let outputs = synthetic_output(512, 16, 42);

    c.bench_function("localization_decode_512x16", |b| {
        b.iter(|| black_box(decode_all(black_box(&outputs), 0.5)))
    });
}

criterion_group!(benches, bench_localization_decode);
criterion_main!(benches);
