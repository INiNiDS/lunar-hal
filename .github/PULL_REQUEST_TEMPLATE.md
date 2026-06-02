## Lunar-HAL Pull Request Template

### Description
Provide a clear description of what changes this PR introduces. Why is this modification necessary? How does it improve the local universe simulation?

### Type of Change
Please delete options that are not relevant:
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Code refactoring / Clean Code optimization
- [ ] Documentation update (README, DevLogs, etc.)

---

## Quality Control Checklist

Before submitting this PR, please ensure you have completed the following steps. **Unchecked items will result in automated or manual review delays.**

### Rust Alignment
- [ ] My code follows the clean, safe, and idiomatic patterns of the Rust programming language.
- [ ] I have run `cargo fmt --all` and my code is perfectly formatted.
- [ ] I have run `cargo clippy --all-targets -- -D warnings` and there are **zero** lints or warnings remaining.
- [ ] I have verified that all data structures comply with strict memory safety rules (no hidden leaks or unsafe blocks without justification).

### Frontend & UX (Dioxus / Tailwind)
- [ ] I have verified the components render correctly without breaking the layout or overlapping elements.
- [ ] Class naming conventions strictly align with the existing glassmorphic/dark setup.

### Performance & Hardware Impact
- [ ] The code introduces zero unnecessary allocations where performance matters (Zero-Cost Abstractions).
- [ ] Tested execution locally to ensure it doesn’t cause sudden VRAM leaks or thread panics.

---

### Related Issues
Fixes # (reference the issue here, e.g., `Fixes #42`)

### Screenshots / Clips
*If applicable, add screenshots or clips of the UI changes or terminal logs showing the performance optimization.*
