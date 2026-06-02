# 🛡️ Security Policy

## Supported Versions

We actively monitor and fix security vulnerabilities only in the current development cycle of Lunar-HAL. Because this project is evolving rapidly, we do not maintain long-term support (LTS) for older releases at this stage.

| Version | Supported          |
| ------- | ------------------ |
| >= 0.1.x| 🚀 Active Support  |
| < 0.1.0 | ❌ Unsupported      |

---

## 🔒 Reporting a Vulnerability

**Please do not open public GitHub issues for security vulnerabilities.** If you discover a security flaw—especially regarding local model execution, memory safety issues in our core Rust layer, or arbitrary code execution vulnerabilities within the AI-generated physics environments—please report it responsibly.

To report a vulnerability:
1. Send an email to **security@ininids.in.rs**  with the subject line `[SECURITY] Lunar-HAL Vulnerability Report`.
3. Include a detailed description of the vulnerability.
4. Provide a minimal working Proof of Concept (PoC) or step-by-step instructions to reproduce the issue.
5. Mention your system environment (OS, GPU architecture, and driver versions), as some vulnerabilities might be specific to underlying hardware acceleration layers (CUDA/Vulkan).

We appreciate your help in keeping Lunar-HAL secure for the open-source community. We will acknowledge your report within 48 hours and coordinate a fix before public disclosure.

---

## 🧠 Local AI & VRAM Safety Guidelines

Since Lunar-HAL relies entirely on loading and executing model weights locally on your GPU:
* **Untrusted Weights:** Never attempt to load unverified model checkpoints or custom configuration files from untrusted sources into the `ai_pipeline/` directory. 
* **Process Isolation:** The engine runs within user-space constraints, but always ensure your local environment is patched against GPU driver-level privilege escalation exploits.
