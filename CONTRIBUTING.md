
#  Contributing to Lunar-HAL

First off, thank you for considering contributing to Lunar-HAL!  It is space explorers and developers like you who make building an infinite local universe possible.
Since Lunar-HAL is distributed under the **AGPL v3 License**, any contribution you make will also be open-source under the same copyleft terms. Let's keep the cosmos free and open.

---

## 🗺️ Code of Conduct

By participating in this project, you agree to maintain a respectful, welcoming, and collaborative environment. Treat fellow developers with respect—we are all trying to survive out here in the void.

---

## 🚀 How Can I Contribute?

### 1. Reporting Bugs 
If you find a bug (such as a compiler panic or a memory leak in our local AI pipeline), please open an Issue with the following details:
* **Clear Title:** Use a concise and descriptive title.
* **Steps to Reproduce:** Provide the exact steps needed to reproduce the problem.
* **System Specs:** Include your CPU, GPU, and RAM details, as our neural generation relies heavily on local hardware.
* **Logs:** Paste the terminal stack trace, panic logs, or relevant console output inside code blocks.

### 2. Suggesting Features 
Have an idea for the procedural planetary ecosystem or want to enhance the Dioxus UI? We welcome your suggestions:
* Open an Issue and select the feature request template.
* Explain *why* this feature would be useful and how it aligns with the project's "local hardware / zero-latency" manifesto.

### 3. Submitting Pull Requests (PRs) 
Ready to write some Rust? Here is the workflow to get your code merged:

1. **Fork the Repository:** Create your own fork of the project.
2. **Create a Branch:** Create your feature branch from `main`:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```
3. **Write idiomatic Rust:** Follow standard Rust design principles, ensure proper ownership/borrowing, and keep your code documented.
4. **Format & Lint:** We enforce clean code standards. Always run these checks locally before committing:
   ```bash
   cargo fmt --all --check
   cargo clippy --all-targets -- -D warnings
   ```
5. **Test Your Changes:** Run the test suite to ensure nothing is broken:
   ```bash
   cargo test --all-targets
   ```
   *Note: If your changes touch physics or procedural generation modules, make sure your local hardware can handle execution without run-time panic.*
6. **Open the PR:** Describe your changes clearly, link any relevant issues, and wait for the maintainers to review.

---

## 🛠️ Development Setup

Getting your local environment ready is straightforward. Ensure you have the standard Rust toolchain installed, and then set up the Dioxus CLI:

### 1. Install Dioxus CLI
```bash
cargo install dioxus-cli
```

### 2. Clone and Setup the Project
```bash
git clone https://github.com/your-username/lunar-hal.git
cd lunar-hal
```

### 3. Verify the Project Structure
You can check if the project compiles and is ready for development:
```bash
dx check
```

### 4. Run Locally
To run the application locally in development mode:
```bash
dx serve
```

> **Graphics Drivers:** Make sure your graphics drivers (CUDA/Vulkan) are updated to the latest version to prevent the local AI modules from failing during runtime.

---

Ready to build the future? Grab your space suit, fire up your GPU, and let's write some Rust! 🚀
