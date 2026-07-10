# 📦 Release Guidelines & Release Process

This document outlines the release process and versioning conventions for the ESC-50 AST repository.

---

## 1. Versioning Conventions

This project strictly adheres to **Semantic Versioning (SemVer)**:
- **MAJOR version (X.0.0)**: Introduced when making incompatible API or architectural changes (e.g., swapping the transformer model for a different architecture that breaks preprocessing, or refactoring CLI arguments).
- **MINOR version (0.Y.0)**: Introduced when adding backward-compatible features (e.g., adding model configurations, implementing the trainer, or adding new test suites).
- **PATCH version (0.0.Z)**: Introduced when fixing backward-compatible bugs (e.g., resolving the checkpoint validation loss sync ordering bug).

---

## 2. Release Steps

When publishing a new release milestone:

### Step 1: Branch Isolation
Ensure all features for the release are merged into the `main` branch, and the automated test suite passes:
```bash
python -m unittest discover -s tests
```

### Step 2: Document the Changes
1. Update [CHANGELOG.md](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/CHANGELOG.md) by adding the new version number, release date, and listing added, changed, or fixed features.
2. Update the version badges in the main [README.md](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/README.md) if applicable.

### Step 3: Git Tagging
Tag the release commit using Semantic Versioning prefixes:
```bash
# Tag the current commit
git tag -a v1.1.0 -m "Release v1.1.0: AST model training, trainer, and tests integration"

# Push the tag to GitHub
git push origin v1.1.0
```

---

## 3. Deployment Artifact Packaging

When releasing a deployment container or inference package:
- Package the weights of the best performing model (`outputs/checkpoints/best_model.pt`).
- Do **not** package temporary training checkpoints (`latest_model.pt`), local log files, or raw WAV audio files (`dataset/`). These are automatically excluded using `.gitignore`.
