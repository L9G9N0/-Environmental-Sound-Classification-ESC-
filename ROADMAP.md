# 🗺️ Project Roadmap

This roadmap outlines the milestones completed so far and lists the planned features for upcoming releases.

---

## 🏁 Completed Milestones

### Milestone 1: Data Engineering Foundation (v1.0.0)
*   [x] Programmatic ESC-50 dataset download and folder extraction.
*   [x] Metadata loading, validation, and balance checks.
*   [x] DSP transforms: resampling to 16kHz, mono mixing, and length standardization (5.0s).
*   [x] Log-Mel spectrogram generation and AudioSet normalization.
*   [x] Lazy-loading PyTorch `Dataset` and multi-process `DataLoaders`.
*   [x] Synthetic audio unittests for DSP validation.

### Milestone 2: AST Modeling & MLOps Pipeline (v1.1.0)
*   [x] Hugging Face AST model loading with custom classification head (50 classes).
*   [x] Parametric switches for Linear Probing (encoder backbone freezing).
*   [x] Training and validation loops with metric logging.
*   [x] Early stopping and scheduler integration (Cosine, Step, Plateau).
*   [x] Checkpoint saving/reloading to resume training.
*   [x] CSV and TensorBoard event logging.
*   [x] Unit tests for model training and convergence checks.

---

## ⏳ Upcoming Milestones

### Milestone 3: Evaluation, Metrics, and Analysis (Member 3)
*   [ ] Run predictions on the unseen test fold (Fold 5) using the best checkpoint.
*   [ ] Calculate evaluation metrics: **Accuracy, Precision, Recall, and F1-Score**.
*   [ ] Generate and plot a **50x50 Confusion Matrix** to analyze class-level misclassifications.
*   [ ] Generate classification reports (macro/micro averages).

### Milestone 4: Production Inference Service (Member 3)
*   [ ] Build a CLI inference script `src/predict.py` that takes a path to a raw `.wav` file, preprocesses it, runs model forward passes, and prints the predicted class name.
*   [ ] Develop a FastAPI server wrapper (as templated in `DEPLOYMENT.md`) to serve classification predictions over HTTP endpoints.

### Milestone 5: Interactive Web UI Demo (Future)
*   [ ] Develop an interactive Web UI demo using **Gradio** or **Streamlit**.
*   [ ] Support file uploads and microphone input to record and classify audio in real-time.
*   [ ] Visualize class probabilities using horizontal bar charts.
*   [ ] Package the entire application as a **Docker** container for simple one-command deployment.
