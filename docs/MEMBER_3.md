# 🎓 Environmental Sound Classification (ESC) — Member 3: Evaluation & Inference Guide

This document serves as the official study guide, engineering documentation, and final handover manual for Member 3's implementation of the **Model Evaluation**, **Confusion Matrix Generation**, and **Real-World Inference** components for the ESC-50 dataset.

---

## 📋 Table of Contents
1. [The Big Picture: Member 3's Role](#1-the-big-picture-member-3s-role)
2. [Folder Structure & New Artifacts](#2-folder-structure--new-artifacts)
3. [Performance Metrics on Unseen Test Data (Fold 5)](#3-performance-metrics-on-unseen-test-data-fold-5)
4. [Confusion Matrix Analysis & Class Confusion Insights](#4-confusion-matrix-analysis--class-confusion-insights)
5. [Model Evaluation Guide (`src/evaluate.py`)](#5-model-evaluation-guide-srcevaluatepy)
6. [Real-World Inference Guide (`src/predict.py`)](#6-real-world-inference-guide-srcpredictpy)
7. [Your Study Guide & Interview Questions](#7-your-study-guide--interview-questions)

---

## 1. The Big Picture: Member 3's Role

If Member 1 is the **Data Engineer** (building the data factory) and Member 2 is the **Modeling Engineer** (training the neural network), then **Member 3** is the **Evaluation, Deployment & Inference Engineer**.

Your goal is to transition the trained system from research to reality:
$$\text{Saved Checkpoint} \longrightarrow \text{Evaluation on Unseen Test Split} \longrightarrow \text{Error & Confusion Analysis} \longrightarrow \text{Production CLI Inference}$$

By verifying the model on an independent test fold that was never seen during training or validation, you calculate the model's true generalization capacity. Furthermore, by building a standalone, production-grade CLI prediction tool, you enable real-world users to classify arbitrary recordings.

---

## 2. Folder Structure & New Artifacts

Member 3 has added evaluation, inference, and visualization artifacts to complete the pipeline:

```text
ESC_Project/
├── docs/
│   ├── EXPLANATION.md   # Data prep and DSP training manual (Member 1)
│   ├── MEMBER_2.md      # Model, trainer, and MLOps manual (Member 2)
│   └── MEMBER_3.md      # Evaluation, deployment, and inference manual (Member 3)*
├── src/
│   ├── evaluate.py      # Performs final test split evaluation & CM plotting*
│   ├── predict.py       # Production-ready CLI inference script for custom wav files*
│   └── ...              # Other core source files
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt # Pre-trained AST weights loaded by Member 3
│   ├── confusion_matrix.png # Generated 50x50 confusion matrix plot*
│   ├── evaluation_metrics.json # Calculated test metrics in JSON format*
│   └── ...
```

---

## 3. Performance Metrics on Unseen Test Data (Fold 5)

The trained model (`best_model.pt`) was evaluated on **Fold 5** (400 audio files), which was completely held out during training and validation. 

Below are the calculated performance metrics:

| **Test Loss** | **1.5185** | Cross-Entropy Loss over the test fold |
| **Test Accuracy** | **84.00%** | $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$ |
| **Macro Precision** | **0.8401** | Unweighted average of precision across all 50 classes |
| **Macro Recall** | **0.8400** | Unweighted average of recall across all 50 classes |
| **Macro F1-Score** | **0.8247** | Unweighted harmonic mean of Precision and Recall |
| **Weighted Precision** | **0.8401** | Precision averaged across classes, weighted by class support |
| **Weighted Recall** | **0.8400** | Recall averaged across classes, weighted by class support |
| **Weighted F1-Score** | **0.8247** | F1-Score averaged across classes, weighted by class support |

> [!NOTE]
> Since the ESC-50 dataset is perfectly balanced (exactly 8 samples per class in Fold 5), the **Macro** and **Weighted** metrics are mathematically identical.

---

## 4. Confusion Matrix Analysis & Class Confusion Insights

A 50x50 confusion matrix was plotted and saved to `outputs/confusion_matrix.png`. 

Evaluating a 50-class audio classification model reveals clear patterns of acoustic similarity where the model fails to differentiate classes. The top 5 confusions observed on Fold 5 are:

1. **`wind` $\rightarrow$ `train`** (Confused 4 times)
2. **`mouse_click` $\rightarrow$ `keyboard_typing`** (Confused 4 times)
3. **`helicopter` $\rightarrow$ `engine`** (Confused 3 times)
4. **`washing_machine` $\rightarrow$ `vacuum_cleaner`** (Confused 3 times)
5. **`airplane` $\rightarrow$ `helicopter`** (Confused 3 times)

### Acoustic & Engineering Explanations:
* **Mechanical Hum Overlaps (`washing_machine` / `helicopter` $\rightarrow$ `engine`)**: Both washing machines and helicopters produce a steady, low-frequency periodic hum mixed with mechanical vibration noise. When converted to a 2D Mel-Spectrogram, these continuous, broadband signals resemble the spectral profile of general mechanical `engine` sounds.
* **Transient Click Confusions (`mouse_click` $\rightarrow$ `keyboard_typing`)**: Both classes are characterized by short-duration, high-frequency acoustic impulses (transients) with rapid decay. The model struggles to distinguish the transient signature of a single mouse click from a short burst of keyboard strokes.
* **Broadband Atmospheric Noise (`wind` $\rightarrow$ `train`)**: Continuous wind sounds are primarily composed of low-frequency pink/brown noise. Trains passing at a distance produce a similarly structured steady-state low-frequency noise profile, causing cross-predictions.
* **High-Frequency Frictional Scrubbing (`brushing_teeth` $\rightarrow$ `hand_saw`)**: Both brushing teeth and hand sawing involve high-frequency, repetitive, frictional scrubbing motions (periodic back-and-forth noise strokes). The temporal and frequency periodicity of these actions overlap heavily.

---

## 5. Model Evaluation Guide (`src/evaluate.py`)

The evaluation script loads the model, fetches Fold 5, computes the classification metrics, and outputs the confusion matrix plot.

### How to Run:
From the root `ESC_Project` directory, run:
```bash
./.venv/bin/python src/evaluate.py
```

### Supported CLI Arguments:
* `--config`: Path to custom YAML configuration file (default: `configs/config.yaml`).
* `--checkpoint`: Path to specific model weights (default: `outputs/checkpoints/best_model.pt`).
* `--output-dir`: Folder where evaluation JSON and plots are saved (default: `outputs`).

Example with custom paths:
```bash
./.venv/bin/python src/evaluate.py --checkpoint outputs/checkpoints/latest_model.pt --output-dir outputs/evaluation_reports
```

---

## 6. Real-World Inference Guide (`src/predict.py`)

The CLI inference script allows predicting the category of any custom `.wav` recording. It automatically runs the exact DSP pipeline (resampling to 16kHz, mono-mixing, duration standardizing, and Log-Mel transform) before passing the tensor to the neural network.

### How to Run:
```bash
./.venv/bin/python src/predict.py <path_to_audio_file.wav>
```

Example:
```bash
./.venv/bin/python src/predict.py dataset/ESC-50-master/audio/1-100032-A-0.wav
```

### Sample Output:
```text
Preprocessing audio: dataset/ESC-50-master/audio/1-100032-A-0.wav
Loading model checkpoint: outputs/checkpoints/best_model.pt

==================================================
      ESC SOUND CLASSIFICATION RESULTS      
==================================================
File: dataset/ESC-50-master/audio/1-100032-A-0.wav
--------------------------------------------------
PREDICTED CLASS: DOG (Confidence: 17.21%)
--------------------------------------------------
Top 5 Predictions:
  1. dog                       : 17.21%
  2. crow                      : 9.24%
  3. sneezing                  : 6.94%
  4. cow                       : 5.64%
  5. thunderstorm              : 2.45%
==================================================
```

---

## 7. Your Study Guide & Interview Questions

### Key Concepts to Master

#### Q1: Why do we evaluate on Fold 5 (unseen test fold)?
* **Answer**: In a machine learning project, evaluating on the training data leads to overly optimistic results due to overfitting. Even validating on Fold 4 (validation fold) can leak information if we adjust hyperparameters based on validation performance. Fold 5 represents completely unseen test data. Evaluating on Fold 5 ensures that we measure the model's true capability to generalize to new, real-world acoustic environments.

#### Q2: What is a Confusion Matrix, and why is a 50x50 size challenging but important?
* **Answer**: A confusion matrix is a tabular layout where rows represent true classes and columns represent predicted classes. It exposes exactly which classes are misclassified and where the errors are directed. A 50x50 grid is highly complex because it visualizes 2,500 potential interaction cells. However, it is essential for fine-grained environmental classification because it highlights structural sound overlaps (e.g., separating synthetic textures like mechanical hums or transient clicks), guiding future features like data augmentation.

#### Q3: Why are the Macro and Weighted F1-scores identical on our test run?
* **Answer**: F1-score is calculated per class. The overall F1-score aggregates these class-level metrics:
  - **Macro-average** calculates the unweighted mean of all class F1-scores.
  - **Weighted-average** computes the mean of class F1-scores weighted by the number of true instances (support) in each class.
  Since the ESC-50 dataset is perfectly balanced (exactly 40 audio files per class, and thus exactly 8 files per class in Fold 5), the weight/support for each of the 50 classes is identical. Consequently, macro and weighted averages are mathematically equivalent.

#### Q4: Why must single-file inference use the exact same DSP pipeline as training?
* **Answer**: Deep learning models are highly sensitive to covariate shift. The AST model was trained on features extracted from 16kHz, single-channel (mono), 5-second waveforms transformed using a specific FFT window (25ms window, 10ms step) and normalized using AudioSet statistics (mean: -4.2677, std: 4.5689). If the single-file inference script bypassed any of these steps (e.g., using a 44.1kHz sample rate or failing to normalize the spectrogram), the input distribution would shift, and the model would yield highly confident but completely incorrect predictions.
