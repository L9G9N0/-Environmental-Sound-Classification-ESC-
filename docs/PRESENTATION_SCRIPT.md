# 🎙️ Video Presentation Script: ESC-50 Sound Classification via AST

This document provides a slide-by-slide structure, screen actions, and spoken narration scripts for preparing your **5-to-10 minute presentation video** (`.mp4`) as required for the final project submission.

---

## 🎬 Production Guide

* **Duration**: 5 - 8 minutes
* **Format**: Recorded screen share (e.g., using Zoom, OBS Studio, or QuickTime) showing your slides and terminal, with your voice (and optional webcam).
* **Tips**:
  - Keep your tone professional and structured.
  - Speak clearly and highlight engineering decisions rather than just reading code.
  - Make sure your terminal font is large enough to read easily.

---

## 📂 Presentation Slide Deck & Narration Outline

### Slide 1: Title & Overview (0:00 - 0:45)
* **Screen Visuals**: 
  - Slide showing: Title, course name, your name, and a high-level flowchart of Project 1.
* **Screen Actions**:
  - Start recording with your webcam visible in the corner.
* **Spoken Script**:
  > "Hello everyone, my name is Vinayak Abrol. Today I will present my course project on Environmental Sound Classification using the Audio Spectrogram Transformer, or AST. In this project, we move away from traditional convolutional models to leverage a purely attention-based architecture for categorizing non-speech and non-music audio signals. I will walk you through our DSP pipeline, model architecture, cross-validation setup, and final test results on unseen test data."

---

### Slide 2: Motivation & Problem Statement (0:45 - 1:30)
* **Screen Visuals**:
  - Key points explaining the limitations of CNNs (local receptive fields) vs. self-attention (global temporal-spectral mapping).
  - Details of the ESC-50 dataset: 2,000 clips, 50 classes, perfectly balanced.
* **Spoken Script**:
  > "Environmental sound classification is challenging because sounds like dog barking, rain, or engines lack rigid sequential structures and have highly diverse acoustic properties. While CNNs are the traditional choice, they process spectrograms locally. 
  > 
  > The Audio Spectrogram Transformer solves this by dividing 2D spectrograms into patches and modeling them as a sequence. This allows the model's self-attention layers to capture global temporal and spectral dependencies from day one, matching timbre and rhythm across the entire clip."

---

### Slide 3: Digital Signal Processing (DSP) Preprocessing (1:30 - 2:30)
* **Screen Visuals**:
  - Diagram showing the pipeline: 44.1 kHz stereo WAV $\rightarrow$ mono $\rightarrow$ 16 kHz $\rightarrow$ 5.0s crop/pad $\rightarrow$ Log-Mel Spectrogram $\rightarrow$ Normalization.
  - Code snippet of `src/preprocessing.py`.
* **Spoken Script**:
  > "Before feeding audio into the AST, it must match the model's pre-trained inductive bias. We downsample our files to 16 kHz and convert them to mono. We standardize the duration to exactly 5.0 seconds. 
  > 
  > We then compute a 128-band log-Mel spectrogram using a 25-millisecond window and a 10-millisecond stride. Lastly, we normalize the spectrogram using AudioSet population statistics to center our values around a mean of -4.26 and standard deviation of 4.56. This ensures numerical stability."

---

### Slide 4: Model Adaptation & Linear Probing (2:30 - 3:30)
* **Screen Visuals**:
  - Illustration of AST model showing frozen transformer backbone (86M params) and a trainable classifier head (39k params).
  - Highlight of the parameter audit numbers: Trainable parameters = 39,986, Total parameters = 86.2M.
* **Spoken Script**:
  > "For our model backbone, we load the pre-trained AST model from MIT and Hugging Face. Because our dataset has only 2,000 clips, fine-tuning 86 million parameters from scratch would lead to immediate overfitting. 
  > 
  > Instead, we implement a memory-efficient linear probing strategy. We freeze the entire transformer encoder backbone, meaning gradients are not calculated for these weights. We attach a newly initialized linear classifier head that projects the 768-dimensional output of the CLS token into our 50 target classes. This reduces our trainable parameter count to just under 40,000, enabling fast, robust training."

---

### Slide 5: Cross-Validation & Telemetry (3:30 - 4:15)
* **Screen Visuals**:
  - Folds layout: Folds 1-3 (Train), Fold 4 (Validation), Fold 5 (Unseen Test).
  - Brief snippet of training curves from TensorBoard or logs.
* **Spoken Script**:
  > "To prevent data leakage, we structure the dataset using the official ESC-50 splits. We train on Folds 1 to 3, validate on Fold 4 for early stopping, and hold out Fold 5 entirely as our unseen test split. 
  > 
  > Our trainer optimizes cross-entropy loss using the AdamW optimizer and a Cosine Annealing learning rate scheduler. We also integrated double telemetry: logging metrics to local CSV history files, and real-time visualization using TensorBoard, which monitors train and val performance."

---

### Slide 6: Performance on Unseen Test Data & Confusion Analysis (4:15 - 5:15)
* **Screen Visuals**:
  - Table of metrics: Accuracy (78.25%), Macro F1 (0.7617), Precision/Recall.
  - The 50x50 confusion matrix plot (`outputs/confusion_matrix.png`).
* **Spoken Script**:
  > "Evaluating the final best checkpoint on our held-out Fold 5 test split yielded a strong test accuracy of 78.25% and a macro F1-score of 0.7617. 
  > 
  > Looking at our 50x50 confusion matrix, we can pinpoint specific acoustic confusions. For example, mechanical hums like washing machines and helicopters are occasionally confused with general engines due to their low-frequency spectra. Similarly, mouse clicks are confused with keyboard typing due to transient impulse shapes. These insights guide our future work on specialized audio augmentations."

---

### Slide 7: Live Prototype Demonstration (5:15 - 6:30)
* **Screen Actions**:
  - Minimize slides and open your Terminal.
  - Run the evaluation script or print output from the CLI predict script.
  - Run this command live:
    ```bash
    python src/predict.py dataset/ESC-50-master/audio/1-100032-A-0.wav
    ```
  - Show the output table printing the class 'DOG' with the top 5 probabilities.
* **Spoken Script**:
  > "Let me show you a working prototype demo. We built a production-ready CLI inference script called predict.py. We pass it any arbitrary WAV file. 
  > 
  > Here, I run it on a sample recording from our dataset. As you can see, the script automatically processes the audio, loads our best model weights, and classifies it. It correctly identifies the class as DOG with a confidence of 20.93 percent, followed by other similar acoustic classes. This CLI tool makes our model ready for local deployment."

---

### Slide 8: Conclusion & Q&A (6:30 - 7:00)
* **Screen Visuals**:
  - Summary slide showing next steps (SpecAugment, unfreezing encoder layers, ensembling).
  - Repository link and thank you message.
* **Spoken Script**:
  > "In conclusion, we successfully built and verified a purely attention-based ESC pipeline. Our test accuracy of 78.25 percent demonstrates the capability of transformer backbones on audio spectrograms even under heavy parameter constraints. For future iterations, we plan to unfreeze the final transformer blocks and introduce SpecAugment to push accuracy beyond 85 percent. Thank you, and I am happy to answer any questions."

---

## 🎥 Recording Checklist
1. [ ] **Clean Terminal**: Close all irrelevant tabs and logs.
2. [ ] **Check Paths**: Ensure the `predict.py` script has access to `outputs/checkpoints/best_model.pt`.
3. [ ] **Video Format**: Set your recorder to output `.mp4` format.
4. [ ] **Audio Check**: Ensure your microphone is clear and background noise is minimized.
