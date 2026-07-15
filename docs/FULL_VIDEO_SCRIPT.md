# 🎙️ Word-for-Word Video Presentation Script

This script is designed for you to read directly while recording your screen-share presentation. It aligns with the slides and live code demonstration.

---

## ⏱️ Video Outline & Timing
* **0:00 - 0:45**: Slide 1: Introduction & Title
* **0:45 - 1:45**: Slide 2: Project Overview & Motivation
* **1:45 - 2:45**: Slide 3: Digital Signal Processing (DSP) Pipeline
* **2:45 - 3:45**: Slide 4: Model Architecture & Linear Probing
* **3:45 - 4:30**: Slide 5: Cross-Validation & Telemetry
* **4:30 - 5:45**: Slide 6: Unseen Test Results & Error Analysis
* **5:45 - 7:00**: Live Prototype Demo (Terminal run)
* **7:00 - 7:30**: Slide 7: Conclusion & Future Directions

---

## 🎬 Step-by-Step Script

### Slide 1: Title & Title Page
* **Visual on Screen**: Slide showing: "Project 1: Environmental Sound Classification via Audio Spectrogram Transformer", "Course: Building the Future of Voice & Audio", "Submitted by: Vinayak Abrol".
* **Action**: Start your recording, turn on your webcam, and ensure your microphone is active.
* **Spoken Narration**:
  > "Hello everyone, my name is Vinayak Abrol. Welcome to my final project presentation for the course 'Building the Future of Voice & Audio'. Today, I am excited to present our implementation of an Environmental Sound Classification system. Our project is built using the Audio Spectrogram Transformer, or AST. 
  > 
  > In this presentation, I will discuss the limitations of traditional convolutional networks for audio, explain our data preprocessing pipeline, walk through our parameter-efficient linear probing strategy, share our performance results on unseen test data, and demonstrate our working command-line prototype. Let's begin."

---

### Slide 2: Project Overview & Motivation
* **Visual on Screen**: Bullet points showing:
  - "Goal: Categorize non-speech & non-music sounds (animals, urban noises, nature)."
  - "The CNN Bottleneck: Local receptive fields miss long-range spectral-temporal dependencies."
  - "The AST Solution: Purely self-attention-based transformer processes spectrograms globally."
  - "Dataset: ESC-50 benchmark (2,000 balanced clips, 50 classes)."
* **Spoken Narration**:
  > "Environmental Sound Classification involves categorizing complex, non-speech, and non-music audio signals—ranging from natural soundscapes like rain and crickets to urban noises like engines and sirens. 
  > 
  > Historically, Convolutional Neural Networks, or CNNs, have been used for this task. However, CNNs have a major limitation: local convolutional kernels struggle to capture long-range, global dependencies between distant frequency bands and time intervals. 
  > 
  > To solve this, we implemented the Audio Spectrogram Transformer. AST is a purely attention-based model that treats audio spectrograms as a sequence of patches, measuring the correlation between all patches globally. We use the ESC-50 dataset as our benchmark, which contains 2,000 balanced recordings across 50 distinct sound categories."

---

### Slide 3: Digital Signal Processing (DSP) Preprocessing
* **Visual on Screen**: Flowchart:
  - "Raw WAV (44.1 kHz, Stereo) $\rightarrow$ Mono-Mix (Average) $\rightarrow$ Resample (16 kHz) $\rightarrow$ Standardize Duration (5.0 seconds) $\rightarrow$ Log-Mel Transform (128 Mel bands, 400 FFT window, 160 hop size) $\rightarrow$ AudioSet Normalization."
* **Spoken Narration**:
  > "Before passing audio to the transformer, we must ensure it matches the model's pre-trained inductive bias. We built a robust DSP pipeline to standardize our inputs. 
  > 
  > First, we load the raw WAV files and convert them from stereo to mono by averaging the channels. Next, we downsample the signal from 44.1 kHz to strictly 16 kHz. We standardize the clip duration to exactly 5.0 seconds—zero-padding shorter clips and truncating longer ones. 
  > 
  > We then compute a 128-band log-Mel spectrogram using an FFT window size of 400 samples, which corresponds to 25 milliseconds, and a hop length of 160 samples, which corresponds to a 10-millisecond stride. Finally, we normalize the spectrogram using AudioSet statistics to center our values around a mean of -4.26 and a standard deviation of 4.56."

---

### Slide 4: Model Architecture & Linear Probing
* **Visual on Screen**: Diagram of the model:
  - "Normalized Spectrogram $\rightarrow$ 16x16 Patches $\rightarrow$ Linear Projection $\rightarrow$ Prepend [CLS] Token $\rightarrow$ Transformer Encoder (Frozen) $\rightarrow$ Custom Linear Head (Trainable)."
  - "Parameter Count: Total = 86.2 Million, Trainable = 39,986, Frozen = 86.1 Million."
* **Spoken Narration**:
  > "For our model architecture, we load the pre-trained Audio Spectrogram Transformer from MIT. The model splits our normalized spectrogram into overlapping 16x16 patches, projects them into a 768-dimensional space, and prepends a classification token—referred to as the CLS token. 
  > 
  > Since the ESC-50 dataset is relatively small, training a model with 86 million parameters from scratch would lead to severe overfitting. To prevent this, we implement a memory-efficient linear probing strategy. 
  > 
  > We freeze all encoder backbone weights, ensuring they remain completely unchanged. We then attach a newly initialized linear classification head that maps the 768-dimensional CLS token output into our 50 target classes. This reduces our trainable parameters to just 39,986, enabling fast training and preventing overfitting."

---

### Slide 5: Cross-Validation & Telemetry
* **Visual on Screen**: Points:
  - "5-Fold Cross-Validation: Folds 1-3 for Training, Fold 4 for Validation, Fold 5 for Unseen Test."
  - "Optimization: AdamW Optimizer ($10^{-4}$ LR), Cosine Annealing learning rate scheduler, Cross-Entropy Loss."
  - "Telemetry: Real-time TensorBoard tracking & flat-file CSV history logging."
* **Spoken Narration**:
  > "To ensure rigorous evaluation and prevent data leakage, we set up our pipeline using cross-validation folds. We train our model on Folds 1, 2, and 3, which is 1,200 samples. We validate on Fold 4 for early stopping, and we hold out Fold 5 completely as our unseen test split. 
  > 
  > We optimize cross-entropy loss using the AdamW optimizer with a learning rate of $10^{-4}$ and weight decay of $10^{-4}$. We employ a Cosine Annealing learning rate scheduler and set early stopping patience to 3 epochs. 
  > 
  > For logging, we integrated dual-telemetry: local CSV files logging epoch-level metrics, and TensorBoard running in the background to monitor loss and accuracy curves in real-time."

---

### Slide 6: Unseen Test Results & Error Analysis
* **Visual on Screen**: 
  - Metrics table: Accuracy = 84.00%, Loss = 1.5185, F1-Score = 0.8247, Precision/Recall = 0.84.
  - Image of the 50x50 confusion matrix (`outputs/confusion_matrix.png`).
  - Top 3 confusions list.
* **Spoken Narration**:
  > "After training, we loaded our best checkpoint and ran it on the unseen Fold 5 test split. The model achieved a **Test Accuracy of 84.00%** with a **Macro F1-Score of 0.8247** and a **Test Loss of 1.5185**. 
  > 
  > To inspect class-level performance, we generated a 50x50 confusion matrix, which you can see on the screen. Evaluating a 50-class model reveals clear acoustic similarities. Our top confusions include wind being misclassified as a train, and mouse clicks being confused with keyboard typing. 
  > 
  > These are logical confusions: wind and trains share similar continuous broadband noise envelopes, while mouse clicks and typing keyboards both share rapid, high-frequency transient characteristics. These patterns help us understand the model's acoustic reasoning."

---

### Slide 7: Live Prototype Demonstration
* **Action**: Minimize your slides and share your Terminal. Show the command and run it live.
* **Spoken Narration**:
  > "Now, I will show you a live demonstration of our working prototype. We created a command-line script called `predict.py` to classify custom audio files. 
  > 
  > Let's run the script on a dog barking file from the dataset. I will type:
  > `python src/predict.py dataset/ESC-50-master/audio/1-100032-A-0.wav`
  > 
  > As you can see, the script starts by loading our audio clip, preprocessing it through our DSP pipeline, and initializing our best checkpoint model. 
  > 
  > The script prints a clean results table. It correctly classifies the sound as a DOG with a confidence of 17.21%, followed by other acoustically similar categories like crow or sneezing. This proves that our model is fully functional and ready for real-world inference."

---

### Slide 8: Conclusion & Future Directions
* **Visual on Screen**: 
  - Summary: Pure attention-based classification is highly effective.
  - Future work: Full fine-tuning, SpecAugment, Multi-fold ensembling.
  - "Thank you! GitHub Repo: [Link]"
* **Spoken Narration**:
  > "In conclusion, this project demonstrates that the purely attention-based Audio Spectrogram Transformer is highly effective for environmental sound classification. Under a frozen linear probing constraint, we achieved an 84% test accuracy on unseen data. 
  > 
  > For future work, we plan to slowly unfreeze the transformer backbone layers using discriminative learning rates, apply SpecAugment to prevent overfitting during full training, and ensemble multiple cross-validation checkpoints to push the accuracy even higher. 
  > 
  > That concludes my presentation. Thank you for your time, and I am happy to take any questions."
