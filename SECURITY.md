# 🛡️ Security Policy

This document outlines security practices, vulnerability reporting procedures, and security warnings related to deserializing model checkpoints and validating audio inputs.

---

## 1. Secure Checkpoint Deserialization Warning

This project uses `torch.load` to serialize and deserialize model checkpoints (`best_model.pt` and `latest_model.pt`).

> [!CAUTION]
> **PyTorch checkpoints use Python's `pickle` module under the hood.** 
> Deserializing pickle files can execute arbitrary code. 
> 
> *   **Do not load untrusted checkpoints** downloaded from third-party websites or untrusted users.
> *   Only load checkpoints that you have trained yourself or downloaded from verified, secure sources.
> *   If you need to distribute the model, consider exporting it to the **ONNX** (Open Neural Network Exchange) format or utilizing **Safetensors** to avoid code injection vulnerabilities.

---

## 2. Input File Validation & Ingestion Security

To prevent denial of service (DoS) or buffer overflow vulnerabilities during audio ingestion (e.g., in a production web server):
1. **File Type Verification**: Ensure that uploads strictly match the `audio/wav` MIME type.
2. **File Size Limits**: Enforce maximum file size limits (e.g., 5MB for a 5-second audio clip).
3. **Format Checks**: Use `soundfile` or `torchaudio` in try-except blocks to catch corrupt audio headers and reject invalid files early.
4. **WAV Header Sanitation**: Ensure that sampling rates, bit depths, and channel configurations are verified before allocating memory tensors.

---

## 3. Local Execution Security & Networking

* **Offline Capabilities**: Except for downloading the ESC-50 ZIP archive from GitHub during the first run and downloading pre-trained weights from the Hugging Face Hub, the training pipeline runs fully offline.
* **No Telemetry**: No tracking data, usage logs, or custom datasets are sent to any remote servers.

---

## 4. Reporting a Vulnerability

If you discover a security issue or vulnerability within this repository, do **not** open a public GitHub issue. Instead, please report it directly by contacting the project maintainers or email security contacts.
