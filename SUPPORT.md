# 🙋 Support & Troubleshooting

This document provides resources for resolving common issues and getting support for the ESC-50 AST project.

---

## 1. Common Troubleshooting Steps

Before asking for help, please review the troubleshooting table in [README.md](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/README.md#9-troubleshooting-guide) or check these common issues:

### 1. `ModuleNotFoundError: No module named 'tensorboard'`
This occurs if the requirements were not installed after our MLOps updates. Run:
```bash
pip install -r requirements.txt
```

### 2. `UserWarning: At least one mel filterbank has all zero values...`
This is a standard warning output by `torchaudio` when calculating Mel Spectrograms on low-frequency ranges or short window sizes. It is expected and does not impact training or correctness.

### 3. macOS SSL Download Crashes
If the downloader fails to fetch the ESC-50 ZIP file with SSL errors, ensure you run the certificate installation script on your Mac:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

---

## 2. Getting Support

If you cannot resolve an issue, please choose one of these channels:

*   **GitHub Issues**: Search the existing issues to check if your problem has already been solved. If not, open a new issue detailing:
    - Your operating system and Python version.
    - The exact command you ran.
    - The complete error stack trace.
    - Relevant log outputs from `logs/data_pipeline.log`.
*   **Academic Support**: If you are working on this project as part of IIITD classwork, check the course forum or reach out to the course instructors and TAs.
