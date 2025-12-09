# K230-OCR-Optimization
基于 CanMV K230 的高性能嵌入式 OCR 项目。包含 CTC 解码修复、置信度过滤、视觉防抖及小目标过滤优化，显著提升识别准确率与帧率。High-performance embedded OCR on CanMV K230 (MicroPython). Features optimized CTC decoding, log-probability confidence filtering, visual stabilization, and small object suppression for real-time detection.
# K230 Optimized OCR (嵌入式字符识别优化版)

This project implements a robust OCR system on the **01Studio CanMV K230** platform using MicroPython. It features a two-stage pipeline (Detection + Recognition) with significant performance and logic optimizations.

## 🚀 Key Features (核心特性)

* **Visual Stabilization (视觉防抖)**: Implements an interval-based update strategy (updates every 3 frames) to prevent screen flickering and improve FPS.
* **CTC Decoding Fix (CTC 解码修复)**: Solves the "Dictionary Not Found" crash caused by the CTC algorithm's `Blank` index (N+1 problem).
* **Confidence Filtering (置信度拦截)**: Converts Log-Softmax outputs to real probabilities using `math.exp` and filters out characters with <50% confidence.
* **Performance Boost**: Filters out small noise (<15px) to save inference time.

## 🛠️ Hardware & Environment

* **Platform**: 01Studio CanMV K230
* **Language**: MicroPython
* **Sensor**: GC2093 (Configured to VGA 640x480 for optimal latency)
* **Display**: Supports HDMI / LCD3.5 / LCD2.4

## 🔧 Optimization Details (优化细节)

### 1. The CTC Blank Issue
Standard CTC decoding outputs a `Blank` character as the last class index. Previous implementations incorrectly flagged this as an "Index Out of Bounds" error. This project implements a whitelist logic to correctly ignore `Blank` tokens.

### 2. Log-Probability Conversion
The K230 NPU outputs probabilities in Log-Softmax format (negative values). We implemented an exponential conversion:
$$P_{real} = e^{log\_prob}$$
This allows for intuitive thresholding (e.g., `conf_threshold = 0.5`).

## 📦 How to Run

1. Copy `main.py` to the K230 file system.
2. Ensure `ocr_det_int16.kmodel`, `ocr_rec_int16.kmodel`, and `dict.txt` are in `/sdcard/examples/kmodel/`.
3. Run the script in CanMV IDE.

---
*Created for [Your Event/Course Name]*
