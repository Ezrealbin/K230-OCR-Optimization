# K230-OCR-Optimization
基于 CanMV K230 的高性能嵌入式 OCR 项目。包含 CTC 解码修复、置信度过滤、视觉防抖及小目标过滤优化，显著提升识别准确率与帧率。High-performance embedded OCR on CanMV K230 (MicroPython). Features optimized CTC decoding, log-probability confidence filtering, visual stabilization, and small object suppression for real-time detection.
# K230 Optimized OCR (嵌入式字符识别优化版)

This project implements a robust OCR system on the **01Studio CanMV K230   studio CanMV K230** platform using MicroPython. It features a two-stage pipeline (Detection + Recognition) with significant performance and logic optimizations.本项目使用MicroPython在**01Studio CanMV K230   studio CanMV K230**平台上实现了一个健壮的OCR系统。它具有两阶段的管道（检测识别），具有显著的性能和逻辑优化。

## 🚀 Key Features (核心特性)

* **Visual Stabilization (视觉防抖)**: Implements an interval-based update strategy (updates every 3 frames) to prevent screen flickering and improve FPS.** *视觉稳定（每周一次）**：实现基于间隔的更新策略（每3帧更新一次），以防止屏幕闪烁并提高FPS。
* **CTC Decoding Fix (CTC 解码修复)**: Solves the "Dictionary Not Found" crash caused by the CTC algorithm's `Blank` index (N+1 problem).** *CTC解码修复(CTC)**：解决了CTC算法的“空白”索引（N 1问题）导致的“字典未找到”崩溃。
* **Confidence Filtering (置信度拦截)**: Converts Log-Softmax outputs to real probabilities using `math.exp` and filters out characters with <50% confidence.** *Confidence Filtering()**：使用“math”将Log-Softmax输出转换为真实概率。Exp’并过滤掉置信度<50%的字符。
* **Performance Boost**: Filters out small noise (<15px) to save inference time.** *性能提升**：滤除小噪音（<15px），以节省推理时间。

## 🛠️ Hardware & Environment##推荐️硬件和环境

* **Platform**: 01Studio CanMV K230** *平台：01Studio CanMV K230
* **Language**: MicroPython** *语言：MicroPython
* **Sensor**: GC2093 (Configured to VGA 640x480 for optimal latency)** *传感器**:GC2093（配置为VGA 640x480以获得最佳延迟）
* **Display**: Supports HDMI / LCD3.5 / LCD2.4** *显示：支持HDMI / LCD3.5 / LCD2.4

## 🔧 Optimization Details (优化细节)

### 1. The CTC Blank Issue# # # 1。CTC空白问题
Standard CTC decoding outputs a `Blank` character as the last class index. Previous implementations incorrectly flagged this as an "Index Out of Bounds" error. This project implements a whitelist logic to correctly ignore `Blank` tokens.标准CTC解码输出一个‘ Blank ’字符作为最后一个类索引。以前的实现错误地将此标记为“索引越界”错误。这个项目实现了一个白名单逻辑来正确地忽略‘ Blank ’令牌。

### 2. Log-Probability Conversion# # # 2。对数概率转换
The K230 NPU outputs probabilities in Log-Softmax format (negative values). We implemented an exponential conversion:K230 NPU以Log-Softmax格式输出概率（负值）。我们实现了一个指数转换：
$$P_{real} = e^{log\_prob}$$
This allows for intuitive thresholding (e.g., `conf_threshold = 0.5`).这允许直观的阈值设置（例如，‘ conf_threshold = 0.5 ’）。

## 📦 How to Run

1. Copy `main.py` to the K230 file system.
2. Ensure `ocr_det_int16.kmodel`, `ocr_rec_int16.kmodel`, and `dict.txt` are in `/sdcard/examples/kmodel/`.
3. Run the script in CanMV IDE.

---
*Created for [PKU_ESP2025]*
