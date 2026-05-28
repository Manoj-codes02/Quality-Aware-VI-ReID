# Quality-Aware Infrared Enhancement and Cross-Modal Person Re-Identification Using Zero-DCE and SIGAN

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-CUDA-orange.svg)

---

# 📌 Overview

This project implements a **Quality-Aware Visible-Infrared Person Re-Identification (VI-ReID)** framework based on adaptive infrared image enhancement and cross-modal feature learning.

The framework improves low-quality infrared (IR) images using:

* **Zero-DCE** for low-light enhancement
* **SIGAN** for semantic-aware refinement
* **BRISQUE, PSNR, and SSIM** for adaptive image quality assessment

Enhanced images are processed using a **ResNet-50** backbone for feature extraction and matched against visible-spectrum gallery images using cosine similarity and ECN re-ranking.

The system is evaluated on the **SYSU-MM01** dataset for cross-modal person re-identification.

---

# ✨ Features

* Cross-Modal Visible-Infrared Person Re-Identification
* Quality-Aware Enhancement Pipeline
* Zero-DCE Low-Light Enhancement
* SIGAN Semantic Refinement
* BRISQUE / PSNR / SSIM Quality Assessment
* ResNet-50 Feature Extraction
* ECN Re-Ranking
* Triplet Loss + CrossEntropy Loss
* CUDA Mixed Precision Training
* CMC Curve Generation
* Automatic Figure Generation
* Publication-Style Evaluation Outputs

---

# 🧠 Methodology

The proposed pipeline follows these stages:

1. **Infrared Image Input**
2. **Image Quality Assessment**

   * BRISQUE
   * PSNR
   * SSIM
3. **Adaptive Enhancement**

   * Zero-DCE for acceptable-quality images
   * SIGAN for low-quality images
4. **Feature Extraction**

   * ResNet-50 backbone
5. **Cross-Modal Matching**

   * Cosine Similarity
6. **Re-Ranking**

   * Expanded Cross Neighborhood (ECN)
7. **Evaluation**

   * Rank-1
   * mAP
   * CMC

---

# 🔄 Pipeline Architecture

![Pipeline](results/figures/Figure_1.png)

---

# 📊 Experimental Results

## Evaluation Metrics

| Method                    | Rank-1 (%) | mAP (%) | SSIM | PSNR (dB) |
| ------------------------- | ---------- | ------- | ---- | --------- |
| Baseline (No enhancement) | 2.15       | 2.00    | 0.57 | 31.92     |
| Zero-DCE only             | 2.62       | 2.48    | 0.60 | 33.65     |
| Ours (Zero-DCE + SIGAN)   | 3.54       | 2.48    | 0.71 | 37.44     |

---

# 📈 CMC Curve

![CMC Curve](results/figures/cmc_curve.png)

---

# 🖼️ Generated Outputs

## Figure 3 — Enhanced IR Images using Zero-DCE

![Figure 3](results/figures/Figure_3.png)

---

## Figure 4 — Further Refined IR Images using SIGAN

![Figure 4](results/figures/Figure_4.png)

---

# 📂 Project Structure

```text
Quality_Aware_VI_ReID/
├── checkpoints/
├── configs/
├── dataloaders/
├── dataset/
├── enhancement/
├── evaluation/
├── models/
├── quality_metrics/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── plots/
├── scripts/
├── main.py
├── requirements.txt
├── README.md
└── paper.pdf
```

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd Quality_Aware_VI_ReID
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚡ CUDA Setup

Install CUDA-enabled PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 📂 Dataset Setup

Dataset used:

* SYSU-MM01

Download separately from the official academic source and place inside:

```text
dataset/
```

Expected structure:

```text
dataset/
├── train/
├── test/
│   ├── query/
│   └── gallery/
```

---

# 🚀 Training

```bash
python main.py
```

---

# 📊 Generated Outputs

After evaluation, outputs are automatically saved into:

```text
results/
├── figures/
├── final_metrics.json
```

---

# 📚 Technologies Used

* Python
* PyTorch
* CUDA
* NumPy
* OpenCV
* Matplotlib
* scikit-image

---

# 📖 Research References

* Zero-DCE — CVPR 2020
* ResNet-50 — CVPR 2016
* SIGAN — ACM Multimedia
* SYSU-MM01 Dataset
* ECN Re-Ranking

---

# 🔮 Future Improvements

* Vision Transformer Backbones
* RegDB / LLCM Dataset Support
* Real-Time Inference Optimization
* Better Cross-Modal Alignment
* Stronger GAN-Based Refinement

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author
J Manoj