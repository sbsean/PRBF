# PRBF: Phase-based Receive Beamforming for Ultrasound Imaging

This repository implements **IQ-domain Pixel-Centric Phase-based Receive Beamforming (PRBF)** for high-resolution ultrasound imaging. The core processing leverages phase rotation beamforming combined with IQ demodulation, pixel-wise delay compensation, and coherence optimization.

---

## 📄 Reference Paper

The implementation and methodology of this repository are inspired by:

**New Demodulation Method for Efficient Phase-Rotation-Based Beamforming**  
Anup Agarwal, Yang Mo Yoo, Fabio Kurt Schneider, Changqing Gao, Liang Mong Koh, and Yongmin Kim  
*IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, Vol. 54, No. 8, August 2007  
DOI: [10.1109/TUFFC.2007.437](https://doi.org/10.1109/TUFFC.2007.437)

> This paper proposes a two-stage demodulation (TSD) method for PRBF to reduce hardware complexity while maintaining high beamforming quality, and provides the theoretical foundation for PRBF development in this repository.

---

## 📂 Repository Structure

```bash
PRBF/
├── RF_data/                 # Raw RF input data (binary)
├── IQBF_1024x1024/          # IQ beamformed results (Pixel grid size: 1024x1024)
├── IQBF_512x512/
├── IQBF_256x256/
├── IQBF_2048x2048/
├── PixelGrid.py             # Pixel grid generator for Cartesian beamforming
├── main.py                  # Main PRBF beamforming pipeline
├── model.py                 # Core phase rotation beamforming model
├── utils.py                 # Utility functions for delay, apodization, DC cancel, etc.
├── dclinference.ipynb       # Coherence-based post-processing notebook
├── *.png                    # Sample output B-mode images
