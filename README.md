# SMORFFI Extra Feature Extraction

This repository provides a comprehensive data preprocessing and feature extraction pipeline for the SMORFFI (Large-Scale Same-Model 2.4 GHz Wi-Fi Dataset) framework. It extends the original raw I/Q dataset by extracting 14 additional statistical, non-linear, and power-based features from preamble signals to enhance Machine Learning models for same-model RF fingerprinting and IoT device identification.

## Original Dataset Source

The raw dataset required to run this pipeline can be downloaded from Kaggle:
Kaggle Dataset: https://www.kaggle.com/datasets/yinchen1986/rffi-123-m5stack-iq-wifi-802-11g-2-4g

The dataset consists of raw I/Q preamble samples collected from 123 same-model commercial IEEE 802.11g devices (M5Stack Core2 transmitters) received by a USRP B210 Software-Defined Radio.

## Extracted Features

While the baseline SMORFFI framework extracts standard physical layer attributes (such as Carrier Frequency Offset (CFO), phase/magnitude errors, and fractal dimensions), this pipeline enriches the feature space by calculating 14 supplementary characteristics directly from the preambles.

The features extracted via `second_features.py` include:

| Category | Features |
| :--- | :--- |
| **Statistical Distribution** | Skewness I, Skewness Q, Kurtosis I, Kurtosis Q |
| **Offset & Imbalance** | DC Offset I, DC Offset Q, I/Q Gain Imbalance, I/Q Phase Imbalance |
| **Power & Amplitude** | PAPR (Peak-to-Average Power Ratio), RMS Power, Amplitude STD |
| **Time-Series & Complexity** | CFO Drift Rate, Sample Entropy, Hurst Exponent |

## Pipeline Scripts

The repository processes the raw dataset through a modular four-step pipeline:

1. **`first_features.py`**: Reads the raw preamble data, estimates the coarse and fine CFO, corrects the frequency offsets, performs subcarrier equalization, and computes base physical layer features (CFO, phase error vectors, magnitude error vectors, and fractal dimensions).
2. **`second_features.py`**: Iterates over the raw signals to extract the 14 supplementary statistical, power-based, and non-linear complexity features.
3. **`merge.py`**: Combines the outputs of the first two scripts side-by-side into a single dataset, removing duplicate identification columns to prevent redundancy.
4. **`kick_bozuk_lines_etc.py`**: Cleans the merged dataset by filtering out corrupted or incomplete packets (retaining rows with "Geçerli" status) and dropping unnecessary columns to output a ready-to-train dataset.

```bash
python first_features.py
python second_features.py
python merge.py
python kick_bozuk_lines_etc.py
```
### Prerequisites
Install the required Python libraries using pip:
```bash
pip install pandas numpy scipy tqdm
```
## References and Citation

If you use this dataset or feature extraction pipeline in your research, please cite the original SMORFFI paper:

### IEEE Format
Z. Guo, Z. Jia, J. Zhu, W. Huang, and Y. Chen, "SMORFFI: A Large-Scale Same-Model 2.4 GHz Wi-Fi Dataset and Reproducible Framework for RF Fingerprinting," arXiv preprint arXiv:2511.07770v2 [cs.NI], Nov. 2025.

### BibTeX
```bibtex
@article{guo2025smorffi,
  title={SMORFFI: A Large-Scale Same-Model 2.4 GHz Wi-Fi Dataset and Reproducible Framework for RF Fingerprinting},
  author={Guo, Zewei and Jia, Zhen and Zhu, JinXiao and Huang, Wenhao and Chen, Yin},
  journal={arXiv preprint arXiv:2511.07770v2},
  year={2025},
  month={Nov}
}
```
