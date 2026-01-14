# 🫀 Takotsubo Syndrome (TTS) Anomaly Classification

This repository contains the implementation and analysis of an **Unsupervised Anomaly Detection** framework designed to identify Takotsubo Syndrome (TTS) from cardiac imaging. By training exclusively on healthy control data, the model learns to recognize pathological deviations in TTS patients without the need for manual labels during training.

## 📊 Performance Summary
Our model demonstrates high diagnostic accuracy and aligns with clinical pathological findings:
* **Global Classification Accuracy (ROC-AUC)**: **0.92**
* **Myocardium-Specific Accuracy (ROC-AUC)**: **0.8905**
* **Statistical Significance**: **p < 0.001** (Mann-Whitney U test between Normal and TTS groups)

## 🌟 Key Highlights
* **Anatomical Validation**: Although the model was never provided with anatomical masks during training, post-hoc analysis reveals that it naturally identifies the **Myocardium** as the primary site of anomaly (AUC 0.89), consistent with the clinical presentation of TTS.
* **Unsupervised Feature Learning**: Utilizes a 3D Autoencoder combined with Flow Matching to capture the complex geometry of a "normal" heart, allowing it to detect "ballooning" patterns as high-error regions.
* **Clinical Explainability**: Through error score distribution analysis, we provide quantitative evidence that the model's decision-making process is rooted in actual tissue deformation rather than random noise.

## 📈 Visualizations

### 1. ROC Curve Comparison
The ROC curve demonstrates that the **Myocardium** region provides the strongest signal for differentiating TTS from healthy controls compared to the LV or RV cavities alone.
<img src="https://raw.githubusercontent.com/chanhobong/AE_TTS_Analysis/main/outputs/lv_rv_analysis/myocardium_boxplot.png" width="500">

### 2. Region-Specific Error Distribution
The boxplot illustrates a clear, statistically significant separation ($p < 0.001$) in reconstruction error scores within the myocardial region between the two groups.
<img src="https://raw.githubusercontent.com/chanhobong/AE_TTS_Analysis/main/outputs/lv_rv_analysis/roc_curves_comparison.png" width="400">

## 📂 Project Structure
* `utils/analyze_lv_rv_error_scores.py`: Core script for region-specific error quantification and statistical testing.
* `outputs/`: Contains generated plots (`.png`) and quantitative summary reports (`.csv`).
* `models/`: Architecture definition for the 3D-AE and Flow Matching components.

## 🛠 Future Work (Planned)
* Integration of clinical metadata (Age, Sex) to evaluate potential improvements in diagnostic specificity.
* Matched-group analysis to mitigate potential "Shortcut Learning" from demographic biases.

---
*Note: This project is part of a research thesis focusing on robust medical anomaly detection.*
