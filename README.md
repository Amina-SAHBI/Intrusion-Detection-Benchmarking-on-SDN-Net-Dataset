# 🛡️ AI-Driven SDN/NFV Intrusion Detection Benchmarking  
**Binary & Multi-Class Pipelines with Extensive ML/DL Evaluation**

---

## 📌 Repository Vision
Software-Defined Networking (SDN) and Network Function Virtualization (NFV) introduce centralized, programmable and controller-driven network intelligence. These properties change traffic behaviour, feature relevance and attack visibility compared to traditional distributed networks.  

This repository benchmarks **intrusion-detection classifiers at the flow level** through:

- **2 experimental branches**
  - `binary/` → `normal` vs `attack`
  - `multiclass/` → fine-grained **attack-type classification**

---

## 🧠 AI Learning Families

### 🔹 Classical Machine Learning (8 algorithms)
The ML scripts evaluate fast, transparent and feature-driven intrusion-detection biases using:

1. `Random Forest`
2. `Decision Tree`
3. `Logistic Regression`
4. `Gradient Boosting`
5. `K-Nearest Neighbors (KNN)`
6. `Gaussian Naive Bayes (GNB)`
7. `SGD Classifier`
8. `Linear SVM (SVC/SVM)`

### 🔹 Deep Learning (4 architectures)
DL scripts capture SDN flow abstraction and temporal behaviour using:

1. `Convolutional Neural Network (CNN)`
2. `Recurrent Neural Network (RNN)`
3. `Perceptron (minimal NN baseline)`
4. `Auto-Encoders`

---

## 🔍 Why 7 Pipeline Variants Per Experiment File?
Each classifier is tested under **7 different training pipelines** to observe the impact of:

- **class imbalance**
- **feature noise/redundancy**
- **synthetic sampling stability**
- **decision-boundary ambiguity reduction**

| Variant   | Sampling        | Feature Space | Goal |
|----------|-----------------|--------------|------|
| Model 1  | Unbalanced      | Full         | Baseline bias |
| Model 2  | Unbalanced      | RF top-K     | Noise-resistant FS |
| Model 3  | Unbalanced      | MI top-K     | Info-rich FS |
| Model 4  | SMOTE           | Full         | Oversampling effect |
| Model 5  | SMOTE          | RF top-K     | FS stability post-SMOTE |
| Model 6  | SMOTE          | MI top-K     | MI stability post-sampling |
| Model 7  | SMOTE-Tomek     | Full         | Boundary-ambiguity cleanup |

🎯 **Objective:** identify the most **stable and reliable** pipeline for SDN/NFV intrusion detection.

---

## ⚡ Results & Metrics (Inline Inspection)
Every model evaluation displays:

- `Accuracy`, `Precision`, `Recall`, `F1-Score`
- `ROC Curve + AUC` (when supported)
- `Confusion Matrix` (heatmap)
- `Classification Report`
- All plots and metrics are shown **inline**, not hidden in folders.

---


# 📂 Repository Structure

### 🔹 Branch: `binary/`

```

binary/
│
├── 📁 ML_Classifiers/
│   ├── rf_binary.py
│   ├── dt_binary.py
│   ├── lr_binary.py
│   ├── gb_binary.py
│   ├── knn_binary.py
│   ├── gnb_binary.py
│   ├── sgd_binary.py
│   └── svm_binary.py
│
└── 📁 DL_Models/
├── cnn_binary.py
├── rnn_binary.py
├── perceptron_binary.py
└── autoencoder_binary.py

```

---

### 🔹 Branch: `multiclass/`

```

multiclass/
│
├── 📁 ML_Classifiers/
│   ├── rf_multiclass.py
│   ├── dt_multiclass.py
│   ├── lr_multiclass.py
│   ├── gb_multiclass.py
│   ├── knn_multiclass.py
│   ├── gnb_multiclass.py
│   ├── sgd_multiclass.py
│   └── svm_multiclass.py
│
└── 📁 DL_Models/
├── cnn_multiclass.py
├── rnn_multiclass.py
├── perceptron_multiclass.py
└── autoencoder_multiclass.py

```

---

> 🌟 **Both branches contain the same classifiers, and each script runs 7 pipeline variants inside it.**
```


### 📦 Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib imblearn

python <file>.py --csv ../SDN-Net.csv --k 30
📜 License
This work is shared under the MIT License for open research usage.
You may reuse, adapt and redistribute the code while keeping credit and citation.

✨ Conclusion

The objective of this repository is to support research conclusions that are not tied to one classifier or one feature space, but validated through multiple pipelines and learning families. This ensures stronger reliability for intrusion detection in centralized, programmable SDN/NFV traffic.
