


# 🛡️ SDN-Net Intrusion Detection Benchmarking

A benchmarking framework for **flow-level intrusion detection in SDN/NFV networks**, evaluated across:
- **2 branches:** `Binary` and `Multi-Class`
- **12 classifiers per branch:** 8 ML + 4 DL
- **7 pipeline variants in each experiment file**
- **Parallel evaluation on both Binary and Multi-Class tracks**

---

## 🎯 Purpose of the 7-Variant Pipelines
Each classifier is trained **7 times under different data and feature-space configurations** to analyze:
- the effect of **class imbalance**
- the sensitivity to **noisy or redundant features**
- the reliability of **decision boundaries**
- the stability of models under **synthetic sampling shifts**

🔬 *This design prevents conclusions that are tied to a single data condition or feature view.*

---

## 🧠 Why ML & DL Diversity?
We include multiple learning philosophies to observe how SDN flow behavior is captured differently:
- **ML (8 algos):** rule-based, ensemble, linear, margin-based, neighborhood, probabilistic, scalable learners
- **DL (4 archs):** spatial feature abstraction (CNN), temporal flow learning (RNN), neural baseline, anomaly reconstruction (Auto-Encoder)

✅ *Broad coverage ensures findings are usable for research decisions and real network-security insights.*

---

## 📊 Metrics Displayed for Each Variant
Every pipeline outputs **inline results** including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (heatmap)
- ROC Curve & AUC (when supported)
- Classification Report
- Learning Curve (accuracy proxy for convergence and capacity trends)

---

## 📁 Branch Structure

### `Binary/` and `Multi-Class/` contain the same classifier list:





















# AI-Driven SDN/NFV Intrusion Detection Benchmarking
  # Binary & Multi-Class Pipelines with Extensive ML/DL Evaluation

Modern SDN/NFV networks operate under centralized, programmable control, producing traffic patterns that differ radically from conventional distributed architectures. Detecting attacks in these environments requires evaluating more than one learning path and more than one feature view.
This repository benchmarks intrusion-detection pipelines over flow-level traffic using:
•	Two experiment branches:
o	binary/ → normal vs attack detection
o	multiclass/ → precise attack-type identification
•	A broad spectrum of learning models:
o	8 classical Machine Learning algorithms
o	4 Deep Learning architectures
•	Seven pipeline variants per experiment file, to rigorously measure:
o	bias caused by imbalance
o	sensitivity to feature noise
o	stability under synthetic sampling
o	decision-boundary trustworthiness
The project delivers insights that are reproducible, measurable, interpretable, and suitable for real SDN/NFV security conclusions.

# Preprocessing Principles
All experiments start from the same preparation workflow:
✔ Removes empty or unusable columns
✔ Cleans numerical inconsistencies (NaN, ±∞)
✔ Converts attack labels into numeric format
✔ Standardizes features using StandardScaler
✔ Performs stratified train/test splitting
✔ Guarantees a clean, numerical input space suitable for ML/DL

# Models Variants
We train seven versions per experiment file, not seven classifiers, but seven pipeline conditions, all built on the same algorithm family.
This allows us to compare how data structure and feature filtering influence detection behaviour in SDN flows.
Variant	Sampling Strategy	Feature View	Purpose
Model 1	Original (Unbalanced)	All Features	Baseline model bias measurement
Model 2	Unbalanced	RF Importance top-K	Tests detection after noise-resistant ranking
Model 3	Unbalanced	Mutual Information top-K	Tests info-rich feature efficiency under skew
Model 4	SMOTE Balanced	All Features	Measures synthetic oversampling generalization
Model 5	SMOTE	RF Importance top-K	Stability of RF-FS under sampled distribution
Model 6	SMOTE	Mutual Info top-K	Stability of IG/MI-FS under balanced input
Model 7	SMOTE-Tomek Balanced	All Features	Impact of cleaning ambiguous flow boundaries
--> Scientific Goal: determine the most robust and trustworthy training pipeline before tying conclusions to model performance.
# AI Evaluation Algorithms
Intrusion detection in programmable networks cannot rely on a single classifier philosophy. We evaluate models that learn differently, decide differently, and generalize differently: 
•	To compare linear vs non-linear learning on SDN flows
•	To validate classification reliability across different inductive biases
•	To detect overfitting behaviours (deep vs non-deep families)
•	To understand how decision boundaries shift under different feature views
•	To benchmark static inference vs sequential flow learning
•	To ensure conclusions are not dependent on one model family

Classical ML (8 scripts per branch) These models test feature-quality impact and detection bias under non-deep learning assumptions:

Random Forest	ensemble voting + noise resilience
Decision Tree	rule-based logic + explainability
Logistic Regression	linear detection baseline
Gradient Boosting	sequential error-correction
K-Nearest Neighbors	neighbourhood proximity reasoning
Gaussian NB	probabilistic independence view
SGD Classifier	scalable online learning
Linear SVM/SVC	margin-based separation bias

Deep Learning (4 scripts per branch) These models test representation abstraction and traffic dynamics:

CNN	Auto-extracts spatial/flow feature combinations
RNN	Learns temporal dependencies in flow sequences
Perceptron	Minimal neural baseline (no depth), for sanity check
Auto-Encoders	Learns reconstruction behaviour to model anomalies

--> Conclusion: broad algorithmic coverage ensures scientific defensibility and practical generalization.

# Performance Metrics
For every model in every file:
📊 Accuracy, Precision, Recall, F1-Score
📈 ROC curve + AUC (when available)
🔢 Confusion Matrix (heatmap)
📑 Full classification report (attack distribution, false alerts, missed detection)
Visual results appear inline so you never have to open folders to inspect plots.

# Final Repository Layout
Branch: binary/
binary/
   ├── rf_binary.py            (7 models inside)
   ├── dt_binary.py            (7 models inside)
   ├── lr_binary.py            (7 models inside)
   ├── gb_binary.py            (7 models inside)
   ├── knn_binary.py           (7 models inside)
   ├── gnb_binary.py           (7 models inside)
   ├── sgd_binary.py           (7 models inside)
   └── svc_binary.py           (7 models inside)

   ├── cnn_binary.py           (7 models inside)
   ├── rnn_binary.py           (7 models inside)
   ├── perceptron_binary.py    (7 models inside)
   └── autoencoder_binary.py   (7 models inside)
Branch: multiclass/
multiclass/
   ├── rf_multiclass.py            (7 models inside)
   ├── dt_multiclass.py            (7 models inside)
   ├── lr_multiclass.py            (7 models inside)
   ├── gb_multiclass.py            (7 models inside)
   ├── knn_multiclass.py           (7 models inside)
   ├── gnb_multiclass.py           (7 models inside)
   ├── sgd_multiclass.py           (7 models inside)
   └── svc_multiclass.py           (7 models inside)

   ├── cnn_multiclass.py           (7 models inside)
   ├── rnn_multiclass.py           (7 models inside)
   ├── perceptron_multiclass.py    (7 models inside)
   └── autoencoder_multiclass.py   (7 models inside)
🌟 Rule: Every script exists in both branches, and each script trains 7 models internally.

📦 Local Installation
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib imblearn

▶️ Run Experiments
Display only (default):
python <file>.py --csv ../SDN-Net.csv --k 20
Save artifacts too:
python <file>.py --csv ../SDN-Net.csv --k 40 --save --outdir results/
Pause between plots:
python <file>.py --csv ../SDN-Net.csv --k 10 --pause

📜 Research License
You may reuse or distribute with:
📝 MIT

✨ Final Words
This repository exists to help the research community understand how model bias, feature noise, and class imbalance jointly affect intrusion detection in centralized programmable networks. By evaluating multiple algorithmic philosophies across seven pipeline variants for both binary and multi-class scenarios, we provide scientifically defensible conclusions built on reproducible benchmarks.




# Intrusion Detection Benchmarking on SDN-Net Dataset

This repository contains the code used to reproduce all benchmarking experiments from our manuscript in SDN/NFV intrusion detection. It includes 12 implemented classifiers covering ML (Random Forest, Decision Tree, KNN, Gaussian NB, Logistic Regression, SGD, Linear SVC, Gradient Boosting) and DL models (RNN, CNN, MLP, Autoencoder) evaluated on SDN-Net flows.

## Repository Content
- **Preprocessing scripts**: data cleaning, encoding, scaling and dataset fusion.
- **ML/DL model notebooks and scripts**: training pipelines for each classifier.
- **Evaluation scripts**: metrics computation (accuracy, precision, recall, F1-score), confusion matrix, ROC and learning curves as reported in the manuscript.

## How to Use
1. Download the fused SDN-Net CSV dataset from Zenodo using the DOI already referenced in the manuscript.
2. Place the dataset in the local `data/` folder.
3. Run the notebooks or Python scripts in the order described in the manuscript’s experimental section.
4. Cite the manuscript and the Zenodo DOI when reusing this code.

## Tools & Dependencies
- Python, TensorFlow/Keras, and scikit-learn
- Install dependencies

## Experiment Scenarios Replicated
The 12 classifiers are tested under 7 dataset configurations described in the manuscript: unbalanced flows, SMOTE-balanced data, SMOTE-Tomek hybrid, and feature selection using Information Gain and Random Forest methods.

## License
- MIT License
