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
- Install dependencies via: `pip install -r requirements.txt`

## Experiment Scenarios Replicated
The 12 classifiers are tested under 7 dataset configurations described in the manuscript: unbalanced flows, SMOTE-balanced data, SMOTE-Tomek hybrid, and feature selection using Information Gain and Random Forest methods.

## License
- MIT License
