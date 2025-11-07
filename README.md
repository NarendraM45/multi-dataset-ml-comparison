# 🧠 Multi-Dataset Machine Learning Comparative Analysis

A comprehensive machine learning project analyzing the performance of multiple algorithms across **16 diverse datasets**, covering **Classification**, **Regression**, and **Clustering** tasks.  
This repository provides a unified framework to evaluate, visualize, and interpret how different models behave on datasets of varying complexity and data types.

---

## 📘 Project Overview

The goal of this project is to **compare algorithmic performance** on real-world and synthetic datasets using consistent preprocessing, hyperparameters, and evaluation metrics.  
The results highlight how certain models generalize better across different learning paradigms.

This study combines:
- 🧩 **Classification Models:** Logistic Regression, Random Forest, AdaBoost, SVM, KNN, Naive Bayes  
- 📈 **Regression Models:** Linear Regression, SVR, Random Forest Regressor, Bagging Regressor  
- 🔍 **Clustering Models:** K-Means, Agglomerative Clustering  

---

## 🧠 Datasets Used

| **#** | **Dataset Name** | **Type** | **Source / Description** |
|:--:|:----------------------|:----------------|:----------------|
| 1 | **Iris** | Classification | Flower classification — built-in (Scikit-learn) |
| 2 | **Wine** | Classification | Chemical analysis dataset — built-in |
| 3 | **Breast Cancer** | Classification | Tumor diagnostic dataset — built-in |
| 4 | **Digits** | Classification | Handwritten digits (0–9) — built-in |
| 5 | **Titanic** | Classification | Passenger survival prediction dataset |
| 6 | **Adult Income** | Classification | Predict income > \$50K/year — UCI dataset |
| 7 | **Credit Card Fraud** | Classification | Imbalanced fraud detection dataset — [Download manually](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| 8 | **MNIST** | Classification | Image-based digit dataset — Keras/Scikit-learn |
| 9 | **IMDB** | Classification | Sentiment classification for movie reviews |
| 10 | **Diabetes** | Regression | Predict disease progression — built-in |
| 11 | **California Housing** | Regression | Predict housing prices — built-in |
| 12 | **Linnerud** | Regression | Multi-output regression dataset — built-in |
| 13 | **make_classification** | Synthetic Classification | Generated with `sklearn.datasets.make_classification()` |
| 14 | **make_moons** | Synthetic Classification | Non-linear 2D dataset for testing kernel methods |
| 15 | **make_circles** | Synthetic Classification | Concentric circular dataset — kernel-based models |
| 16 | **make_blobs** | Clustering | Synthetic blobs for unsupervised clustering (K-Means, Agglomerative) |

> ⚠️ The `creditcard.csv` dataset is **too large to include in this repository**.  
> Please download it manually from Kaggle and place it in your project folder before running the script.

---

## 🧰 Models & Algorithms

| **Type** | **Models Used** |
|-----------|----------------|
| Classification | Logistic Regression, SVM (RBF), Decision Tree, Random Forest, AdaBoost, GaussianNB, MultinomialNB, KNN |
| Regression | Linear Regression, SVR, Decision Tree Regressor, Random Forest Regressor, Bagging Regressor |
| Clustering | K-Means, Agglomerative Clustering |

Each model was evaluated using the following metrics:
- **Classification:** Accuracy, F1-Score, Precision, Recall  
- **Regression:** Mean Squared Error (MSE)  
- **Clustering:** Silhouette Score  

---

## 📊 Results Overview

The experiments revealed several insights:

- **Random Forest** and **AdaBoost** consistently achieved top accuracy across most classification datasets.  
- **SVM (RBF)** excelled in handling non-linear patterns (e.g., `make_moons`, `make_circles`).  
- **KNN** performed effectively on small, well-separated datasets like Iris and Wine.  
- **Linear Regression** was efficient for low-dimensional regression problems like Diabetes.  
- **Random Forest Regressor** and **Bagging Regressor** minimized MSE for complex datasets such as California Housing.  
- For **Clustering**, both **K-Means** and **Agglomerative Clustering** achieved strong silhouette scores on `make_blobs`.

---

## 📂 Repository Structure
```
multi-dataset-ml-comparison/
│
├── plots/ # All generated charts and evaluation visuals
│ ├── classification_accuracy_boxplot.png
│ ├── classification_accuracy_heatmap.png
│ ├── classification_avg_accuracy.png
│ ├── classification_f1score_comparison.png
│ ├── clustering_silhouette_comparison.png
│ ├── regression_mse_boxplot.png
│ ├── regression_mse_comparison.png
│ ├── model_performance_summary.csv
│ └── top_models_summary.csv
│
├── mulitidaatset_compariosn.py # Main Python script for model training and evaluation
├── visual.ipynb # Jupyter notebook for visualization and insights
├── visual.pdf # Exported PDF report of visualizations
├── ml_results_full_2.csv # Master table of model results
├── ml_results_full_2.xlsx # Excel version of results
├── ml_results_full_2.html # Interactive HTML results summary
├── top_models_summary.csv # Summary of best-performing models
├── requirements.txt # Python dependencies list
├── .gitignore # Ignored files and system metadata
└── creditcard.csv # Credit Card Fraud dataset (download manually from Kaggle)
```

---

## 🧩 Visualizations

Key plots generated under `/plots` include:
- 📈 **Classification Accuracy Heatmap**
- 📊 **F1-Score Comparison**
- 🧮 **Regression MSE Distribution**
- 🔍 **Clustering Silhouette Score Visualization**

All visuals are compiled into [`visual.pdf`](visual.pdf).

---

## 🧪 How to Run the Project

### 1️⃣ Clone the repository
```bash
cd multi-dataset-ml-comparison
git clone https://github.com/NarendraM45/multi-dataset-ml-comparison.git
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣Add the Kaggle dataset

Download the Credit Card Fraud dataset from:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

and place it as creditcard.csv in the project folder.

### 4️⃣ Run the Python script
```bash
python mulitidaatset_compariosn.py
```
### 5️⃣ Explore results

# View:

Metrics in ml_results_full_2.csv

Graphs in the plots/ directory

Detailed analysis in visual.ipynb or visual.pdf

### 📋 Requirements

Core dependencies are listed in requirements.txt.
Typical stack:

pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm

### 🧭 Key Findings

No single model dominates across all datasets.
Performance depends on dataset size, structure, and feature complexity.

Ensemble models (Random Forest, AdaBoost) show superior generalization.

Simple models like Logistic Regression and Linear Regression offer interpretability and speed.

Dataset-specific optimization (balancing, feature scaling, hyperparameter tuning) remains crucial.

### 🧩 Future Enhancements

Integration of deep learning architectures for image and text datasets

Automated hyperparameter optimization via GridSearchCV or Bayesian tuning

Explainable AI (XAI) metrics for interpretability

Inclusion of runtime and efficiency benchmarking

### 🧾 Citation & Credits

This project was developed as part of an academic machine learning study by Narendra Mishra
(3rd Year B.Tech, CSIT).

Dataset Credit:

Credit Card Fraud Detection Dataset:
MLG-ULB, Kaggle

### 🌐 Repository Link

🔗 GitHub Repository:
https://github.com/NarendraM45/multi-dataset-ml-comparison

### 🏁 License

This project is released under the MIT License.
Feel free to fork, modify, and build upon it for research or learning purposes.
