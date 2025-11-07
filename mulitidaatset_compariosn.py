# ==============================================================
# ml_models_comparison_full_2.py
# --------------------------------------------------------------
# Refined ML comparison system addressing flagged issues
# ==============================================================
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, silhouette_score,
    precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Core models ---
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, BaggingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.multioutput import MultiOutputRegressor

# --- Dataset loaders ---
from sklearn.datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer,
    load_diabetes, load_linnerud, make_moons, make_circles,
    make_blobs, make_classification, fetch_california_housing, fetch_openml
)
import seaborn as sns
from tensorflow.keras.datasets import mnist, imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================================================
# Helper functions
# ==============================================================

def safe_silhouette(X, labels):
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        return silhouette_score(X, labels)
    except Exception:
        return np.nan

def fmt(d):
    return ", ".join([f"{k}={v}" for k, v in d.items()]) if d else "-"

def safe_scale(X):
    """Safely scale numeric data if numeric only"""
    if isinstance(X, pd.DataFrame):
        if np.issubdtype(X.dtypes[0], np.number):
            return StandardScaler().fit_transform(X)
        else:
            return X
    elif np.issubdtype(X.dtype, np.number):
        return StandardScaler().fit_transform(X)
    return X

# ==============================================================
# Evaluation functions
# ==============================================================

def evaluate_classification(dataset_name, X, y):
    results = []
    X = safe_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y if len(np.unique(y)) > 1 else None,
        random_state=RANDOM_STATE
    )

    models = [
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight='balanced', random_state=RANDOM_STATE))
        ]), fmt({"max_iter":400,"balanced":True})),
        ("SVM (RBF)", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight='balanced', random_state=RANDOM_STATE))
        ]), fmt({"kernel":"rbf","C":1.0,"balanced":True})),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
         fmt({"balanced":True})),
        ("Random Forest", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, class_weight='balanced'),
         fmt({"n_estimators":150,"balanced":True})),
        ("KNN", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
         fmt({"n_neighbors":7})),
        ("GaussianNB", GaussianNB(), fmt({})),
        ("MultinomialNB", Pipeline([("minmax", MinMaxScaler()), ("clf", MultinomialNB())]), fmt({})),
        ("AdaBoost", AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE), fmt({"n_estimators":150}))
    ]

    for name, model, params in tqdm(models, desc=f"🔹 {dataset_name} (Classification)", leave=False):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            results.append([dataset_name, "Classification", name, params, acc, f1, np.nan, prec, rec, ""])
        except Exception as e:
            results.append([dataset_name, "Classification", name, params, np.nan, np.nan, np.nan, np.nan, np.nan, str(e)])
    return results


def evaluate_regression(dataset_name, X, y):
    results = []
    X = safe_scale(X)
    multi = len(y.shape) > 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    models = [
        ("Linear Regression", LinearRegression(), fmt({})),
        ("SVR", Pipeline([("scaler", StandardScaler()), ("reg", SVR(C=1.0))]), fmt({"C":1.0})),
        ("Decision Tree Regressor", DecisionTreeRegressor(random_state=RANDOM_STATE), fmt({"max_depth":None})),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE), fmt({"n_estimators":150})),
        ("KNN Regressor", Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor(n_neighbors=7))]), fmt({"n_neighbors":7})),
        ("Bagging Regressor", BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, random_state=RANDOM_STATE), fmt({"n_estimators":100}))
    ]
    for name, model, params in tqdm(models, desc=f"🔹 {dataset_name} (Regression)", leave=False):
        try:
            if multi:
                model = MultiOutputRegressor(model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append([dataset_name, "Regression", name, params, np.nan, np.nan, mse, np.nan, np.nan, ""])
        except Exception as e:
            results.append([dataset_name, "Regression", name, params, np.nan, np.nan, np.nan, np.nan, np.nan, str(e)])
    return results


def evaluate_clustering(dataset_name, X, n_clusters):
    results = []
    models = [
        ("KMeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE), fmt({"n_clusters":n_clusters})),
        ("Agglomerative Clustering", AgglomerativeClustering(n_clusters=n_clusters), fmt({"n_clusters":n_clusters}))
    ]
    for name, model, params in tqdm(models, desc=f"🔹 {dataset_name} (Clustering)", leave=False):
        try:
            labels = model.fit_predict(X)
            sil = safe_silhouette(X, labels)
            results.append([dataset_name, "Clustering", name, params, sil, np.nan, np.nan, np.nan, np.nan, "Silhouette Score"])
        except Exception as e:
            results.append([dataset_name, "Clustering", name, params, np.nan, np.nan, np.nan, np.nan, np.nan, str(e)])
    return results

# ==============================================================
# Dataset Loader (improved sanitization)
# ==============================================================

def load_all_datasets():
    datasets = []

    # --- Toy datasets ---
    datasets += [
        ("Iris", "Classification", lambda: load_iris(return_X_y=True)),
        ("Digits", "Classification", lambda: load_digits(return_X_y=True)),
        ("Wine", "Classification", lambda: load_wine(return_X_y=True)),
        ("Breast Cancer", "Classification", lambda: load_breast_cancer(return_X_y=True)),
        ("Diabetes", "Regression", lambda: load_diabetes(return_X_y=True)),
        ("Linnerud", "Regression", lambda: load_linnerud(return_X_y=True)),
        ("make_classification", "Classification",
         lambda: make_classification(n_samples=1200, n_features=10, n_informative=4, n_classes=3, random_state=RANDOM_STATE)),
        ("make_moons", "Classification", lambda: make_moons(n_samples=800, noise=0.25, random_state=RANDOM_STATE)),
        ("make_circles", "Classification", lambda: make_circles(n_samples=800, noise=0.1, factor=0.5, random_state=RANDOM_STATE)),
        ("make_blobs", "Clustering", lambda: make_blobs(n_samples=800, centers=3, random_state=RANDOM_STATE))
    ]

    # --- Real datasets ---
    try:
        df_titanic = sns.load_dataset("titanic")
        df_titanic = df_titanic.dropna(subset=["survived"])
        df_titanic = df_titanic.select_dtypes(["number", "category", "object"])
        for col in df_titanic.columns:
            if str(df_titanic[col].dtype) in ["float64", "int64"]:
                df_titanic[col] = df_titanic[col].fillna(0)
            else:
                df_titanic[col] = df_titanic[col].fillna(df_titanic[col].mode()[0])
        if "alive" in df_titanic.columns:
            df_titanic = df_titanic.drop(columns=["alive"])
        X = pd.get_dummies(df_titanic.drop(columns=["survived"]), drop_first=True)
        y = df_titanic["survived"].astype(int)
        datasets.append(("Titanic", "Classification", lambda X=X, y=y: (X, y)))
    except Exception as e:
        print(f"Titanic dataset failed to load: {e}")

    # California Housing
    cal = fetch_california_housing()
    datasets.append(("California Housing", "Regression", lambda cal=cal: (cal.data, cal.target)))

    # Adult Income
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.dropna()
    X = pd.get_dummies(df.drop(columns=["class"]), drop_first=True)
    y = LabelEncoder().fit_transform(df["class"])
    datasets.append(("Adult Income", "Classification", lambda X=X, y=y: (X, y)))

    # MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.vstack([x_train.reshape((60000, -1)), x_test.reshape((10000, -1))]) / 255.0
    y = np.concatenate([y_train, y_test])
    datasets.append(("MNIST", "Classification", lambda X=X, y=y: (X, y)))

    # IMDB with TF-IDF
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_index = {v: k for k, v in word_index.items()}
    decode_review = lambda seq: " ".join([reverse_index.get(i - 3, "?") for i in seq if i >= 3])
    reviews = [decode_review(x) for x in np.concatenate([x_train[:3000], x_test[:3000]])]
    labels = np.concatenate([y_train[:3000], y_test[:3000]])
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(reviews).toarray()
    y = labels
    datasets.append(("IMDB", "Classification", lambda X=X, y=y: (X, y)))

    # Credit Card Fraud
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    datasets.append(("Credit Card Fraud", "Classification", lambda X=X, y=y: (X, y)))

    return datasets


# ==============================================================
# Main Runner
# ==============================================================

def run_all_models(save_csv=True, save_html=True):
    print("Running full ML model comparison (Refined)...\n")
    results = []

    for name, task, loader in tqdm(load_all_datasets(), desc="📦 Datasets", ncols=100):
        try:
            X, y = loader()
            print(f"\nProcessing {name} [{task}] ...")

            if task == "Classification":
                results += evaluate_classification(name, X, y)
            elif task == "Regression":
                results += evaluate_regression(name, X, y)
            elif task == "Clustering":
                n_clusters = len(np.unique(y)) if y is not None else 3
                results += evaluate_clustering(name, X, n_clusters)

        except Exception as e:
            print(f"{name} failed: {e}")

    cols = ["Dataset", "TaskType", "Model", "Hyperparameters", "Accuracy", "F1-Score", "MSE", "Precision", "Recall", "Notes"]
    df = pd.DataFrame(results, columns=cols)

    if save_csv:
        df.to_csv("ml_results_full_2.csv", index=False)
    if save_html:
        styled = (
            df.style
            .format(precision=4, na_rep="—")
            .background_gradient(subset=["Accuracy", "F1-Score"], cmap="Blues")
            .background_gradient(subset=["MSE"], cmap="Oranges_r")
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )
        with open("ml_results_full_2.html", "w", encoding="utf-8") as f:
            f.write(styled.to_html())

    print("\nAll refined evaluations complete!")
    print("📁 CSV saved as: ml_results_full_2.csv")
    print("📄 HTML report: ml_results_full_2.html\n")
    return df


if __name__ == "__main__":
    df = run_all_models()
    print(df.head())
