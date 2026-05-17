import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import joblib


def train_model(df, model_path="models/dt_model.pkl"):

    # =========================
    # 1. Encoding
    # =========================
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Revenue', axis=1)
    y = df['Revenue']

    # =========================
    # 2. Split data
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # 3. Base Model
    # =========================
    dt = DecisionTreeClassifier(random_state=42)

    # =========================
    # 4. Hyperparameter Grid
    # =========================
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"]
    }

    # =========================
    # 5. GridSearchCV
    # =========================
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # =========================
    # 6. Best Model
    # =========================
    best_model = grid_search.best_estimator_

    print("Best Parameters Found:")
    print(grid_search.best_params_)

    # =========================
    # 7. Save Model + Features
    # =========================
    joblib.dump((best_model, X.columns), model_path)

    return best_model, X_test, y_test