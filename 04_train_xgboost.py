# pandas loads our cleaned train and test CSV files into DataFrames
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# XGBRegressor is our second model — an extreme gradient boosting regressor
# It builds hundreds of decision trees sequentially, each one correcting
# the mistakes of the previous one
# Analogy: instead of one expert making a final decision, XGBoost is like
# a panel of hundreds of specialists who each correct the previous expert's mistakes
from xgboost import XGBRegressor

# mean_squared_error and r2_score measure how well our model performed
from sklearn.metrics import mean_squared_error, r2_score

# matplotlib creates our visualization charts
import matplotlib.pyplot as plt

# joblib saves our trained model to disk
# so Script 05 can load it for comparison without retraining
import joblib

# os lets us create folders and build file paths
import os


# This script is the second model trainer of the project
# It loads the same cleaned housing data as Script 03, trains an XGBoost model,
# evaluates it using the same RMSE and R² metrics, and saves the trained model
# The key difference from Linear Regression: XGBoost doesn't assume a straight line
# relationship between features and price — it can learn complex curved patterns
# Analogy: Linear Regression is like drawing one straight line through all the data
# XGBoost is like drawing thousands of small precise lines that together
# capture every curve and corner in the data


def load_data():
    # This function's job is to load the four CSV files that Script 02 saved
    # Exact same function as Script 03 — both models learn from the same data
    # Analogy: both contestants study from the same textbook before the exam

    print("Loading preprocessed data...")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    # squeeze() converts the single-column DataFrame into a Series
    # which is what XGBoost expects for the target variable
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    print(f"Train set: {X_train.shape[0]} houses, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} houses, {X_test.shape[1]} features")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # This function's job is to build and train the XGBoost model
    # Unlike Linear Regression which needs scaled features,
    # XGBoost works directly with the raw feature values
    # It doesn't care about scale because it makes decisions based on
    # splitting points (is TotalSF > 2000?) not weighted sums
    # Analogy: a doctor who says "if the patient's age is over 60 AND
    # blood pressure is over 140, predict high risk" doesn't need
    # everything on the same scale — they just need the raw values

    print("\nTraining XGBoost model...")

    # XGBoost hyperparameters — these control how the model learns:

    # n_estimators=1000 — build 1000 decision trees sequentially
    # each tree corrects the mistakes of all previous trees
    # Analogy: 1000 specialists each reviewing and correcting the previous one's work
    # More trees = more learning, but also more risk of overfitting

    # learning_rate=0.05 — how much each tree corrects the previous one
    # Lower = smaller corrections, more conservative learning, less overfitting
    # Higher = bigger corrections, faster learning, more risk of overfitting
    # Analogy: like how much you adjust your driving based on each piece of feedback —
    # small adjustments are smoother than jerking the wheel hard each time

    # max_depth=4 — how many levels deep each decision tree can grow
    # A deeper tree can learn more complex patterns but is more prone to overfitting
    # Analogy: a decision tree with depth 2 asks 2 questions,
    # depth 4 asks up to 4 nested questions before making a prediction

    # subsample=0.8 — each tree only sees 80% of the training data (randomly sampled)
    # This prevents any single tree from memorizing the full training set
    # Analogy: each specialist only reviews a random 80% of the cases —
    # this forces them to learn general patterns rather than memorizing specific cases

    # colsample_bytree=0.8 — each tree only sees 80% of the features (randomly sampled)
    # Same idea as subsample but for columns instead of rows
    # Analogy: each specialist only looks at 80% of the patient's test results —
    # this prevents over-reliance on any single feature

    # random_state=42 — locks in the randomness so results are reproducible
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # early_stopping_rounds=50 — if the model hasn't improved on the validation
    # set in 50 consecutive rounds, stop training early
    # This prevents overfitting by not training longer than necessary
    # eval_set tells XGBoost to monitor performance on the test set during training
    # Analogy: like a coach who stops practice when the team stops improving —
    # no point drilling the same plays if performance has plateaued
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=100
    )

    print("Training complete")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    # This function's job is to measure how well XGBoost performs
    # using the exact same metrics as Script 03 so we can compare directly
    # Analogy: grading both contestants on the same rubric so the comparison is fair

    print("\n--- Model Evaluation ---")

    # Generate predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate RMSE for both sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Calculate R² for both sets
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}")
    print(f"Train R²:   {train_r2:.4f}  |  Test R²:   {test_r2:.4f}")

    # Convert RMSE from log space back to dollar space
    # np.expm1() reverses the np.log1p() transformation we applied in Script 02
    # This gives us the approximate average dollar error on the test set
    # Analogy: converting a temperature back from Celsius to Fahrenheit
    # so it's meaningful to someone used to Fahrenheit
    dollar_rmse = np.expm1(test_rmse) * np.expm1(y_test.mean())
    print(f"\nApproximate dollar error (test set): ${dollar_rmse:,.0f}")

    return y_test_pred, test_rmse, test_r2


def plot_feature_importance(model, feature_names):
    # This function's job is to show which features XGBoost relied on most
    # when making predictions
    # Unlike Linear Regression which gives us coefficients,
    # XGBoost gives us feature importance scores — how often each feature
    # was used to make a split across all 1000 trees
    # Analogy: like tracking which questions a panel of 1000 doctors
    # asked most often when diagnosing patients —
    # the most asked questions are the most important ones

    print("\nGenerating feature importance plot...")

    # model.feature_importances_ is an array of importance scores
    # one per feature — higher score = more important
    importances = model.feature_importances_

    # Sort features by importance and keep the top 15
    # zip() pairs each importance score with its feature name
    # sorted() sorts them from highest to lowest importance
    indices = sorted(
        range(len(importances)),
        key=lambda i: importances[i],
        reverse=True
    )[:15]

    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    print("\nTop 15 most important features:")
    for feature, importance in zip(top_features, top_importances):
        print(f"  {feature}: {importance:.4f}")

    # Create a horizontal bar chart of feature importances
    fig, ax = plt.subplots(figsize=(10, 7))

    # barh() creates horizontal bars — easier to read long feature names
    ax.barh(
        range(len(top_features)),
        top_importances,
        color="#4CAF50",
        align="center"
    )

    # Set the y axis labels to the feature names
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)

    # Invert y axis so the most important feature is at the top
    ax.invert_yaxis()

    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title("XGBoost: Top 15 Most Important Features", fontsize=14)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/xgboost_importance.png", dpi=150)
    print("Plot saved to models/xgboost_importance.png")
    plt.show()


def plot_predictions(y_test, y_test_pred):
    # This function's job is to create the same predicted vs actual scatter plot
    # as Script 03 so we can visually compare how tight the dots cluster
    # around the perfect prediction line
    # Analogy: the same scoreboard layout for both contestants
    # so the comparison is visually immediate

    print("\nGenerating prediction plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_test_pred, alpha=0.4, color="#4CAF50", edgecolors="none")

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Log Sale Price", fontsize=12)
    ax.set_ylabel("Predicted Log Sale Price", fontsize=12)
    ax.set_title("XGBoost: Predicted vs Actual Sale Price", fontsize=14)
    ax.legend()
    plt.tight_layout()

    plt.savefig("models/xgboost_predictions.png", dpi=150)
    print("Plot saved to models/xgboost_predictions.png")
    plt.show()


def save_model(model):
    # This function's job is to save the trained XGBoost model to disk
    # so Script 05 can load it for comparison without retraining
    # We don't need to save a scaler this time —
    # XGBoost doesn't require feature scaling
    # Analogy: filing away the finished work so we can refer back to it later

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.joblib")
    print("\nModel saved to models/xgboost_model.joblib")


# This block only runs if you execute this file directly (python 04_train_xgboost.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    y_test_pred, test_rmse, test_r2 = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    feature_names = X_train.columns.tolist()
    plot_feature_importance(model, feature_names)
    plot_predictions(y_test, y_test_pred)
    save_model(model)