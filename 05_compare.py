# pandas loads our test data and helps us organize results
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# We need these to reload and run both trained models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# mean_squared_error and r2_score let us evaluate both models
# on the exact same test set so the comparison is perfectly fair
from sklearn.metrics import mean_squared_error, r2_score

# joblib loads our saved models and scaler from disk
import joblib

# matplotlib creates our comparison charts
import matplotlib.pyplot as plt

# os lets us create folders and build file paths
import os


# This script is the final script of the project
# It loads both trained models, runs them on the same test set,
# and produces a side-by-side comparison of their performance
# including RMSE, R², dollar error, and visual charts
# Analogy: the final judging panel that takes two contestants
# and puts them through the exact same challenge to compare them fairly


def load_data():
    # This function's job is to load the test set that Script 02 saved
    # Both models will be evaluated on this exact same set of houses
    # so the comparison is completely fair
    # Analogy: both contestants sit the same exam on the same day

    print("Loading test data...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    print(f"Test set: {X_test.shape[0]} houses, {X_test.shape[1]} features")
    return X_test, y_test


def load_linear_model():
    # This function's job is to load the saved Linear Regression model
    # and its scaler from disk
    # We need both — the scaler to scale the test features the same way
    # we scaled the training features, and the model to make predictions
    # Analogy: retrieving both the recipe and the measuring cups from the filing cabinet

    print("\nLoading Linear Regression model...")
    model = joblib.load("models/linear_model.joblib")
    scaler = joblib.load("models/linear_scaler.joblib")
    print("Linear Regression model loaded")
    return model, scaler


def load_xgboost_model():
    # This function's job is to load the saved XGBoost model from disk
    # No scaler needed — XGBoost doesn't require feature scaling
    # Analogy: retrieving just the recipe, no measuring cups needed

    print("Loading XGBoost model...")
    model = joblib.load("models/xgboost_model.joblib")
    print("XGBoost model loaded")
    return model


def evaluate_linear(model, scaler, X_test, y_test):
    # This function's job is to run the test houses through the Linear Regression
    # model and calculate its performance metrics
    # Analogy: contestant 1 sits the exam

    print("\n--- Evaluating Linear Regression ---")

    # Scale the test features using the same scaler from training
    # We must use transform() not fit_transform() — we don't relearn the scaling rules
    # Analogy: applying the same grading curve from the practice test to the real exam
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Convert RMSE from log space to dollar space
    # We multiply by the exponent of the mean price to get a dollar figure
    dollar_rmse = np.expm1(rmse) * np.expm1(y_test.mean())

    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"Approximate dollar error: ${dollar_rmse:,.0f}")

    return y_pred, rmse, r2


def evaluate_xgboost(model, X_test, y_test):
    # This function's job is to run the test houses through the XGBoost model
    # and calculate its performance metrics using the exact same approach
    # Analogy: contestant 2 sits the same exam

    print("\n--- Evaluating XGBoost ---")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    dollar_rmse = np.expm1(rmse) * np.expm1(y_test.mean())

    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"Approximate dollar error: ${dollar_rmse:,.0f}")

    return y_pred, rmse, r2


def plot_comparison(linear_rmse, linear_r2, xgb_rmse, xgb_r2):
    # This function's job is to create a side-by-side bar chart
    # comparing both models on RMSE and R²
    # Analogy: the scoreboard at the end of the competition

    print("\nGenerating comparison chart...")

    # Create a figure with two side-by-side charts
    # One for RMSE (lower is better) and one for R² (higher is better)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    models = ["Linear\nRegression", "XGBoost"]
    colors = ["#2196F3", "#4CAF50"]

    # --- Left chart: RMSE comparison ---
    rmse_values = [linear_rmse, xgb_rmse]
    bars1 = ax1.bar(models, rmse_values, color=colors, width=0.5)

    # Add value labels on top of each bar
    for bar, val in zip(bars1, rmse_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax1.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax1.set_title("RMSE Comparison", fontsize=14)
    ax1.set_ylim(0, max(rmse_values) * 1.2)

    # --- Right chart: R² comparison ---
    r2_values = [linear_r2, xgb_r2]
    bars2 = ax2.bar(models, r2_values, color=colors, width=0.5)

    # Add value labels on top of each bar
    for bar, val in zip(bars2, r2_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax2.set_ylabel("R² (higher is better)", fontsize=12)
    ax2.set_title("R² Comparison", fontsize=14)
    ax2.set_ylim(0, 1.1)

    plt.suptitle("Model Comparison: Linear Regression vs XGBoost\nHouse Price Prediction", fontsize=14)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/comparison_chart.png", dpi=150)
    print("Chart saved to models/comparison_chart.png")
    plt.show()


def plot_predictions_comparison(y_test, linear_pred, xgb_pred):
    # This function's job is to plot predicted vs actual prices for both models
    # side by side so we can visually see which model's dots cluster
    # more tightly around the perfect prediction line
    # Analogy: showing both contestants' answer sheets next to each other

    print("Generating predictions comparison chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    min_val = y_test.min()
    max_val = y_test.max()

    # --- Left chart: Linear Regression predictions ---
    ax1.scatter(y_test, linear_pred, alpha=0.4, color="#2196F3", edgecolors="none")
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("Actual Log Sale Price", fontsize=11)
    ax1.set_ylabel("Predicted Log Sale Price", fontsize=11)
    ax1.set_title("Linear Regression", fontsize=13)
    ax1.legend()

    # --- Right chart: XGBoost predictions ---
    ax2.scatter(y_test, xgb_pred, alpha=0.4, color="#4CAF50", edgecolors="none")
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax2.set_xlabel("Actual Log Sale Price", fontsize=11)
    ax2.set_ylabel("Predicted Log Sale Price", fontsize=11)
    ax2.set_title("XGBoost", fontsize=13)
    ax2.legend()

    plt.suptitle("Predicted vs Actual Sale Price: Both Models", fontsize=14)
    plt.tight_layout()

    plt.savefig("models/predictions_comparison.png", dpi=150)
    print("Chart saved to models/predictions_comparison.png")
    plt.show()


def print_summary(linear_rmse, linear_r2, xgb_rmse, xgb_r2, y_test):
    # This function's job is to print a clean final summary table
    # comparing both models side by side in the terminal
    # Analogy: the final scoreboard after the competition

    print("\n" + "=" * 55)
    print("FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<25} {'Linear Reg':>12} {'XGBoost':>12}")
    print("-" * 55)
    print(f"{'Test RMSE':<25} {linear_rmse:>12.4f} {xgb_rmse:>12.4f}")
    print(f"{'Test R²':<25} {linear_r2:>12.4f} {xgb_r2:>12.4f}")

    # Dollar errors
    linear_dollar = np.expm1(linear_rmse) * np.expm1(y_test.mean())
    xgb_dollar = np.expm1(xgb_rmse) * np.expm1(y_test.mean())
    print(f"{'Approx Dollar Error':<25} ${linear_dollar:>10,.0f} ${xgb_dollar:>10,.0f}")

    print("=" * 55)

    # Determine winner on each metric
    rmse_winner = "XGBoost" if xgb_rmse < linear_rmse else "Linear Regression"
    r2_winner = "XGBoost" if xgb_r2 > linear_r2 else "Linear Regression"
    print(f"\nLower RMSE (better):  {rmse_winner}")
    print(f"Higher R² (better):   {r2_winner}")


# This block only runs if you execute this file directly (python 05_compare.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    X_test, y_test = load_data()

    linear_model, scaler = load_linear_model()
    xgb_model = load_xgboost_model()

    linear_pred, linear_rmse, linear_r2 = evaluate_linear(
        linear_model, scaler, X_test, y_test
    )
    xgb_pred, xgb_rmse, xgb_r2 = evaluate_xgboost(
        xgb_model, X_test, y_test
    )

    plot_comparison(linear_rmse, linear_r2, xgb_rmse, xgb_r2)
    plot_predictions_comparison(y_test, linear_pred, xgb_pred)
    print_summary(linear_rmse, linear_r2, xgb_rmse, xgb_r2, y_test)