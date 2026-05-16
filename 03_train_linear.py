# pandas loads our cleaned train and test CSV files into DataFrames
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# LinearRegression is our first model — it fits a straight line through the data
# and learns a coefficient (weight) for every feature
# Analogy: LinearRegression is like a pricing formula —
# every feature adds or subtracts a certain amount from the predicted price
from sklearn.linear_model import LinearRegression

# StandardScaler normalizes our features so they're all on the same scale
# Without scaling, a feature like TotalSF (values in the thousands) would
# dominate features like HouseAge (values under 100)
# Analogy: like converting all measurements to the same unit before comparing them —
# you can't compare miles and centimeters without converting first
from sklearn.preprocessing import StandardScaler

# mean_squared_error calculates how wrong our predictions are on average
# r2_score measures how much of the price variation our model explains
from sklearn.metrics import mean_squared_error, r2_score

# matplotlib creates our visualization charts
import matplotlib.pyplot as plt

# joblib saves our trained model and scaler to disk
# so Script 05 can load them for comparison without retraining
import joblib

# os lets us create folders and build file paths
import os


# This script is the first model trainer of the project
# It loads the cleaned housing data, scales the features, trains a Linear Regression
# model, evaluates it using RMSE and R², and saves the trained model to disk
# Analogy: think of this script as a real estate appraiser who studies thousands
# of past sales (training data), learns how each feature affects price (coefficients),
# then uses that knowledge to estimate the value of new houses (predictions)


def load_data():
    # This function's job is to load the four CSV files that Script 02 saved
    # X_train and y_train are used for training
    # X_test and y_test are used for evaluation
    # Analogy: picking up the prepped and portioned ingredients the prep cook left ready

    print("Loading preprocessed data...")

    # Load the feature matrices
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    # Load the target vectors — squeeze() converts the DataFrame into a Series
    # (a single column) which is what sklearn expects for the target variable
    # Analogy: instead of a table with one column, we want a simple list of numbers
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    print(f"Train set: {X_train.shape[0]} houses, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} houses, {X_test.shape[1]} features")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    # This function's job is to normalize all features to the same scale
    # Linear Regression is sensitive to feature scale — a feature with values
    # in the thousands will have a much smaller coefficient than a feature
    # with values between 0 and 1, even if both are equally important
    # StandardScaler fixes this by converting every feature to have
    # mean = 0 and standard deviation = 1
    # Analogy: like grading all students on a curve so that every subject
    # is on the same scale regardless of how hard the exam was

    print("\nScaling features...")

    # Create the scaler object
    scaler = StandardScaler()

    # fit_transform() on training data:
    # fit = learn the mean and standard deviation of each feature
    # transform = apply the scaling using those learned values
    # Analogy: the scaler reads all the training data to understand
    # the typical range of each feature, then rescales everything
    X_train_scaled = scaler.fit_transform(X_train)

    # transform() only on test data — we use the same scaling rules
    # learned from the training data, not new rules from the test data
    # Analogy: the exam uses the same grading curve as the practice tests —
    # we don't invent new rules for the test
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled to mean=0, standard deviation=1")

    return scaler, X_train_scaled, X_test_scaled


def train_model(X_train_scaled, y_train):
    # This function's job is to train the Linear Regression model
    # It finds the best coefficient (weight) for every one of the 265 features
    # so that the weighted sum of features predicts the log sale price as accurately
    # as possible
    # Analogy: like a pricing formula where every feature adds or subtracts
    # a certain dollar amount — "each additional bathroom adds $8,000,
    # each year of age subtracts $500" and so on

    print("\nTraining Linear Regression model...")

    # Create and train the model in one step
    # fit() is where all the learning happens — it solves for the best
    # coefficients using a mathematical technique called Ordinary Least Squares
    # Analogy: like a student finding the formula that best fits all the past data points
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("Training complete")
    return model


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    # This function's job is to measure how well the model performs
    # on both the training set and the test set
    # Comparing both helps us detect overfitting —
    # if training score is much higher than test score, the model memorized the data
    # Analogy: comparing a student's practice test scores to their real exam score —
    # if practice was 95% but the exam was 60%, they memorized rather than understood

    print("\n--- Model Evaluation ---")

    # Generate predictions on both sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate RMSE on log-transformed prices
    # We take the square root of mean_squared_error to get RMSE
    # RMSE in log space tells us how far off our log predictions are
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Calculate R² — how much of the price variation our model explains
    # 1.0 = perfect, 0.0 = no better than always predicting the average price
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}")
    print(f"Train R²:   {train_r2:.4f}  |  Test R²:   {test_r2:.4f}")

    # Convert RMSE back from log space to dollar space so it's interpretable
    # np.expm1() is the reverse of np.log1p() — it undoes the log transformation
    # This gives us the approximate average dollar error of our predictions
    # Analogy: like converting a temperature back from Celsius to Fahrenheit
    # so it's meaningful to someone used to Fahrenheit
    dollar_rmse = np.expm1(test_rmse)
    print(f"\nApproximate dollar error (test set): ${dollar_rmse:,.0f}")

    return y_test_pred, test_rmse, test_r2


def plot_predictions(y_test, y_test_pred):
    # This function's job is to create a scatter plot of predicted vs actual prices
    # A perfect model would have all points on a straight diagonal line
    # Points scattered around the line show us where the model goes wrong
    # Analogy: like plotting your estimated arrival times vs actual arrival times —
    # the closer to the diagonal, the more accurate your estimates

    print("\nGenerating prediction plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot — each point is one house
    # x axis = actual log price, y axis = predicted log price
    ax.scatter(y_test, y_test_pred, alpha=0.4, color="#2196F3", edgecolors="none")

    # Draw the perfect prediction line — if predicted = actual, the point lies here
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Log Sale Price", fontsize=12)
    ax.set_ylabel("Predicted Log Sale Price", fontsize=12)
    ax.set_title("Linear Regression: Predicted vs Actual Sale Price", fontsize=14)
    ax.legend()
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/linear_predictions.png", dpi=150)
    print("Plot saved to models/linear_predictions.png")
    plt.show()


def show_top_features(model, feature_names):
    # This function's job is to show which features the model weighted most heavily
    # Each feature has a coefficient — a positive coefficient means the feature
    # pushes the price up, a negative coefficient means it pushes the price down
    # Analogy: like reading the pricing formula and seeing which ingredients
    # add the most value to the final dish

    print("\n--- Top 15 Most Influential Features ---")

    # model.coef_ is an array of coefficients — one per feature
    # zip() pairs each coefficient with its feature name
    # sorted() sorts them by absolute value so we see the most influential first
    coef_pairs = sorted(
        zip(model.coef_, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )[:15]

    for coef, feature in coef_pairs:
        direction = "+" if coef > 0 else "-"
        print(f"  {direction} {feature}: {coef:.4f}")


def save_model(model, scaler):
    # This function's job is to save the trained model and scaler to disk
    # Both need to be saved — the scaler is needed to scale new data
    # the same way we scaled the training data
    # Analogy: saving both the recipe and the measuring cups —
    # you need both to recreate the dish exactly

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_model.joblib")
    joblib.dump(scaler, "models/linear_scaler.joblib")

    print("\nModel saved to models/linear_model.joblib")
    print("Scaler saved to models/linear_scaler.joblib")


# This block only runs if you execute this file directly (python 03_train_linear.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    y_test_pred, test_rmse, test_r2 = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    feature_names = X_train.columns.tolist()
    show_top_features(model, feature_names)
    plot_predictions(y_test, y_test_pred)
    save_model(model, scaler)