# Project 3: House Price Prediction (Regression)

## The Problem This Project Solves

When someone wants to buy or sell a house, one of the hardest questions is: **what is this house actually worth?**

Real estate agents estimate prices based on experience and intuition. But what if a machine could look at 79 measurable features of a house — square footage, neighborhood, number of bathrooms, year built, garage size — and predict the sale price with high accuracy?

That's exactly what this project builds. A machine learning pipeline that takes raw housing data and predicts the sale price of any house it hasn't seen before. This kind of model is used by companies like Zillow, Redfin, and major banks for property valuation at scale.

---

## What We're Building

A full regression pipeline that:
1. Downloads and explores the Ames Housing dataset
2. Cleans and engineers features from raw housing data
3. Trains **two different models** on the same data
4. Compares them side-by-side to understand the tradeoffs

The two models:
- **Linear Regression** — the classic baseline, fast and interpretable
- **XGBoost (Gradient Boosting)** — a powerful tree-based model that wins most regression competitions

---

## Project Structure
```
project3_house_price_prediction/
│
├── venv/                   # Virtual environment (not committed to git)
├── data/                   # Raw and processed data (not committed to git)
├── models/                 # Saved trained models (not committed to git)
│
├── 01_download_data.py     # Download the Ames Housing dataset
├── 02_preprocess.py        # Clean and engineer features
├── 03_train_linear.py      # Train the Linear Regression model
├── 04_train_xgboost.py     # Train the XGBoost model
├── 05_compare.py           # Compare both models side-by-side
│
├── .gitignore
└── README.md
```
---

## Libraries We'll Use

### `pandas`
**What it is:** A library for loading and manipulating data in table format (like a spreadsheet in Python).
**Why we need it:** The Ames Housing dataset has 79 features per house. Pandas lets us load it, inspect it, handle missing values, and engineer new features cleanly.

### `numpy`
**What it is:** A library for fast numerical computing — working with arrays and matrices of numbers.
**Why we need it:** Under the hood, almost every ML operation turns data into grids of numbers. NumPy is the foundation that makes that fast.

### `scikit-learn`
**What it is:** The go-to Python library for traditional machine learning.
**Why we need it:** We'll use it for Linear Regression, data splitting, feature scaling, and evaluation metrics like RMSE and R².

### `xgboost`
**What it is:** A highly optimized gradient boosting library that has won hundreds of ML competitions.
**Why we need it:** XGBoost is one of the most powerful and widely used models for structured tabular data like housing datasets. It builds an ensemble of decision trees that each correct the mistakes of the previous one.

### `matplotlib`
**What it is:** A library for creating charts and visualizations.
**Why we need it:** We'll plot predicted vs actual prices, feature importances, and a side-by-side model comparison chart.

### `seaborn`
**What it is:** A higher-level visualization library built on top of matplotlib.
**Why we need it:** We'll use it to create cleaner, more informative charts — like correlation heatmaps and distribution plots.

### `kagglehub`
**What it is:** A library for accessing datasets from Kaggle directly in Python.
**Why we need it:** The Ames Housing dataset is hosted on Kaggle. KaggleHub lets us download it cleanly with one function call.

### `joblib`
**What it is:** A library for saving and loading Python objects to disk efficiently.
**Why we need it:** We'll save our trained models so Script 05 can load them for comparison without retraining.

---

## Concepts We'll Learn

### Regression vs Classification
In our previous two projects we predicted categories — disease/no disease, positive/negative. Regression is different — we predict a **continuous number**. Instead of "is this tweet positive?" we ask "how much will this house sell for?" The evaluation metrics are completely different too.

### RMSE (Root Mean Squared Error)
The primary way we measure how wrong our predictions are. It calculates the average distance between our predicted prices and the actual sale prices in dollars. Lower is better. If RMSE is $20,000 it means our predictions are off by about $20,000 on average.

### R² (R-Squared)
A score between 0 and 1 that tells us how much of the variance in house prices our model explains. An R² of 0.90 means the model explains 90% of why prices differ between houses. Higher is better.

### Missing Value Handling
Real world datasets are messy — some houses are missing values for certain features. We'll learn strategies for dealing with this: filling with the median, filling with the most common value, or dropping the column entirely.

### Feature Engineering
Creating new useful features from existing ones. For example, combining "year built" and "year sold" to create "age of house at time of sale" — a more meaningful signal than either column alone.

### Log Transformation
House prices are skewed — most houses sell for $100k-$300k but a few sell for millions. This skew can confuse regression models. We'll apply a log transformation to the target variable to make the distribution more symmetric and the model more accurate.

### Linear Regression
The simplest regression model — it fits a straight line (or hyperplane in multiple dimensions) through the data. Every feature gets a coefficient that represents how much it affects the price. Completely interpretable — you can read exactly what the model learned.

### Gradient Boosting (XGBoost)
An ensemble method that builds hundreds of decision trees sequentially — each tree corrects the mistakes of the previous one. It's more powerful than linear regression but less interpretable. XGBoost is the most commonly used model in ML competitions involving tabular data.

### Feature Importance
XGBoost can tell us which features it relied on most when making predictions. Did square footage matter more than neighborhood? Did the number of bathrooms matter more than the garage size? Feature importance answers these questions.

### Overfitting
When a model memorizes the training data instead of learning general patterns. It scores perfectly on training data but poorly on new houses. We'll watch for this especially with XGBoost since powerful models are more prone to overfitting.

---

## Script-by-Script Breakdown

### `01_download_data.py`
**Purpose:** Download the Ames Housing dataset from Kaggle and save it to the `data/` folder.
The Ames Housing dataset contains 1,460 houses with 79 features each, collected from real sales in Ames, Iowa between 2006 and 2010. It's the standard dataset for learning regression in ML.

### `02_preprocess.py`
**Purpose:** Clean the raw data, handle missing values, engineer new features, and prepare it for both models.
This script handles the messiest part of the project — filling in missing values, converting categorical features (like neighborhood names) into numbers, creating new features, and applying a log transformation to the sale price. It saves clean train and test sets to disk.

### `03_train_linear.py`
**Purpose:** Train the first model — Linear Regression.
This script scales the features, trains a Linear Regression model, evaluates it using RMSE and R², and shows which features had the largest coefficients — revealing what the model weighted most heavily when predicting price.

### `04_train_xgboost.py`
**Purpose:** Train the second model — XGBoost Gradient Boosting.
This script trains an XGBoost model on the same data, evaluates it using the same RMSE and R² metrics, and plots feature importances — showing which of the 79 features XGBoost relied on most.

### `05_compare.py`
**Purpose:** Load both trained models and compare them directly.
This script runs both models on the same test set and produces a side-by-side comparison of RMSE and R², a bar chart, and a predicted vs actual price scatter plot for both models so you can visually see where each one goes wrong.

---

## What "Good" Looks Like

Since this is a regression task, we don't measure accuracy — we measure how close our predictions are to the real sale prices.

A naive baseline that always predicts the average house price would have an R² of 0.0 — it explains nothing. We're aiming for:

- **Linear Regression:** R² ~0.85, RMSE ~$25,000–$30,000
- **XGBoost:** R² ~0.90+, RMSE ~$18,000–$22,000

The gap between the two models will be more visible here than it was in Project 2 — XGBoost is significantly better suited to structured tabular data like this than it was for tweets.