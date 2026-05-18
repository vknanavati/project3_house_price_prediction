## Project 3 Complete — Final Summary

### What we built
A full regression pipeline that takes raw housing data, cleans and engineers features, trains two different models, and compares them side by side to predict house sale prices.

---

### Final results

| Model | Test RMSE | Test R² | Dollar Error |
|---|---|---|---|
| Naive baseline (always predict average) | — | 0.00 | — |
| Linear Regression | 0.1731 | 0.8394 | ~$30,686 |
| XGBoost | 0.1339 | 0.9039 | ~$23,270 |

---

### What each script did

**`01_download_data.py`** — Downloaded the Ames Housing dataset from Kaggle using kagglehub and saved it to the `data/` folder. Required Kaggle API authentication and accepting competition rules.

**`02_preprocess.py`** — Handled missing values, engineered 5 new features (TotalSF, TotalBaths, HouseAge, Remodeled, YearsSinceRemodel), applied a log transformation to SalePrice, one-hot encoded 43 categorical columns, and split into 1,168 training and 292 test houses.

**`03_train_linear.py`** — Scaled features with StandardScaler, trained a Linear Regression model, achieved R² of 0.8394 and dollar error of ~$30,686 on the test set. Revealed that roof material and pool features dominated the coefficients — a sign of overfitting on rare categories.

**`04_train_xgboost.py`** — Trained an XGBoost model with 1000 trees, achieved R² of 0.9039 and dollar error of ~$23,270. Correctly identified OverallQual and TotalSF as the two most important features — exactly what any real estate agent would tell you.

**`05_compare.py`** — Loaded both models, ran them on the same 292 test houses, generated side by side bar charts and scatter plots, and printed a final summary table.

---

### Key lessons learned

**1. Regression is fundamentally different from classification**
Instead of predicting a category we predicted a continuous number. This changes everything — the metrics (RMSE, R²), the target transformation (log), and how we interpret results (dollar error instead of accuracy percentage).

**2. Feature engineering matters**
We created 5 new features from existing columns. YearsSinceRemodel appeared in XGBoost's top 15 most important features — the model found our engineered feature genuinely useful.

**3. Log transformation is a powerful trick**
House prices ranged from $34,900 to $755,000 — a 20x range that would confuse regression models. After log transformation the range compressed to 10.46 to 13.53. This made both models significantly more accurate.

**4. XGBoost beats Linear Regression on tabular data**
Unlike Project 2 where TF-IDF and DAN finished less than 1% apart, here XGBoost won by a clear margin — 6% better R² and $7,400 less dollar error per house. Structured tabular data with complex feature interactions is exactly where XGBoost shines.

**5. More complex models overfit more**
XGBoost had a Train R² of 0.9990 but Test R² of 0.9039 — nearly perfect on training data but less so on new houses. Linear Regression showed the same pattern. Overfitting is something to always watch for and can be reduced with techniques like cross-validation and regularization.

**6. Feature importance reveals what models actually learned**
Linear Regression was fooled by rare one-hot encoded categories like roof material. XGBoost correctly identified overall quality and total square footage as the dominant signals. The same data, the same features — but very different understanding of what matters.

---

### New concepts learned this project

- Regression vs classification
- RMSE and R² as evaluation metrics
- Missing value handling strategies
- Feature engineering
- Log transformation of skewed targets
- One-hot encoding of categorical variables
- StandardScaler and why Linear Regression needs it but XGBoost doesn't
- Gradient boosting and how XGBoost builds trees sequentially
- Feature importance scores
- Hyperparameters and how they control model learning
- Overfitting detection by comparing train vs test scores
- Kaggle API authentication