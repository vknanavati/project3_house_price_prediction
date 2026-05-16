## Linear Regression

### What is Linear Regression?

Linear Regression finds the best straight line (or flat surface) through a set of data points. It learns a **coefficient** (weight) for every feature, and then predicts the output by multiplying each feature by its coefficient and adding them all up.

For our housing data it looks like this:
Predicted Price =
(TotalSF × 0.42) +
(HouseAge × -0.18) +
(TotalBaths × 0.15) +
(Neighborhood_OldTown × -0.09) +
... (one term for every feature)

Every feature gets its own coefficient. Positive coefficient = pushes price up. Negative coefficient = pushes price down. The model's job during training is to find the exact set of coefficients that minimizes the gap between its predictions and the real sale prices.

**Analogy:** Imagine a real estate appraiser who has a pricing formula written on a notepad. Every feature of the house adds or subtracts a certain dollar amount. "Each extra bathroom adds $8,000. Each year of age subtracts $500. Being in the best neighborhood adds $20,000." Linear Regression is the machine that figures out that formula automatically by studying thousands of past sales.

---

### How does it actually find those coefficients?

It uses a technique called **Ordinary Least Squares (OLS)**. Here's the idea:

1. Start with a guess for every coefficient
2. Make predictions using those coefficients
3. Calculate the error — how far off were the predictions?
4. Adjust the coefficients to reduce the error
5. Repeat until the error can't get any smaller

The "least squares" part means it minimizes the **sum of squared errors** — squaring the errors penalizes big mistakes more than small ones, which pushes the model to avoid wildly wrong predictions.

**Analogy:** Imagine you're trying to draw the best straight line through a scatter plot of points. You try different angles and positions, measuring how far each point is from the line each time. You keep adjusting until the total distance from all points to the line is as small as possible. That's exactly what OLS does mathematically.

---

### Why is it good for this dataset?

House prices are a natural fit for Linear Regression for a few reasons:

- **The relationship is roughly linear** — more square footage generally means higher price in a predictable, proportional way
- **The features are meaningful numbers** — square footage, age, number of bathrooms all have clear numerical relationships with price
- **It's interpretable** — we can read the coefficients and understand exactly what the model learned
- **It's a strong baseline** — if a fancier model like XGBoost can't beat Linear Regression by much, it tells us the problem isn't that complex

---

### Could we have used it for the previous projects?

Technically yes — but it would have been a bad fit:

**Project 1 (Heart Disease)** — this was binary classification (yes/no). Linear Regression predicts continuous numbers, not categories. You'd need to awkwardly threshold the output (above 0.5 = disease, below = no disease) which is essentially reinventing Logistic Regression but worse. Logistic Regression was the right tool.

**Project 2 (Sentiment Analysis)** — Linear Regression can technically do classification but it struggles badly with high-dimensional sparse data like TF-IDF vectors. It would have performed noticeably worse than Logistic Regression.

---

### The key distinction to remember

| Problem type | Output | Right tool |
|---|---|---|
| Binary classification | Yes or No | Logistic Regression |
| Multiclass classification | Category A, B, or C | Random Forest, etc. |
| Regression | A number | Linear Regression, XGBoost |

Linear Regression is specifically designed for when the answer is a number on a continuous scale — like price, temperature, age, or salary. That's exactly what we have here.

# Here's exactly what the Linear Regression model will show us:

## 1. Train RMSE vs Test RMSE
How wrong the model's predictions are on average, in log price space. We get both training and test scores so we can check for overfitting — if training RMSE is much lower than test RMSE, the model memorized rather than learned.

**Root Mean Squared Error**

Each word describes one step of the calculation:

Squared — square every error so negatives become positive
Mean — average all the squared errors
Root — take the square root to bring the units back to dollars
Error — the difference between predicted and actual
The steps are actually done in reverse order of the name — you calculate the Error first, then Square it, then take the Mean, then the Root. But the name reads from the outside in: Root of the Mean of the Squared Errors.

Great question — let's build it up step by step.

---

**Start with the basic idea:**

After the model makes predictions, we want to know: on average, how far off were those predictions?

For each house we can calculate the **error** — the difference between the predicted price and the actual price:

```
Error = Predicted Price - Actual Price
```

If we predicted $200,000 and the actual price was $220,000, the error is -$20,000.

---

**Why not just average the errors?**

Because positive and negative errors cancel each other out. If one prediction is $20,000 too high and another is $20,000 too low, the average error is $0 — which makes it look like the model is perfect when it clearly isn't.

---

**So we square the errors first:**

Squaring does two things:
1. Makes all errors positive so they can't cancel out
2. Penalizes big mistakes more than small ones — a $40,000 error is 4x worse than a $20,000 error, not just 2x

```
Squared Error = (Predicted - Actual)²
```

---

**Then we average the squared errors:**

This gives us the **Mean Squared Error (MSE)**:

```
MSE = average of all squared errors
```

---

**Then we take the square root:**

Squaring the errors changed the units — if prices are in dollars, squared errors are in dollars². Taking the square root brings us back to dollars:

```
RMSE = √MSE
```

---

**So RMSE in plain English:**

RMSE is the typical size of the model's mistakes, in the same units as the thing you're predicting. An RMSE of $25,000 means the model's predictions are typically about $25,000 off from the real price.

**Analogy:** Imagine you're estimating how long your commute will take every day for a month. Some days you're 5 minutes off, some days 20 minutes off. RMSE gives you one number that summarizes your typical estimation error — accounting for the fact that being 20 minutes wrong is much worse than being 5 minutes wrong.

---

**Lower RMSE = better model.** That's it.

## 2. Train R² vs Test R²
How much of the variation in house prices the model explains. A score of 0.85 means 85% of why prices differ between houses is captured by our model. The remaining 15% is noise or features we don't have.

Start with the question R² is answering:
Houses have different prices. A tiny 1920s bungalow sells for $120,000. A huge modern house sells for $450,000. Why are they different? What explains that gap?
R² answers: how much of that difference can our model explain?

Great — let me break that down more clearly.

---

**Start with the question R² is answering:**

Houses have different prices. A tiny 1920s bungalow sells for $120,000. A huge modern house sells for $450,000. **Why are they different?** What explains that gap?

R² answers: **how much of that difference can our model explain?**

---

**What R² = 0.85 actually means:**

It means 85% of the reasons why house prices vary from each other are captured by the features in our model — things like square footage, neighborhood, number of bathrooms, age of the house.

The remaining 15% is variation the model can't explain — things like:
- The seller was in a hurry and accepted a lower offer
- The buyer fell in love with the house and overpaid
- The house had a beautiful view that wasn't in the dataset
- Pure luck or timing of the market

---

**Analogy:**

Imagine you're trying to explain why some students get higher grades than others. You collect data on study hours, attendance, and GPA. Your model gets R² = 0.85.

That means 85% of why grades differ between students is explained by study hours, attendance, and GPA. The other 15% is stuff you can't measure — natural talent, test anxiety, whether they slept well the night before.

---

**The simple version:**

- **R² = 1.0** — the model perfectly explains every difference in price
- **R² = 0.85** — the model explains most of it, misses some
- **R² = 0.0** — the model explains nothing, no better than guessing the average price every time
- **R² < 0** — the model is actually worse than just guessing the average

Higher is better. For house prices, 0.85+ is considered strong.


## 3. Approximate dollar error
We convert the RMSE back from log space into actual dollars so it's meaningful. Something like "$25,000 average error" is much easier to understand than "0.14 log error."

## 4. Top 15 most influential features**
The 15 features with the largest coefficients — both positive and negative. This tells us what the model actually learned. We'd expect to see things like:
- `TotalSF` as a strong positive — more space = higher price
- `HouseAge` as a strong negative — older houses = lower price
- Certain neighborhoods as strong positives or negatives

## 5. A scatter plot
Every house in the test set plotted as a dot — actual price on the x axis, predicted price on the y axis. A perfect model would have all dots on a straight diagonal line. The more scattered the dots, the more the model is struggling.

---

**Analogy:** It's like getting back a graded exam with your score (RMSE, R²), the dollar value of your mistakes, which questions you weighted most heavily (top features), and a visual map of where you went right and wrong (scatter plot).

__

Evaluation result:
Excellent results — this is a big improvement over Linear Regression. Let's break everything down:

---

**The training progress:**
```
[0]   validation_0-rmse: 0.37517
[100] validation_0-rmse: 0.07955
[200] validation_0-rmse: 0.05947
...
[999] validation_0-rmse: 0.01240
```
You can watch the model getting smarter with every 100 trees. It started with an RMSE of 0.375 and by tree 1000 had dropped all the way to 0.012 on the training set. That's the sequential correction process working in real time.

---

**Model Evaluation:**
```
Train RMSE: 0.0124  |  Test RMSE: 0.1339
Train R²:   0.9990  |  Test R²:   0.9039
```
Train R² of 0.9990 is essentially perfect — the model memorized the training data almost completely. But the test R² of 0.9039 is still excellent — it explains 90% of price variation on houses it has never seen. The gap between train and test is still there (overfitting) but the test score is meaningfully better than Linear Regression's 0.8394.

**Dollar error: $23,270** — on average the model's predictions are about $23,000 off from the real price. That's actually working correctly unlike Script 03.

---

**Feature importance — this is the interesting part:**

Compare XGBoost's top features to Linear Regression's:

| XGBoost | Linear Regression |
|---|---|
| OverallQual | RoofMatl_CompShg |
| TotalSF | PoolArea |
| KitchenAbvGr | PoolQC_None |

XGBoost's top features make complete intuitive sense:
- **OverallQual** — the overall quality rating of the house is the single strongest predictor of price. Makes total sense
- **TotalSF** — total square footage is the second most important. Also makes total sense
- **CentralAir_Y** — whether the house has central air conditioning. Very reasonable

Linear Regression was being thrown off by rare one-hot encoded categories like roof material and pool features. XGBoost correctly identified that overall quality and square footage matter most — exactly what any real estate agent would tell you.

**YearsSinceRemodel** appearing in the top 15 is also satisfying — that was one of the new features we engineered ourselves in Script 02. The model found it useful.

---

**The scatter plot:**
The green dots in the XGBoost chart cluster more tightly around the red line than the blue dots in the Linear Regression chart — especially in the middle price range. XGBoost is visibly more accurate.

---

**Side by side comparison so far:**

| Model | Test RMSE | Test R² |
|---|---|---|
| Linear Regression | 0.1731 | 0.8394 |
| XGBoost | 0.1339 | 0.9039 |

