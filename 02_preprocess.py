# pandas loads and manipulates our data in table format
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# sklearn's train_test_split divides our data into a training set and a test set
# Analogy: the training set is the textbook, the test set is the final exam
from sklearn.model_selection import train_test_split


# This script is the data janitor of the project
# Raw housing data is messy — missing values, text categories, and skewed prices
# This script loads the raw data, fills in missing values, converts text categories
# into numbers, engineers new features, and saves clean train and test sets to disk
# Analogy: think of this script as a contractor who inspects a house before listing it —
# patching holes, fixing broken things, and making everything presentable
# before the real work (modeling) begins


def load_data():
    # This function's job is to load the raw training CSV into a DataFrame
    # We only use train.csv because it has the sale prices we need to learn from
    # The Kaggle test.csv has no prices so it's useless for our purposes
    # Analogy: like receiving a graded exam (train.csv) vs a blank exam (test.csv) —
    # you can only learn from the one that has the answers

    print("Loading raw data...")
    df = pd.read_csv("data/train.csv")

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    # This function's job is to fill in missing values throughout the dataset
    # Missing values appear as NaN (Not a Number) in pandas
    # Models can't learn from blank cells — we need to fill them with something sensible
    # Analogy: if a house doesn't have a "pool quality" rating, it probably has no pool —
    # so we fill it with "None" rather than leaving it blank

    print("\nHandling missing values...")

    # These columns are missing because the house simply doesn't have that feature
    # For example NaN in PoolQC means no pool, not that the data is missing
    # We fill these with the string "None" to make that explicit
    # Analogy: like writing "N/A" on a form field that doesn't apply to you
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_cols:
        df[col] = df[col].fillna("None")

    # These numeric columns are missing because the feature doesn't exist
    # For example NaN in GarageArea means no garage — so we fill with 0
    # Analogy: if a house has no garage, its garage area is 0, not unknown
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    # LotFrontage (linear feet of street connected to the property) is missing
    # for many houses — we fill with the median value for that neighborhood
    # Houses in the same neighborhood tend to have similar lot frontages
    # Analogy: if we don't know your street frontage, we assume it's typical
    # for your neighborhood
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # For any remaining categorical columns with missing values,
    # fill with the most common value in that column
    # Analogy: if we don't know what something is, assume it's the most typical option
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # For any remaining numeric columns with missing values,
    # fill with the median value of that column
    # We use median instead of mean because it's less sensitive to outliers
    # Analogy: if we don't know a measurement, assume it's the middle-of-the-road value
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    # This function's job is to create new features from existing ones
    # Sometimes combining two columns tells the model something more useful
    # than either column alone
    # Analogy: knowing someone's height and weight separately is less useful
    # than knowing their BMI — a single number that combines both meaningfully

    print("Engineering new features...")

    # Total square footage — combines all living areas into one number
    # A buyer cares about total usable space, not which floor it's on
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Total number of bathrooms — combines full and half baths across all floors
    # A half bath (toilet and sink only) counts as 0.5
    # Analogy: adding up all the bathrooms in a house regardless of which floor they're on
    df["TotalBaths"] = (
        df["FullBath"] +
        df["HalfBath"] * 0.5 +
        df["BsmtFullBath"] +
        df["BsmtHalfBath"] * 0.5
    )

    # Age of the house at the time of sale
    # A house built in 1950 and sold in 2008 is 58 years old
    # This is more meaningful than raw year built
    # Analogy: knowing someone's age is more useful than knowing their birth year
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Whether the house was remodelled — 1 if yes, 0 if no
    # If YearRemodAdd equals YearBuilt, the house was never remodelled
    # Analogy: like a flag on a listing that says "recently renovated"
    df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Years since remodel at time of sale
    # A remodel done 2 years ago is more valuable than one done 20 years ago
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    print(f"New features added: TotalSF, TotalBaths, HouseAge, Remodeled, YearsSinceRemodel")
    return df


def encode_categories(df):
    # This function's job is to convert text categories into numbers
    # Models can only work with numbers — they can't read strings like "Excellent" or "Good"
    # pd.get_dummies() converts each unique category value into its own column of 0s and 1s
    # This is called one-hot encoding
    # Analogy: instead of one column that says "neighborhood = OldTown",
    # you get separate columns: "is_OldTown = 1", "is_NridgHt = 0", "is_Sawyer = 0" etc.
    # The model can then learn that being in OldTown affects price differently than NridgHt

    print("Encoding categorical features...")

    # select_dtypes(include=["object"]) finds all columns that contain text
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"Encoding {len(cat_cols)} categorical columns")

    # pd.get_dummies() creates binary columns for each category
    # drop_first=True drops the first category to avoid redundancy
    # If a house is not OldTown and not NridgHt, we can infer it's Sawyer —
    # so we don't need a column for every single category
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"Dataset shape after encoding: {df.shape}")
    return df


def transform_target(df):
    # This function's job is to apply a log transformation to the sale price
    # House prices are heavily skewed — most are $100k-$300k but some are over $700k
    # This skew can confuse regression models because the outliers pull the predictions
    # np.log1p() applies log(1 + x) which compresses the scale of large values
    # and makes the distribution more symmetric and easier for models to learn from
    # Analogy: instead of measuring distances in miles (where 1 mile vs 1000 miles
    # is a huge difference), you measure in a compressed scale where the differences
    # between large values are proportionally smaller

    print("Applying log transformation to SalePrice...")

    # Store the original prices for reference
    print(f"Original price range: ${df['SalePrice'].min():,.0f} — ${df['SalePrice'].max():,.0f}")

    # Apply log1p transformation to SalePrice
    df["SalePrice"] = np.log1p(df["SalePrice"])

    print(f"Log-transformed price range: {df['SalePrice'].min():.2f} — {df['SalePrice'].max():.2f}")
    return df


def split_and_save(df):
    # This function's job is to separate features from the target,
    # split into train and test sets, and save everything to disk
    # Scripts 03 and 04 will load these files directly
    # Analogy: the prep cook finishing all the chopping and portioning,
    # then putting everything in labeled containers for the chef

    print("\nSplitting into train and test sets...")

    # X = all columns except SalePrice (the features the model learns from)
    # y = just the SalePrice column (what the model is trying to predict)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # train_test_split splits the data into 80% training and 20% test
    # random_state=42 ensures the same split every time we run this
    # Analogy: shuffling a deck and dealing 80% to the training pile, 20% to the test pile
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save all four pieces to disk as CSV files
    # We save X and y separately because scripts 03 and 04 need them that way
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print(f"Train set: {X_train.shape[0]} houses, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} houses, {X_test.shape[1]} features")
    print("Saved to data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")


# This block only runs if you execute this file directly (python 02_preprocess.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    df = load_data()
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = transform_target(df)
    df = encode_categories(df)
    split_and_save(df)