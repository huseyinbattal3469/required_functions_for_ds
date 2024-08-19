# `required_functions.py`

This Python script provides a comprehensive collection of functions and utilities designed to streamline various stages of the data science workflow, including data preprocessing, exploratory data analysis, model training, and evaluation. The script leverages several popular libraries, such as `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib`, among others.

## Table of Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

To use the functions in this script, ensure that you have the required libraries installed. You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib xgboost lightgbm catboost
```

## Features

### 1. Data Preprocessing
- **Imputing missing values**: Functions to handle missing data using various imputation techniques (`KNNImputer`, `SimpleImputer`, etc.).
- **Encoding**: Functions to encode categorical variables using methods like `LabelEncoder`, `OneHotEncoder`, etc.
- **Scaling**: Functions to scale numerical data using techniques like `MinMaxScaler`, `StandardScaler`, and `RobustScaler`.

### 2. Exploratory Data Analysis (EDA)
- **Dataframe Overview**: Functions to get a quick summary of the dataset including shape, data types, and missing values.
- **Column Classification**: Functions to classify columns into categorical, numerical, and categorical but cardinal.
- **Summarization**: Functions to generate summaries for categorical and numerical columns.
- **Correlation Analysis**: Functions to visualize and analyze correlations between features.

### 3. Model Training and Evaluation
- **Supervised Learning**: Pre-defined functions to train and evaluate models using various algorithms such as Random Forest, Gradient Boosting, Logistic Regression, etc.
- **Unsupervised Learning**: Utilities for clustering and dimensionality reduction using `KMeans`, `PCA`, `AgglomerativeClustering`, etc.
- **Model Selection**: Tools for cross-validation, hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`), and ensemble methods.

### 4. Handling Outliers
- **Outlier Detection and Removal**: Functions to detect and handle outliers using statistical methods like IQR.

### 5. Missing Value Analysis
- **Missing Value Analysis**: Functions to summarize and handle missing data, including quick imputation methods.

## Usage

To use these functions in your project, simply import the `required_functions.py` file into your Python script:

```python
from required_functions import *
```

You can then call any of the provided functions directly:

```python
df = pd.read_csv('your_data.csv')
check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
