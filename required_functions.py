import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

################# Imputing
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
# from sklearn.experimental import enable_iterative_imputer , IterativeImputer

################# Unsupervised Learning
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

################# Supervised Learning
from sklearn.metrics import get_scorer_names
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, VotingClassifier, VotingRegressor, AdaBoostClassifier,
                              AdaBoostRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              BaggingClassifier, BaggingRegressor)

from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
                                  SGDClassifier)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR

from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor, NearestCentroid, RadiusNeighborsClassifier,
                               RadiusNeighborsRegressor)

from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor,
                          export_graphviz, export_text)

from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
############# Scoring
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, classification_report, confusion_matrix

############# Train - Test - Cross Validation - Grid Search
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV

############# Encoding
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

############# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)


################ Etc.

################ Etc.

################ Explonatory Data Analysis

def check_df(dataframe: pd.DataFrame, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)


def grab_col_names(dataframe: pd.DataFrame, cat_th=10, car_th=20):
    """

    :param dataframe:
        pandas.DataFrame object
    :param cat_th:
        default: 10
        integer
    :param car_th:
        default: 20
        integer
    :return:
        cat_cols, num_cols, cat_but_cardinal_cols
        list, list, list
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations (Gözlem Birimleri): {dataframe.shape[0]}")
    print(f"Variables (Değişkenler): {dataframe.shape[1]}")
    print(f"cat_cols (kategorik_değişkenler): {len(cat_cols)} - {cat_cols}")
    print(f"num_cols (numerik_değişkenler): {len(num_cols)} - {num_cols}")
    print(f"cat_but_car (kategorik_ama_kardinal): {len(cat_but_car)} - {cat_but_car}")
    print(f"num_but_cat (numerik_ama_kategorik): {len(num_but_cat)} - {num_but_cat}")

    return cat_cols, num_cols, cat_but_car


################ Explonatory Data Analysis

################ Summaries

def cat_summary(dataframe: pd.DataFrame, categorical_col: str, plot=False):
    """
    :param dataframe:
        pandas.DataFrame object
    :param categorical_col:
        string
    :param plot:
        default: False
        boolean
    :return:
        None
    """
    result_df = pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                              "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)})
    result_null_df = 100 * dataframe[categorical_col].isnull().sum() / len(dataframe)
    print(result_df)
    print(
        f"Not Null Count: {dataframe[categorical_col].notnull().sum()}, Perc(%): {round(result_df['Ratio'].sum(), 2)}")
    print(f"Null Count: {dataframe[categorical_col].isnull().sum()}, Perc(%): {round(result_null_df, 2)}")
    print("#####################################")
    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe: pd.DataFrame, numerical_col: str, plot=False):
    """
    :param dataframe:
        pandas.DataFrame object
    :param numerical_col:
        string
    :param plot:
        default: False
        boolean
    :return:
        None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe: pd.DataFrame, target: str, categorical_col: str, plot=False):
    """
    :param dataframe: pandas.DataFrame
    :param target: string
    :param categorical_col: string
    :param plot: boolean
        False
    """
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)


def target_summary_with_num(dataframe: pd.DataFrame, target: str, numerical_col: str, plot=False, method: str = "class",
                            hue=None):
    """
    :param dataframe: pandas.DataFrame
    :param target: string
    :param numerical_col: string
    :param plot: boolean
        False
    :param method: string
        'class' or 'regr'
    :param hue: [optional] | string
        None
    """
    if method == "regr" and plot:
        # plt.figure(figsize=(10, 6))
        sns.lmplot(x=target, y=numerical_col, data=dataframe, hue=hue, height=6, aspect=1.5)
        plt.show(block=True)
    elif method == "class":
        print(pd.DataFrame({numerical_col + '_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
        if plot:
            sns.barplot(x=target, y=numerical_col, data=dataframe)
            plt.show(block=True)


################ Summaries

################ Correlation Summaries

def correlation_matrix(dataframe: pd.DataFrame, cols: list):
    """
    :param dataframe: pandas.Dataframe
    :param cols: list
    """
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show(block=True)


def df_corr(dataframe: pd.DataFrame, annot=True):
    """
    :param dataframe: pandas.DataFrame
    :param annot: boolean
        True
    """
    sns.heatmap(dataframe.corr(), annot=annot, linewidths=.2, cmap='Reds', square=True)
    plt.show(block=True)


def high_correlated_cols(dataframe: pd.DataFrame, head=10):
    """
    :param dataframe: pandas.DataFrame
    :param head: int
        10
    :return: corr_cols
        list
    """
    corr_matrix = dataframe.corr().abs()
    corr_cols = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                   .astype(bool)).stack().sort_values(ascending=False)).head(head)
    return corr_cols


################ Correlation Summaries

################ Outliers
def outlier_thresholds(dataframe: pd.DataFrame, value: str, q1=0.25, q3=0.75):
    """
    :param dataframe: pandas.DataFrame
    :param value: str
    :param q1: float
        0.25
    :param q3: float
        0.75
    :return: up_limit, low_limit
        float, float
    """
    q1 = dataframe[value].quantile(q1)
    q3 = dataframe[value].quantile(q3)
    iqr = q3 - q1
    up = q3 + iqr * 1.5
    low = q1 - iqr * 1.5

    return up, low


def check_outlier(dataframe: pd.DataFrame, value: str, q1=0.25, q3=0.75):
    """
    :param dataframe: pandas.DataFrame
    :param value: string
    :param q1: float
        0.25
    :param q3: float
        0.75
    :return: boolean
    """
    up, low = outlier_thresholds(dataframe, value, q1=q1, q3=q3)
    if dataframe[(dataframe[value] > up) | (dataframe[value] < low)].any(axis=None):
        return True
    else:
        return False


def show_outliers(dataframe: pd.DataFrame, col_name: str, q1=0.25, q3=0.75, index=False):
    """
    :param dataframe: pandas.DataFrame
    :param col_name: string
    :param q1: float
        0.25
    :param q3: float
        0.75
    :param index: boolean
        False
    :return: list
    """
    up, low = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].head())
    else:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)])

    if index:
        return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].index


def remove_outlier(dataframe: pd.DataFrame, col_name: str, q1=0.25, q3=0.75):
    """
    :param dataframe: pandas.DataFrame
    :param col_name: string
    :param q1: float
        0.25
    :param q3: float
        0.75
    :return: pandas.DataFrame
    """
    up, low = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    return df_without_outliers


def replace_with_thresholds(dataframe: pd.DataFrame, variable: str, q1=0.25, q3=0.75, low_threshold=False):
    """
    :param dataframe: pandas.DataFrame
    :param variable: str
    :param q1: float
        0.25
    :param q3: float
        0.75
    :param low_threshold: boolean
        False
    """
    up_limit, low_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    if low_threshold:
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit


############# Outliers

############# Missing Values

def missing_values_table(dataframe: pd.DataFrame, na_name=False):
    """
    :param dataframe: pandas.DataFrame
    :param na_name: boolean
        False
    :return: list
    """
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_cols


def missing_vs_target(dataframe: pd.DataFrame, target: str, na_columns: list):
    """
    :param dataframe: pandas.DataFrame
    :param target: string
    :param na_columns: list
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "TARGET_MEDIAN": temp_df.groupby(col)[target].median(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n")


def quick_missing_imp(dataframe: pd.DataFrame, num_method="median", cat_length=20, dependant="SalePrice"):
    """
    :param dataframe: pandas.DataFrame
    :param num_method: string
        median - mode - etc.
    :param cat_length: int
        20
    :param dependant: string
    :return: pandas.DataFrame
    """
    variables_with_na = [col for col in dataframe.columns if
                         dataframe[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir
    temp_target = dataframe[dependant]
    print("# BEFORE")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")
    dataframe = dataframe.apply(
        lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
        axis=0)
    if num_method == "mean":
        dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
    dataframe[dependant] = temp_target
    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")
    return dataframe


############# Missing Values

############# Scaling the Numbers

def robust_scaler(dataframe: pd.DataFrame, num_col: str):
    """
    :param dataframe: pandas.DataFrame
    :param num_col: string
    :return: pandas.DataFrame
    """
    dataframe[num_col] = RobustScaler().fit_transform(dataframe[[num_col]])
    return dataframe


def min_max_scaler(dataframe: pd.DataFrame, num_col: str):
    """
    :param dataframe: pandas.DataFrame
    :param num_col: string
    :return: pandas.DataFrame
    """
    dataframe[num_col] = MinMaxScaler().fit_transform(dataframe[[num_col]])
    return dataframe


def standard_scaler(dataframe: pd.DataFrame, num_col: str):
    """
    :param dataframe: pandas.DataFrame
    :param num_col: string
    :return: pandas.DataFrame
    """
    dataframe[num_col] = StandardScaler().fit_transform(dataframe[[num_col]])
    return dataframe


############# Scaling the Numbers

############# Encoding the Categorical Variables

def rare_analyser(dataframe: pd.DataFrame, target: str, cat_cols: list):
    """
    :param dataframe: pandas.DataFrame
    :param target: : string
    :param cat_cols: list
    """
    for col in cat_cols:
        print("-"*20)
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n")
        print("-"*20)

def rare_encoder(dataframe: pd.DataFrame, rare_perc):
    """
    :param dataframe: pandas.DataFrame
    :param rare_perc: float - int
    :return: pandas.DataFrame
    """
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and
                    (temp_df.value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def label_encoder(dataframe: pd.DataFrame, col: str):
    """
    Genellikle binary sütunlar için kullanılır ama çoklu sınıflı bağımlı değişkenler için de kullanılır.

    :param dataframe: pandas.DataFrame
    :param col: string
    :return: pandas.DataFrame
    """
    label_encoder = LabelEncoder()
    dataframe[col] = label_encoder.fit_transform(dataframe[col])
    return dataframe


def one_hot_encoder(dataframe: pd.DataFrame, categorical_cols: list, drop_first=True):
    """
    :param dataframe: pandas.DataFrame
    :param categorical_cols: list
    :param drop_first: boolean
        True
    :return: pandas.DataFrame
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


############# Encoding the Categorical Variables


############# Machine Learning Pipeline

def roc_auc_score_multiclass(y_true, y_pred, average="macro"):
    """
    :param y_true: pandas.Series
    :param y_pred: pandas.Series
    :param average: string
        macro - {‘micro’, ‘macro’, ‘samples’, ‘weighted’}

    :return: dict
    """
    # creating a set of all the unique classes using the actual class list
    unique_class = set(y_true)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in y_true]
        new_pred_class = [0 if x in other_class else 1 for x in y_pred]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def base_models(X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], is_classifier=True,
                random_state=40):
    """
    :param X:
    :param y:
    :param cv: int
        5
    :param scoring: list
         ["accuracy", "precision", "recall", "f1", "roc_auc"]
         ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
         https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    :param is_classifier: boolean
        True
    :param random_state: int
        40
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    print("Base Models....")
    if is_classifier:
        models = {
            # Naive Bayes
            "GNB": GaussianNB(),
            # "MNB":MultinomialNB(), Metin saymak için ideal (?)
            # "CNB":ComplementNB(),
            "BNB": BernoulliNB(),
            "CATNB": CategoricalNB(),
            # Gaussian Process
            # "GPC": GaussianProcessClassifier(),
            # Linear Model
            "LR": LogisticRegression(),
            "SGD": SGDClassifier(),
            "RID": Ridge(),  # Regresyon Modeli - sürekli sayılar için
            "LAS": Lasso(),  # Regresyon Modeli
            "ENET": ElasticNet(),  # Regresyon Modeli
            # Support Vector Machines
            "SVC": SVC(),
            # "NUSVC": NuSVC(),
            "LSVC": LinearSVC(),
            # Neighbors
            "KNN": KNeighborsClassifier(),
            # "NECEN": NearestCentroid(),
            "RANEC": RadiusNeighborsClassifier(),
            # Tree
            "CART": DecisionTreeClassifier(),
            "EXTR": ExtraTreeClassifier(),
            # Ensemble
            "RF": RandomForestClassifier(),
            "BAG": BaggingClassifier(),
            "GBM": GradientBoostingClassifier(),
            "HIST": HistGradientBoostingClassifier(),
            "Adaboost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, device="cuda"),
            "LightGBM": LGBMClassifier(verbose=-1),
            "Catboost": CatBoostClassifier(verbose=False, task_type="GPU", devices='0:1'),
            "MLP": MLPClassifier(),
             

        }
    else:
        models = {
            # Naive Bayes
            # "GNB": GaussianNB(),
            # "MNB": MultinomialNB(),
            # "CNB": ComplementNB(),
            # "BNB": BernoulliNB(),
            # "CATNB": CategoricalNB(),
            # Gaussian Process
            "GPR": GaussianProcessRegressor(),
            # Linear Model
            "LiR": LinearRegression(),
            # "LR": LogisticRegression(),
            "SGD": SGDRegressor(),
            "RID": Ridge(),
            "LAS": Lasso(),
            "ENET": ElasticNet(),
            # Support Vector Machines
            "SVR": SVR(),
            "NUSVR": NuSVR(),
            "LSVR": LinearSVR(),
            # Neighbors
            "KNN": KNeighborsRegressor(),
            # "NECEN": NearestCentroid(),
            # "RANER": RadiusNeighborsRegressor(),
            # Tree
            "CART": DecisionTreeRegressor(),
            "EXTR": ExtraTreeRegressor(),
            # Ensemble
            "RF": RandomForestRegressor(),
            "BAG": BaggingRegressor(),
            "GBM": GradientBoostingRegressor(),
            "HIST": HistGradientBoostingRegressor(),
            "Adaboost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor(use_label_encoder=False, device="cuda"),
            "LightGBM": LGBMRegressor(verbose=-1),
            "Catboost": CatBoostRegressor(verbose=False, task_type="GPU", devices='0:1'),
            "MLP":MLPRegressor()
        }

    for name, model in models.items():
        print(f"################## {name} ################## ")
        try:
            for score_param in scoring:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=random_state)
                cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=score_param, error_score="raise")
                print(f"{score_param}: {abs(round(cv_results['test_score'].mean(), 4))} ({name}) ")
        except ValueError as e:
            print("Fit process has failed. There is might be NaN values...")
            # print(e)
            continue
        print()


def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc", is_classifier=True, is_grid_search=True,
                                random_state=40):
    """
    :param X:
    :param y:
    :param cv:
    :param scoring: string
        roc_auc
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    :param is_classifier: boolean
        True
    :param is_grid_search: boolean
        True
    :param random_state: int
        40
    :return: dict
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    print("Hyperparameter Optimization.....")
    # Naive Bayes
    gnb_params = {
        # "priors": [None, [0.2, 0.8]]
        "var_smoothing": [1e-9, 1e-10]
    }

    mnb_params = {
        "alpha": [1.0, 0.5, 0.1],
    }

    cnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "class_prior": [None, 0.2, 0.8]
    }

    bnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "binarize": [0.0, 0.5, 0.8]
    }

    catnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "class_prior": [None, 0.2, 0.8]
    }
    gp_params = {
        "max_iter_predict": [50, 100, 200, 500]
    }

    lir_params = {
        "fit_intercept": [True, False]
    }

    lr_params = {
        "max_iter": [1000, 2000],
        "penalty": ['l2', 'l1', 'elasticnet'],
        "C": [0.1, 1.0, 1.5]
    }

    sgd_params = {
        "alpha": [0.0001, 0.0001, 0.01],
        "penalty": ['l2', 'l1', 'elasticnet', None],
        "learning_rate": ['constant', 'optimal', 'adaptive'],
        "max_iter": [1000, 500, 2000, 10000],
    }
    rid_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    las_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "selection": ["cyclic", "random"],
    }

    enet_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0, 0.5, 0.9, 0.99],
        "selection": ["cyclic", "random"]
    }

    sv_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "degree": [3, 4],
        "gamma": ["scale", "auto", "float"],
    }

    nusv_params = {
        "nu": [.25, .5, .75],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "gamma": ["scale", "auto", "float"],
        "degree": [3, 4],
        "class_weight": [None, "balanced"]
    }

    lsv_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "loss": ['squared_hinge'],  # 'hinge',
        "penalty": ['l2', 'l1'],
        "multi_class": ['ovr', 'crammer_singer'],
        "max_iter": [10000, 20000, 50000, 100000]
    }
    knn_params = {"n_neighbors": np.arange(2, 100, 3),
                  "weights": ["uniform", "distance"],
                  # "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  "leaf_size": [30, 15, 45],
                  # "p": [1, 2],
                  # "metric": ['euclidean', 'manhattan', 'chebyshev', "minkowski"],
                  }

    necen_params = {
        "metric": ['euclidean', 'manhattan']

    }

    rane_params = {
        "radius": [1.0, 2.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        "weights": ['uniform', 'distance'],
        "leaf_size": [30, 15, 45, 60, 100],
        "p": [1, 2],
        "metric": ['euclidean', 'manhattan', 'chebyshev']
    }

    hist_params = {
        "learning_rate": [0.1, 0.01, 0.001],
        "max_iter": [50, 100, 200, 400],
        "max_leaf_nodes": [31, 15, 45, 60],
        "max_depth": [None, 1, 2]
    }

    cart_params = {"max_depth": range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "sqrt"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    xboost_params = {"learning_rate": [0.1, 0.01],
                     "max_depth": [5, 8],
                     "n_estimators": [100, 200],
                     "colsample_bytree": [0.5, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                       "n_estimators": [300, 500, 700],
                       "colsample_bytree": [0.7, 0.5, 1]}

    cat_params = {'learning_rate': [0.09, 0.1, 0.12, 0.13],
                  "max_depth": [3, 4, 5, 6],
                  "n_estimators": [200, 250, 259, 260, 261]}

    ada_params = {
        'n_estimators': [2, 3, 5, 6, 7, 9, 10, 11, 12, 15, 18],
        'learning_rate': [(0.97 + x / 100) for x in range(0, 20)],
        # 'algorithm': ['SAMME', 'SAMME.R']
    }

    gbm_params = {
        'learning_rate': [0.09, 0.1, 0.085, 0.08],
        'max_depth': [2, 3, 4],
        'max_features': [2, 3, 4, None],
        'max_leaf_nodes': [2, 3, None],
        'n_estimators': [100, 200, 250, 300, 500, 1000]}
    
    mlp_params = {'hidden_layer_sizes':[(100,), (100, 50), (100, 50, 20)],
                  'activation':["relu", "tanh","logistic"],
                  'alpha':[0.0001, 0.01, 0.1],
                  'solver':["lbfgs", "sgd", "adam"],
                  'learning_rate_init':[0.001, 0.01, 0.1],
                  'max_iter':[50, 100, 200, 500]
                  }
    if is_classifier:
        models = [
            # Naive Bayes
            ("GNB", GaussianNB(), gnb_params),
            # ("MNB",MultinomialNB(), mnb_params), Metin saymada kullanılıyor (?)
            # ("CNB",ComplementNB(), cnb_params),
            ("BNB", BernoulliNB(), bnb_params),
            # ("CATNB",CategoricalNB(), catnb_params),
            # Gaussian Process
            ## ("GPC", GaussianProcessClassifier(), gp_params),
            # Linear Model
            # ("LiR",LinearRegression(),lir_params),
            ("LR", LogisticRegression(max_iter=10000), lr_params),
            ("SGD", SGDClassifier(loss='log_loss'), sgd_params),
            # ("RID",Ridge(),rid_params),
            # ("LAS",Lasso(), las_params),
            # ("ENET",ElasticNet(), enet_params),
            # Support Vector Machines
            # ("SVC", SVC(), sv_params),
            # ("NUSVC", NuSVC(), nusv_params),
            # ("LSVC", LinearSVC(max_iter=8000), lsv_params),
            # Neighbors
            ("KNN", KNeighborsClassifier(), knn_params),
            # ("NECEN", NearestCentroid(), necen_params),
            # ("RANEC",RadiusNeighborsClassifier(),rane_params),
            # Tree
            ("CART", DecisionTreeClassifier(), cart_params),
            # Ensemble
            ("RF", RandomForestClassifier(), rf_params),
            ("GBM", GradientBoostingClassifier(), gbm_params),
            ("HIST", HistGradientBoostingClassifier(), hist_params),
            ("ADABoost", AdaBoostClassifier(), ada_params),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xboost_params),
            ("LightGBM", LGBMClassifier(verbose=-1), lightgbm_params),
            ("Catboost", CatBoostClassifier(), cat_params),
            # Neural Networks
            ("MLP",MLPClassifier(),mlp_params)
        ]

    else:
        models = [
            # Naive Bayes
            # ("GNB", GaussianNB(), gnb_params),
            # ("MNB",MultinomialNB(), mnb_params), Metin saymada kullanılıyor (?)
            # ("CNB",ComplementNB(), cnb_params),
            # ("BNB", BernoulliNB(), bnb_params),
            # ("CATNB",CategoricalNB(), catnb_params),
            # Gaussion Process
            # ("GPR", GaussianProcessRegressor(), gp_params),
            # Linear Model
            # ("LR", LogisticRegression(), lr_params),
            # ("SGD", SGDRegressor(), sgd_params),
            ("RID", Ridge(), rid_params),
            ("LAS", Lasso(), las_params),
            # ("ENET", ElasticNet(), enet_params),
            # Support Vector Machines
            # ("SVR", SVR(), sv_params),
            # ("NUSVR", NuSVR(), nusv_params),
            # ("LSVR", LinearSVR(), lsv_params),
            # Neighbors
            ("KNN", KNeighborsRegressor(), knn_params),
            # ("NECEN", NearestCentroid(), necen_params),
            # ("RANER", RadiusNeighborsRegressor(), rane_params),
            # Tree
            ("CART", DecisionTreeRegressor(), cart_params),
            # Ensemble
            ("RF", RandomForestRegressor(), rf_params),
            ("GBM", GradientBoostingRegressor(), gbm_params),
            ("HIST", HistGradientBoostingRegressor(), hist_params),
            ("Adaboost", AdaBoostRegressor(), ada_params),
            ("XGBoost", XGBRegressor(use_label_encoder=False), xboost_params),
            ("LightGBM", LGBMRegressor(verbose=-1), lightgbm_params),
            ("Catboost", CatBoostRegressor(verbose=False), cat_params),
            ("MLP", MLPRegressor(), mlp_params)
        ]

    best_models = {}
    for name, model, params in models:
        print(f"########### {name} ###########")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        if is_grid_search:
            s_best = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        else:
            s_best = RandomizedSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = model.set_params(**s_best.best_params_)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=random_state)
        cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {s_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models


def base_multiclass_models(X, y, cv=5, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovo", "roc_auc_ovr"]):
    """
    :param X:
    :param y:
    :param cv: int
        5
    :param scoring:
        ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovo", "roc_auc_ovr"]
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    print("Base Multiclass Models....")
    models = {
        # Naive Bayes
        "GNB": GaussianNB(),
        # "MNB":MultinomialNB(), # not usable with negative values
        # "CNB":ComplementNB(), # not usable with negative values
        # "BNB": BernoulliNB(), # not usable with negative values
        # "CATNB":CategoricalNB(), # not usable with negative values
        # Gaussian Process
        # "GPC": GaussianProcessClassifier(), # takes too long, not useful
        # Linear Model
        "LR": LogisticRegression(),
        "SGD": SGDClassifier(),
        # Support Vector Machines
        # "SVC": SVC(probability=True),  # takes too long sometimes
        # "NUSVC": NuSVC(probability=True), # takes too long sometimes
        "LSVC": LinearSVC(),
        # Neighbors
        "KNN": KNeighborsClassifier(),
        # Tree
        "CART": DecisionTreeClassifier(),
        "EXTR": ExtraTreeClassifier(),
        # Ensemble
        "RF": RandomForestClassifier(),
        "BAG": BaggingClassifier(),
        "GBM": GradientBoostingClassifier(),
        "HIST": HistGradientBoostingClassifier(),
        "Adaboost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, device="cuda"),
        "LightGBM": LGBMClassifier(verbose=-1, device="gpu"), 
        "Catboost": CatBoostClassifier(verbose=False, devices="gpu"),
        # Neural Networks
        "MLP": MLPClassifier()
    }
    for name, model in models.items():
        print(f"################## {name} ################## ")
        for score_param in scoring:
            cv_results = cross_validate(model, X, y, cv=cv, scoring=score_param)
            print(f"{score_param}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

        print()


def hyperparameter_multiclass_optimization(X, y, cv=5, scoring="roc_auc_ovr", is_grid_search=True):
    """
    :param X:
    :param y:
    :param cv: int
        5
    :param scoring: string
        roc_auc_ovr - roc_auc_ovo
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    :param is_grid_search: boolean
        True
    :return: dict

    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    print("Hyperparameter Multiclass Optimization....")
    # Naive Bayes
    gnb_params = {
        # "priors": [None, [0.2, 0.8]]
        "var_smoothing": [1e-9, 1e-10]
    }

    mnb_params = {
        "alpha": [1.0, 0.5, 0.1],
    }

    cnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "class_prior": [None, 0.2, 0.8]
    }

    bnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "binarize": [0.0, 0.5, 0.8]
    }

    catnb_params = {
        "alpha": [1.0, 0.5, 0.1],
        "class_prior": [None, 0.2, 0.8]
    }
    gp_params = {
        "max_iter_predict": [50, 100, 200, 500]
    }

    lir_params = {
        "fit_intercept": [True, False]
    }

    lr_params = {
        "max_iter": [25000, 50000],
        "penalty": ['l2', 'l1', 'elasticnet'],
        "C": [0.1, 1.0, 1.5]
    }

    sgd_params = {
        "alpha": [0.0001, 0.0001, 0.001, 0.01],
        "penalty": ['l2', 'l1', 'elasticnet', None],
        "learning_rate": ['constant', 'optimal', 'adaptive'],
        "max_iter": [1000, 500, 2000, 10000],
    }
    rid_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    las_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "selection": ["cyclic", "random"],
    }

    enet_params = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0, 0.5, 0.9, 0.99],
        "selection": ["cyclic", "random"]
    }

    sv_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "degree": [3, 4],
        "gamma": ["scale", "auto", "float"],
    }

    nusv_params = {
        "nu": [.25, .5, .75],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "gamma": ["scale", "auto", "float"],
        "degree": [3, 4],
        "class_weight": [None, "balanced"]
    }

    lsv_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "loss": ['squared_hinge'],  # 'hinge',
        "penalty": ['l2', 'l1'],
        "multi_class": ['ovr', 'crammer_singer'],
        "max_iter": [10000, 20000, 50000, 100000]
    }
    knn_params = {"n_neighbors": np.arange(2, 100, 3),
                  "weights": ["uniform", "distance"],
                  # "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  "leaf_size": [30, 15, 45],
                  # "p": [1, 2],
                  # "metric": ['euclidean', 'manhattan', 'chebyshev', "minkowski"],
                  }

    necen_params = {
        "metric": ['euclidean', 'manhattan']

    }

    rane_params = {
        "radius": [1.0, 2.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        "weights": ['uniform', 'distance'],
        "leaf_size": [30, 15, 45, 60, 100],
        "p": [1, 2],
        "metric": ['euclidean', 'manhattan', 'chebyshev']
    }

    hist_params = {
        "learning_rate": [0.1, 0.01, 0.001],
        "max_iter": [50, 100, 200, 400],
        "max_leaf_nodes": [31, 15, 45, 60],
        "max_depth": [None, 1, 2]
    }

    cart_params = {"max_depth": range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "sqrt"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    xboost_params = {"learning_rate": [0.1, 0.01],
                     "max_depth": [5, 8],
                     "n_estimators": [100, 200],
                     "colsample_bytree": [0.5, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                       "n_estimators": [300, 500, 700],
                       "colsample_bytree": [0.7, 0.5, 1]}

    cat_params = {'learning_rate': [0.09, 0.1, 0.12, 0.13],
                  "max_depth": [3, 4, 5, 6],
                  "n_estimators": [200, 250, 259, 260, 261]}

    ada_params = {
        'n_estimators': [2, 3, 5, 6, 7, 9, 10, 11, 12, 15, 18],
        'learning_rate': [(0.97 + x / 100) for x in range(0, 20)],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    gbm_params = {
        'learning_rate': [0.09, 0.1, 0.085, 0.08],
        'max_depth': [2, 3, 4],
        'max_features': [2, 3, 4, None],
        'max_leaf_nodes': [2, 3, None],
        'n_estimators': [100, 200, 250, 300, 500, 1000]}
    
    mlp_params = {'hidden_layer_sizes':[(100,), (100, 50), (100, 50, 20)],
                'activation':["relu", "tanh","logistic"],
                'alpha':[0.0001, 0.01, 0.1],
                'solver':["lbfgs", "sgd", "adam"],
                'learning_rate_init':[0.001, 0.01, 0.1],
                'max_iter':[50, 100, 200, 500]
                } 

    models = [
        # Naive Bayes
        ("GNB", GaussianNB(), gnb_params),
        # ("MNB",MultinomialNB(), mnb_params), Metin saymada kullanılıyor (?)
        # ("CNB", ComplementNB(), cnb_params),
        ("BNB", BernoulliNB(), bnb_params),
        # ("CATNB", CategoricalNB(), catnb_params),
        # Gaussian Process
        # ("GPC", GaussianProcessClassifier(), gp_params),
        # Linear Model
        # ("LiR", LinearRegression(), lir_params),
        ("LR", LogisticRegression(), lr_params),
        ("SGD", SGDClassifier(), sgd_params),
        # ("RID", Ridge(), rid_params),
        # ("LAS", Lasso(), las_params),
        # ("ENET", ElasticNet(), enet_params),
        # Support Vector Machines
        # ("SVC", SVC(probability=True), sv_params),
        # ("NUSVC", NuSVC(probability=True), nusv_params),
        # ("LSVC", LinearSVC(max_iter=8000), lsv_params),
        # Neighbors
        ("KNN", KNeighborsClassifier(), knn_params),
        # ("NECEN", NearestCentroid(), necen_params),
        # ("RANEC", RadiusNeighborsClassifier(), rane_params),
        # Tree
        ("CART", DecisionTreeClassifier(), cart_params),
        # Ensemble
        ("RF", RandomForestClassifier(), rf_params),
        ("GBM", GradientBoostingClassifier(), gbm_params),
        ("HIST", HistGradientBoostingClassifier(), hist_params),
        ("ADABoost", AdaBoostClassifier(), ada_params),
        ("XGBoost", XGBClassifier(use_label_encoder=False), xboost_params),
        ("LightGBM", LGBMClassifier(verbose=-1), lightgbm_params),
        ("Catboost", CatBoostClassifier(verbose=False), cat_params),
        ("MLP",MLPClassifier(),mlp_params)
    ]
    best_models = {}
    for name, model, params in models:
        print(f"########### {name} ###########")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
        if is_grid_search:
            s_best = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False, refit=scoring).fit(X, y)
        else:
            s_best = RandomizedSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False, refit=scoring).fit(X, y)
        final_model = model.set_params(**s_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {s_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models


def voting_model(best_models: dict, X, y, cv=5, is_classifier=True, is_multiclass=False, voting_type="soft",
                 show_error=False):
    """
    :param best_models: dictionary
    :param X:
    :param y:
    :param cv:
    :param is_classifier: boolean
        True
    :param is_multiclass: boolean
        True
    :param voting_type: str
        ['hard', 'soft', 'weighted', 'uniform']

        'hard': Bu seçenek, sınıflandırıcıların tahminlerini sert oylama (hard voting)
        kullanarak birleştirir. Her bir sınıf için çoğunluğu belirleyerek son tahmini seçer.
        Örneğin, birden fazla sınıflandırıcı "Sınıf A" olduğunu tahmin ediyorsa ve sadece biri "Sınıf B" tahmin ediyorsa
        "hard voting" çoğunluk olan "Sınıf A" tahminini seçer.

        'soft': Bu seçenek, sınıflandırıcıların tahmin olasılıklarını kullanarak yumuşak oylama (soft voting) kullanarak
         birleştirir. Her sınıf için tahmin olasılıklarının ortalamasını alır ve en yüksek olasılığa sahip sınıfı seçer.
          Bu yaklaşım, sınıflandırıcıların tahminlerinin belirsiz olduğu durumlarda daha iyi çalışabilir.

        'uniform': Bu seçenek, tüm sınıflandırıcıların aynı ağırlığa sahip olduğu eşit ağırlıklı (uniform) oylamayı
        kullanır. Her sınıflandırıcının tahminleri aynı öneme sahiptir.

        'weighted': Bu seçenek, sınıflandırıcıların ağırlıklarını belirlemenize olanak tanır. Her sınıflandırıcıya özel
        bir ağırlık verebilir ve bu ağırlıkların toplamına dayalı olarak tahminlerin birleştirilmesini yapabilirsiniz.
        Bu, bazı sınıflandırıcıların diğerlerinden daha güvenilir olduğu durumlarda kullanışlı olabilir.

    :return: votingRegressor

    """
    if is_classifier:
        print(f"Voting Model: Classifier...")
        voting = VotingClassifier(estimators=list(best_models.items()), voting=voting_type).fit(X, y)
    else:
        print(f"Voting Model: Regression...")
        voting = VotingRegressor(estimators=list(best_models.items()), weights=voting_type).fit(X, y)
    if is_multiclass:
        cv_results = cross_validate(voting, X, y, cv=cv,
                                    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovo",
                                             "roc_auc_ovr"], error_score="raise" if show_error else np.nan)
        print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
        print(f"Precision Macro: {cv_results['test_precision_macro'].mean()}")
        print(f"Recall Macro: {cv_results['test_recall_macro'].mean()}")
        print(f"F1Score Macro: {cv_results['test_f1_macro'].mean()}")
        print(f"ROC_AUC Ovr: {cv_results['test_roc_auc_ovr'].mean()}")
    else:
        cv_results = cross_validate(voting, X, y, cv=cv, scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                                  "neg_root_mean_squared_error", "r2"]
                                    , error_score="raise" if show_error else np.nan)
        print(f"mean_absolute_error: {abs(cv_results['test_neg_mean_absolute_error'].mean())}")
        print(f"mean_squared_error: {abs(cv_results['test_neg_mean_squared_error'].mean())}")
        print(f"neg_root_mean_squared_error: {abs(cv_results['test_neg_root_mean_squared_error'].mean())}")
        print(f"r2: {cv_results['test_r2'].mean()}")

    return voting


############# Machine Learning Pipeline


############# Importance Table

def plot_importance(model, X, save=False):
    """
    :param model: super-unsupervised model
    :param X: pandas.DataFrame
    :param save: boolean
        True
    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len(X)])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

############# Importance Table

### by Hüseyin Battal ###
## https://www.linkedin.com/in/huseyin-battal ##
