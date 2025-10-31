# Core
import pandas as pd
import numpy as np
# Visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectFpr, f_classif, RFECV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score
)
# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTETomek
# LightGBM
import lightgbm as lgb

def first_analysis(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler):
    dataset, X_train, y_train, X_test, y_test = data_processing_for_a_binary_task(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler)
    rf_prediction(dataset, target, X_train, y_train, X_test, y_test)
    return dataset, X_train, y_train, X_test, y_test
    
def data_processing_for_a_binary_task(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler):
    dataset = get_dataset_for_a_binary_task(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical)
    categorical_dataset_processed = process_categorical_dataset(dataset, dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies, target)
    dataset_with_features_selected_by_anova = feautures_selection_with_anova(anova_flag, categorical_dataset_processed, target)
    dataset_without_features_with_variance_null = delete_features_with_variance_null(dataset_with_features_selected_by_anova, target)
    dataset_without_features_most_correlates = drop_features_most_correlated(dataset_without_features_with_variance_null, correlation_percentage_accepted, target, value_1, value_0)
    dataset_without_outliers = manage_outliers(outliers_flag, dataset_without_features_most_correlates, target)
    dataset_without_outliers[target] = dataset_without_outliers[target].map({value_1:1, value_0:0})
    X_train, y_train, X_test, y_test = get_X_train_y_train_X_test_y_test(dataset_without_outliers, target, test_size, balanced_strategy, rfecv_flag, scaler)
    return dataset, dataset_without_outliers, X_train, y_train, X_test, y_test

def get_dataset_for_a_binary_task(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical):
    dataset = pd.read_csv(path)
    dataset = dataset[initial_features]
    dataset_with_only_binary_class = dataset[(dataset[target] == value_1) | (dataset[target] == value_0)] 
    dataset_after_features_selection = dataset_with_only_binary_class.drop(features_to_drop, axis=1)
    dataset_without_feature_with_nan = dataset_after_features_selection.dropna(thresh=missing_data_percentage_accepted*len(dataset_after_features_selection), axis=1) 
    dataset_without_numeric_features_most_correlates = drop_features_most_correlated(dataset_without_feature_with_nan, correlation_percentage_accepted, target, value_1, value_0)
    dataset_without_records_with_nan = manage_nan(dataset_without_numeric_features_most_correlates, strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical)
    return dataset_without_records_with_nan

def drop_features_most_correlated(df, correlation_percentage_accepted, target, value_1, value_0):
    dataset = df.copy()
    numerical_data = dataset.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()
    high_correlation_pairs = correlation_matrix[abs(correlation_matrix) > correlation_percentage_accepted].stack().reset_index()
    high_correlation_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']
    high_correlation_pairs = high_correlation_pairs.loc[(high_correlation_pairs['Feature_1'] != high_correlation_pairs['Feature_2'])]
    dataset[target] = dataset[target].map({value_1:1, value_0:0})
    features_to_drop = []
    for index, row in high_correlation_pairs.iterrows():
        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']
        if feature_1 not in features_to_drop and feature_2 not in features_to_drop:
            correlation_with_target_1 = dataset[feature_1].corr(dataset[target])
            correlation_with_target_2 = dataset[feature_2].corr(dataset[target])
            if abs(correlation_with_target_1) > abs(correlation_with_target_2):
                features_to_drop.append(feature_2)
            else:
                features_to_drop.append(feature_1)
    df = df.drop(features_to_drop, axis=1)
    return df

def manage_nan(df, strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical):
    if strategy_to_convert_nan == 'drop':
        df = df.dropna()
        return df
    if strategy_to_convert_nan == 'mice':
        mice_imputer = IterativeImputer()
        for col in df.columns[df.isnull().any()]:
            df[col] = mice_imputer.fit_transform(df[[col]])
        return df
    numeric_imputer = SimpleImputer(strategy=strategy_to_convert_nan_to_numerical)
    numeric_dataset = df.select_dtypes(include=[np.number])
    numeric_dataset_imputed = numeric_imputer.fit_transform(numeric_dataset)
    df.loc[:, numeric_dataset.columns] = numeric_dataset_imputed
    imputer = SimpleImputer(strategy=strategy_to_convert_nan_to_categorical)
    categorical_dataset_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(categorical_dataset_imputed, columns=df.columns)
    return df

def manage_outliers(outliers_flag, df, target):
    if outliers_flag:
        for col in df.columns:
            if col != target:
                outliers_if = _apply_isolation_forest(df[col])
                outliers_mz = _modified_z_score(df[col])
                outliers_combined = outliers_if & outliers_mz
                df = df[~outliers_combined]
    return df

def _apply_isolation_forest(column):
    column_values = column.values.reshape(-1, 1)
    clf = IsolationForest(max_samples=0.1, contamination=0.001)
    outliers = clf.fit_predict(column_values)
    outliers_if = outliers == -1
    return outliers_if

def _modified_z_score(x):
    median_x = np.median(x)
    mad_x = np.median(np.abs(x - median_x))
    modified_z_scores = 0.6745 * (x - median_x) / mad_x
    outliers_mz = modified_z_scores > 3 
    return outliers_mz

def process_categorical_dataset(df, dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies, target):
    df = convert_boolean_to_numerical(df)
    df = convert_date_to_numerical(df, dates)
    df = convert_strings_with_numbers_inside(df, categorical_features_with_numbers_inside)
    df = group_values(df, features_to_group)
    df = conver_string_to_numerical(df, threshold_dummmies, target)
    return df

def convert_boolean_to_numerical(df):
    bool_columns = df.select_dtypes(include='bool').columns
    df[bool_columns] = df[bool_columns].astype(int)
    return df

def convert_date_to_numerical(df, dates):
    reference_datetime = pd.to_datetime('2007-01-01')
    for date in dates:
        df[date] = pd.to_datetime(df[date])
        df[date] = (df[date] - reference_datetime).dt.days
    return df

def convert_strings_with_numbers_inside(df, categorical_features_with_numbers_inside):
    for feature in categorical_features_with_numbers_inside:
        df[feature] = df[feature].str.extract('(\d+)').astype(float)
    return df

def group_values(df, features_to_group):
    for feature_to_group in features_to_group:
        df.loc[df[feature_to_group['feature']].isin(feature_to_group['values_to_group']), feature_to_group['feature']] = feature_to_group['value_grouped']
    return df

def conver_string_to_numerical(df, threshold_dummmies, target):
    categorical_df = df.select_dtypes(exclude=[np.number])
    dummies = []
    for column in categorical_df.columns:
        if column != target:
            if len(categorical_df[column].unique()) < threshold_dummmies:
                dummies.append(column)
            else:
                df[column] = LabelEncoder().fit_transform(df[column])
                df[column] = (df[column] - df[column].mean()) / df[column].std() #normalization: mean 0, std 1    
    df = pd.get_dummies(df, columns=dummies, drop_first=True)
    return df

def feautures_selection_with_anova(anova_flag, df, target):
    """" 
    Anova per selezionare le features categoriche più significative per un target numerico
    o per selezionare le features numeriche più significative per un target categorico.
    Chi2 per selezionare le features categoriche più significative per un target categorico.
    """
    if anova_flag:
        features = df.drop(target, axis=1)
        target = df[target]
        selector = SelectFpr(f_classif, alpha=0.05)
        selector.fit(features, target)
        features_most_significant = features.columns[selector.get_support()]
        return df[features_most_significant]
    return df

def delete_features_with_variance_null(df, target):
    for column in df.columns:
        if column != target and df[column].var() == 0:
            df = df.drop(column, axis=1)
    return df

def get_X_train_y_train_X_test_y_test(df, target, test_size, balanced_strategy, rfecv_flag, scaler):
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    X_train, y_train = balanced_dataset(train, target, balanced_strategy)
    X_test = test.drop(target, axis=1)
    y_test = test[target].astype(int)
    X_train, X_test = rfecv(rfecv_flag, X_train, y_train, X_test)
    X_train, X_test = normalize_data(scaler, X_train, X_test)
    return X_train, y_train, X_test, y_test

def balanced_dataset(train, target, balanced_strategy):
    X_train_unbalanced = train.drop(target, axis=1)
    y_train_unbalanced = train[target].astype(int) 
    if(balanced_strategy == 'random_undersampling'):
        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X_train_unbalanced, y_train_unbalanced)
    if(balanced_strategy == 'smote_oversampling'):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train_unbalanced, y_train_unbalanced)
    if(balanced_strategy == 'tomek_undersampling'):
        tomek = TomekLinks()
        return tomek.fit_resample(X_train_unbalanced, y_train_unbalanced)
    if(balanced_strategy == 'cluster_centroids_undersampling'):
        cc = ClusterCentroids(ratio={0: 10})
        return cc.fit_sample(X_train_unbalanced, y_train_unbalanced)
    if(balanced_strategy == 'smote_tomek'):
        smote_tomek = SMOTETomek(random_state=42)
        return smote_tomek.fit_resample(X_train_unbalanced, y_train_unbalanced)
    return X_train_unbalanced, y_train_unbalanced

def rfecv(rfecv_flag, X_train, y_train, X_test):
    columns = X_train.columns
    if(rfecv_flag):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='roc_auc') 
        rfecv = rfecv.fit(X_train, y_train)
        print('Optimal number of features :', rfecv.n_features_)
        columns = X_train.columns[rfecv.support_]
        print('Best features :', columns)
    return X_train[columns], X_test[columns]

def normalize_data(scaler, X_train, X_test):
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def rf_prediction(df, target, X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_y_test_pred = rf.predict(X_test)
    print("auc " + str(roc_auc_score(y_test, rf_y_test_pred)))
    print("acc " + str(accuracy_score(y_test, rf_y_test_pred)))
    print(" f1 " + str(f1_score(y_test, rf_y_test_pred)))
    features_importance = rf.feature_importances_
    features_importance_df = pd.DataFrame({'Feature': df.drop(target, axis=1).columns, 'Importance': features_importance})
    features_importance_df = features_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

path = "C:/Users/edoardo.frapiccini/Desktop/vai/AI/pratica/Machine learning for credit scoring/progetti/Lending Club Loan Defaulters Prediction/dataset/All Lending Club loan data/accepted_2007_to_2018Q4.csv"
target = "loan_status"
value_1 = "Fully Paid"
value_0 = "Charged Off"
initial_features = ["loan_amnt", "term", "int_rate", "zip_code", "purpose", "disbursement_method", "loan_status"]
features_to_drop = []
missing_data_percentage_accepted = 0.9
correlation_percentage_accepted = 0.9
strategy_to_convert_nan = 'drop'
strategy_to_convert_nan_to_numerical = 'median'
strategy_to_convert_nan_to_categorical = 'most_frequent'
dates = []
threshold_dummmies = 50
categorical_features_with_numbers_inside = ['term', 'zip_code']
features_to_group = []
anova_flag = False
outliers_flag = True
test_size = 0.33
balanced_strategy = 'random_undersampling'
scaler = MinMaxScaler()
rfecv_flag = False

df, df_processed, X_train, y_train, X_test, y_test = data_processing_for_a_binary_task(path, target, value_1, value_0, initial_features, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler)

# LightGBM Model
feature_columns = [col for col in df_processed.columns.tolist() if col != target]
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': 0
}
model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data])
predictions = model.predict(X_test)
print(f"ROC AUC: {roc_auc_score(y_test, predictions) * 100:.2f}%")
# save the model
model.save_model('lgbm_model.txt')