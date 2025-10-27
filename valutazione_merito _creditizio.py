# Imports

# Dataset: https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt
import hvplot.pandas
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import EarlyStopping
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFpr, f_classif, chi2, RFECV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score
pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

# Pre-Processing Functions
def first_analysis(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler):
    dataset, X_train, y_train, X_test, y_test = data_processing_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler)
    rf_prediction(dataset, target, X_train, y_train, X_test, y_test)
    return dataset, X_train, y_train, X_test, y_test
    
def data_processing_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler):
    dataset = get_dataset_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
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

def get_dataset_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical):
    dataset = pd.read_csv(path)
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

#Settings
path = "C:/Users/edoardo.frapiccini/Desktop/vai/AI/pratica/Machine learning for credit scoring/progetti/Lending Club Loan Defaulters Prediction/dataset/All Lending Club loan data/accepted_2007_to_2018Q4.csv"
target = "loan_status"
value_1 = "Fully Paid"
value_0 = "Charged Off"
features_post_loan = ["collection_recovery_fee","funded_amnt","funded_amnt_inv","initial_list_status","last_credit_pull_d","last_fico_range_high","last_fico_range_low","last_pymnt_amnt","last_pymnt_d","mths_since_last_delinq","mths_since_last_major_derog","mths_since_last_record","next_pymnt_d","num_tl_120dpd_2m","num_tl_30dpd","out_prncp","out_prncp_inv","pymnt_plan","recoveries","total_pymnt","total_pymnt_inv","total_rec_int","total_rec_late_fee","total_rec_prncp","sec_app_inq_last_6mths","sec_app_mths_since_last_major_derog","hardship_flag","hardship_type","hardship_reason","hardship_status","deferral_term","hardship_amount","hardship_start_date","hardship_end_date","payment_plan_start_date","hardship_length","hardship_dpd","hardship_loan_status","orig_projected_additional_accrued_interest","hardship_payoff_balance_amount","hardship_last_payment_amount","debt_settlement_flag","debt_settlement_flag_date","settlement_status","settlement_date","settlement_amount","settlement_percentage","settlement_term"]
features_probably_post_loan = [
    "inq_last_6mths",
    "bc_util","num_actv_rev_tl","avg_cur_bal","num_actv_bc_tl","tot_cur_bal","total_acc",
    "issue_d"
]
lending_club_exclusive_features = ["id", "url", "grade", "sub_grade", "issue_d"]
subcategories_of_other_features = ["title"]
features_to_drop = features_post_loan + features_probably_post_loan + lending_club_exclusive_features + subcategories_of_other_features
missing_data_percentage_accepted = 0.9
correlation_percentage_accepted = 0.9
strategy_to_convert_nan = 'drop'
strategy_to_convert_nan_to_numerical = 'median'
strategy_to_convert_nan_to_categorical = 'most_frequent'
dates = ['earliest_cr_line']
threshold_dummmies = 50
categorical_features_with_numbers_inside = ['term', 'zip_code', 'emp_length']
features_to_group = [
    {
        'feature': 'home_ownership',
        'values_to_group': ['ANY', 'NONE'],
        'value_grouped': 'OTHER'
    }
]
anova_flag = False
outliers_flag = True
test_size = 0.33
balanced_strategy = 'random_undersampling'
scaler = MinMaxScaler()
rfecv_flag = False

# Main
df, df_processed, X_train, y_train, X_test, y_test = data_processing_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, test_size, balanced_strategy, rfecv_flag, scaler)

# Stacking
def stacking(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, balanced_strategy, rfecv_flag, scaler, iteration_number):
    dataset, df_processed, X_train, y_train, X_test, y_test = data_processing_for_a_binary_task(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, 0.5, balanced_strategy, rfecv_flag, scaler)
    number_of_features = len(df_processed.columns) // 2
    for i in range(iteration_number):
        new_feature_name = 'prediction_' + str(i)
        print(new_feature_name)
        results, features_most_important = get_rf_prediction(df_processed, target, X_train, y_train, X_test, y_test, number_of_features)
        df_processed = build_new_df(df_processed, X_test, results, y_test, new_feature_name, features_most_important, target)
        X_train, y_train, X_test, y_test = get_X_train_y_train_X_test_y_test(df_processed, target, 0.5, balanced_strategy, rfecv_flag, scaler)
        number_of_features = len(df_processed.columns) // 2
    

def get_rf_prediction(df, target, X_train, y_train, X_test, y_test, number_of_features):
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
    sns.barplot(x='Importance', y='Feature', data=features_importance_df.head(number_of_features))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    return rf_y_test_pred, features_importance_df.head(number_of_features)['Feature'].values

def build_new_df(df, X_test, results, y_test, new_feature, features_most_important, target):
    columns = df.drop(target, axis=1).columns
    new_df = pd.DataFrame(X_test, columns=columns)
    new_df = new_df[features_most_important]
    new_df[target] = y_test.values
    new_df[new_feature] = results
    return new_df

""" stacking(path, target, value_1, value_0, features_to_drop, 
                                      missing_data_percentage_accepted, correlation_percentage_accepted,
                                      strategy_to_convert_nan, strategy_to_convert_nan_to_numerical, strategy_to_convert_nan_to_categorical,
                                      dates, categorical_features_with_numbers_inside, features_to_group, threshold_dummmies,
                                      anova_flag, outliers_flag, balanced_strategy, rfecv_flag, scaler, 3) """

# Pre-processing custom
#emp_title
""" def replaceProfession(data_with_professions_focus, profession):
    category = data_with_professions_focus['emp_title'].str.contains(profession, case=False)
    data_with_professions_focus.loc[category, 'emp_title'] = profession
    return data_with_professions_focus

# Copia del DataFrame originale
data_with_professions_focus = data_with_categorical_variables_processed.copy()

# Trasformazione delle professioni in minuscolo e rimozione spazi bianchi
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].str.lower().str.strip().str.replace(' ', '_')

#Raggruppo le professioni in categorie 
#professioni sanitarie e sociali
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'nurse')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'therapist')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'medical')
health_and_social_care = ['nurse','therapist','medical','lpn','cna','social_worker','rn','paramedic','physician','lvn','phlebotomist']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(health_and_social_care, 'health_and_social_care')

#farmacisti
pharmacist = ['pharmacist','pharmacy_technician']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(pharmacist, 'pharmacist')

#dentisti
dentist = ['dentist','dental_hygienist']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(dentist, 'dentist')

#istruzione
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'teacher')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'professor')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'school')
instruction = ['teacher','professor','school','educator','instructor']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(instruction, 'instruction')

#postini
postman = ['letter_carrier','mail_carrier','courier']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(postman, 'postman')

#presidenti
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'president')
president = ['owner','president','ceo','executive_director','principal','vp','general_manager','gm','cfo','coo']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(president, 'president')

#professioni legali e giuridiche
legal_and_juridical = ['attorney','paralegal','legal_assistant','correctional_officer']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(legal_and_juridical, 'legal_and_juridical')

#cuochi
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'chef')
catering = ['chef','server','bartender','cook']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(catering, 'catering')

#militari e sicurezza
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'officer')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'security')
security = ['officer','firefighter','deputy_sheriff','deputy','inspector','sergeant','captain','us_army','detective','investigator','lieutenant','shipping']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(security, 'security')

#autisti
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'driver')
driver = ['driver','conductor','courier','bus_operator','pilot','carrier']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(driver, 'driver')

#tecnologia e ingegneria
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'engineer')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'tech')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'program')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'developer')
engineering_and_technology = ['engineer','tech','program','it','developer','it_specialist','electrician',
                              'systems_analyst','systems_administrator','installer','service_technician']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(engineering_and_technology, 'engineering_and_technology')

#produzione e operazioni
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'operator')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'warehouse')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'technician')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'mechanic')
production_and_operations = ['operator','technician','warehouse','mechanic','maintenance','carpenter', 'machinist','laborer','production_worker',
                             'plumber','material_handler','custodian','welder','lineman']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(production_and_operations, 'production_and_operations')

#management
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'manager')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'supervisor')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'director')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'admin')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'lead')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'managing')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'management')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'coordinator')
management = ['manager','supervisor','director','admin','lead','managing','management','superintendent','management','coordinator','dispatcher',
              'foreman']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(management, 'management')

#vendite e business
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'sales')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'account')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'business')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'finance')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'financial')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'bank')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'credit')
business = ['sales','account','business','finance','financial','bank','credit','agent','controller','analyst','buyer','auditor','marketing','teller',
            'realtor','clerk','cashier','associate','underwriter','loan_officer','bookkeeper','clerical']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(business, 'business')

#assistenti
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'assistant')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'secretary')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'attendant')
assistant = ['assistant','secretary','attendant','receptionist','customer_service','customer_service_rep',
              'customer_service_representative','csr','caregiver']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(assistant, 'assistant')

#consulenti
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'consultant')
data_with_professions_focus = replaceProfession(data_with_professions_focus, 'advisor')
consultant = ['consultant','advisor','counselor']
data_with_professions_focus['emp_title'] = data_with_professions_focus['emp_title'].replace(consultant, 'consultant')

#other_frequent_professions = ['partner','pastor','recruiter','human_resources','claims_adjuster']

# Salvataggio dei risultati in un file CSV
#data_with_professions_focus['emp_title'].value_counts().head(100).to_csv('professions.csv')

print("Dimensioni del dataset: " + str(data_with_professions_focus.shape))
hundred_most_frequent_professions = data_with_professions_focus['emp_title'].value_counts().head(100).index
data_with_professions_focus = data_with_professions_focus[data_with_professions_focus['emp_title'].isin(hundred_most_frequent_professions)] 
print("Dimensioni del dataset con le 100 professioni più frequenti: " + str(data_with_professions_focus.shape))
data_with_professions_focus = pd.get_dummies(data_with_professions_focus, columns=['emp_title'], drop_first=True)
print("Dimensioni del dataset con le 100 professioni più frequenti dopo get_dummies: " + str(data_with_professions_focus.shape)) """

# Neural Networks Functions
def train_ann(ann, num_epochs, batch_value, lr):
    ann.compile(optimizer = Adam(learning_rate = lr),  
        loss='binary_crossentropy',  
        metrics=['accuracy'])  
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #model_checkpoint = ModelCheckpoint('best_model_1.h5', save_best_only=True)
    history = ann.fit(X_train, y_train,
        batch_size = batch_value,
        epochs = num_epochs,
        validation_data = (X_test, y_test),
        callbacks = [early_stopping],
        #callbacks = [early_stopping, model_checkpoint],
        verbose=0) 
    plot_performance(history)
    y_test_pred_proba = ann.predict(X_test)
    threshold = 0.5
    y_test_pred = (y_test_pred_proba > threshold).astype(float).flatten()
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_test_pred)}\n")
    print(f"ROC AUC: {roc_auc_score(y_test, y_test_pred) * 100:.2f}%")
    print(f"Accuracy Score: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")
    print(f"f1 Score: {f1_score(y_test, y_test_pred) * 100:.2f}%")
    return history

def plot_performance(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()  

input_shape = X_train.shape[1]
ann = Sequential([  
    Dense(128, input_shape=(input_shape, ) ,activation='relu'),  
    Dense(64 , activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  
])
history = train_ann(ann, num_epochs=100, batch_value=32, lr = 0.001)

""" # Altri modelli per dati tabulari:
- MLP (Multilayer Perceptron): Un modello di rete neurale feedforward classico, adatto per compiti di classificazione e regressione con dati tabulari.
- TabNet: Utilizza un'attenzione sequenziale per selezionare in modo dinamico quali feature considerare a ogni decisione, migliorando l'interpretabilità e le performance su dati tabulari.
- TabTransformer: Combina trasformatori con embedding di feature per catturare relazioni complesse tra le feature tabulari.
- NODE (Neural Oblivious Decision Ensembles): Un modello che combina alberi decisionali e reti neurali per migliorare la performance su dati tabulari.
- SAINT: Utilizza self-attention per considerare le interazioni tra tutte le feature di un record e intersample attention per considerare le interazioni tra diversi record.
- DNDT (Deep Neural Decision Trees): Modello che integra alberi decisionali all'interno di una rete neurale profonda, cercando di combinare la potenza di apprendimento delle reti neurali con l'interpretabilità degli alberi decisionali.
- DeepFM: Combina reti neurali profonde con macchine di fattorizzazione per catturare sia interazioni di feature esplicite che implicite.
- FT-Transformer: Utilizza un approccio a trasformatori per tabular data, facilitando la modellazione delle interazioni tra feature.
- VIME: Utilizza l'esplorazione di informazioni variate per migliorare la robustezza e l'apprendimento su dati tabulari.
- Net-DNF: Modello che utilizza formule normali disgiuntive per migliorare l'interpretabilità e la performance su dati tabulari.
- TabResNet: Adatta le reti ResNet per dati tabulari, sfruttando le connessioni residue per migliorare l'apprendimento.
- Autoint: Modello che utilizza meccanismi di autoattenzione per catturare interazioni complesse tra feature.
- DeepGBM: Combina gradient boosting machine con deep learning per migliorare le capacità predittive su dati tabulari.
- Wide & Deep: Combina modelli "wide" (lineari) e "deep" (reti neurali) per migliorare la capacità di generalizzazione e di apprendimento di pattern complessi.
- CatBoost, XGBoost, LightGBM: Algoritmi di boosting molto efficaci per dati tabulari, noti per la loro velocità e performance.
- DANet (Deep Autoencoder Network): Utilizza autoencoder per catturare rappresentazioni latenti dei dati tabulari, migliorando la capacità di apprendimento e la riduzione del rumore.
- DCN (Deep & Cross Network): Modello che integra reti neurali profonde con cross networks per catturare interazioni feature non lineari.
- GA2M (Generalized Additive Models with interactions): Modelli che combinano la flessibilità delle reti neurali con la trasparenza e l'interpretabilità dei modelli additivi generalizzati. """

""" # Modelli per classificazione:
- MLP (Multilayer Perceptron):
    - Pro: Facile da implementare e adatto per una varietà di compiti di classificazione.
    - Contro: Può non essere il più efficiente in termini di capacità di catturare interazioni complesse tra feature senza una buona pre-elaborazione dei dati.
- TabNet:
    - Pro: Utilizza l'attenzione sequenziale per selezionare dinamicamente le feature più rilevanti. Buona interpretabilità e performance su dati tabulari.
    - Contro: Può essere complesso da addestrare rispetto a modelli più semplici.
- TabTransformer:
    - Pro: Combina la potenza dei trasformatori con embedding di feature, ottimo per catturare relazioni complesse.
    - Contro: Può essere computazionalmente intensivo.
- NODE (Neural Oblivious Decision Ensembles):
    - Pro: Combina alberi decisionali e reti neurali, offre buone performance su dati tabulari.
    - Contro: Relativamente nuovo, potrebbe avere meno documentazione e supporto rispetto ad altri modelli.
- SAINT (Self-Attention and Intersample Attention):
    - Pro: Utilizza l'attenzione per considerare le interazioni tra tutte le feature e tra diversi record, ideale per dati complessi.
    - Contro: Complesso da implementare e addestrare.
- CatBoost:
    - Pro: Algoritmo di boosting specificamente ottimizzato per feature categoriche e numeriche. Molto efficiente e robusto per classificazione.
    - Contro: Come qualsiasi modello di boosting, potrebbe richiedere un'attenta regolazione degli iperparametri.
- XGBoost:
    - Pro: Molto popolare e altamente performante per una vasta gamma di compiti di classificazione su dati tabulari.
    - Contro: Può essere suscettibile all'overfitting se non regolato correttamente.
- LightGBM:
    - Pro: Molto veloce ed efficiente in termini di memoria. Buone performance su grandi dataset.
    - Contro: Richiede una buona pre-elaborazione delle feature categoriche.
- DeepGBM:
    - Pro: Combina gradient boosting con deep learning per migliorare le capacità predittive.
    - Contro: Può essere complesso da implementare e computazionalmente costoso.
- Wide & Deep:
    - Pro: Combina modelli "wide" (lineari) e "deep" (reti neurali) per migliorare la capacità di generalizzazione e di apprendimento di pattern complessi.
    - Contro: Richiede un'attenta progettazione dell'architettura e regolazione degli iperparametri. """

# Modelli per classificazione binaria:
""" #TabNet
#pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
clf = TabNetClassifier()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['auc'], max_epochs=100)
preds = clf.predict(X_test)
print(f"ROC AUC: {roc_auc_score(y_test, preds) * 100:.2f}%") """

""" # CatBoost
#pip install catboost
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', eval_metric='AUC', verbose=100)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)
predictions = model.predict(X_test) """

""" # XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=6, objective='binary:logistic')
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=100)
predictions = model.predict(X_test) """

""" # LightGBM
#gestisce in autonomia il dataset grezzo senza bisogno di preprocessarlo
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': 0
}
model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], early_stopping_rounds=100)
predictions = model.predict(X_test) """

""" # Wide & Deep
from deepctr.models import WDL
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
sparse_features = df.select_dtypes(exclude=[np.number]).columns
dense_features = df.select_dtypes(include=[np.number]).columns
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=X[feat].nunique()) for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['AUC'])
X = df.drop(target, axis=1)
y = df[target]
model.fit(X[feature_names].values, y.values, batch_size=256, epochs=10, validation_split=0.2) """

""" # MLP 
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, solver='adam', verbose=10, random_state=42, tol=0.0001)
model.fit(X_train, y_train)
predictions = model.predict(X_test) """

""" # TabTransformer
#pip install transformers
from transformers import TabTransformer
model = TabTransformer(input_dim=X_train.shape[1], output_dim=1, num_transformer_blocks=6, num_attention_heads=8, hidden_dim=128, dropout_rate=0.1)
model.compile("adam", "binary_crossentropy", metrics=['AUC'])
model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_test, y_test))
predictions = model.predict(X_test) """
