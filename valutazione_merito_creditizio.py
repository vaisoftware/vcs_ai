# Locale
# python -m uvicorn valutazione_merito_creditizio:app --host 0.0.0.0 --port 8080
# http://localhost:8080/ai

# Remoto
# python -m uvicorn valutazione_merito_creditizio:app --host 80.88.88.48 --port 11001 &
# http://80.88.88.48:11001/ai

# Core
import pandas as pd
# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
# LightGBM
import lightgbm as lgb
# Libreria per creare API web
from fastapi import FastAPI
# Libreria per la definizione di modelli di dati
from pydantic import BaseModel
# Libreria per l'interpretabilità dei modelli
import shap

# Inizializza l'app FastAPI
app = FastAPI()

# --- Pydantic model per la richiesta
class Input(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    zip_code: str
    features_selected: list = []

province_cap = {
    "AG": "921", "AL": "150", "AN": "600", "AO": "110", "AR": "521", "AP": "631", "AT": "141", "AV": "831",
    "BA": "701", "BT": "761", "BL": "321", "BN": "821", "BG": "240", "BI": "138", "BO": "401", "BZ": "391",
    "BS": "250", "BR": "721", "CA": "091", "CL": "931", "CB": "861", "CE": "811", "CT": "951", "CZ": "881",
    "CH": "661", "CO": "221", "CS": "871", "CR": "261", "KR": "889", "CN": "120", "EN": "941", "FM": "639",
    "FE": "441", "FI": "501", "FG": "711", "FC": "471", "FR": "031", "GE": "161", "GO": "3417", "GR": "581",
    "IM": "181", "IS": "8617", "SP": "191", "AQ": "671", "LT": "040", "LE": "730", "LC": "238", "LI": "570",
    "LO": "269", "LU": "550", "MC": "620", "MN": "460", "MS": "540", "MT": "751", "ME": "980", "MI": "201",
    "MO": "411", "MB": "208", "NA": "801", "NO": "281", "NU": "080", "OR": "0907", "PD": "350", "PA": "901",
    "PR": "431", "PV": "271", "PG": "061", "PU": "610", "PE": "651", "PC": "291", "PI": "561", "PT": "511",
    "PN": "3317", "PZ": "851", "PO": "591", "RG": "971", "RA": "481", "RC": "891", "RE": "421", "RI": "021",
    "RN": "479", "RM": "001", "RO": "450", "SA": "841", "SS": "071", "SV": "171", "SI": "531", "SR": "961",
    "SO": "231", "TA": "741", "TE": "641", "TR": "051", "TO": "101", "TP": "911", "TN": "381", "TV": "311",
    "TS": "341", "UD": "331", "VA": "211", "VB": "288", "VC": "131", "VE": "301", "VR": "371", "VV": "899",
    "VI": "361", "VT": "011"
}

def preprocess_input(input_data):
    input_dict = input_data.dict()
    features_selected = input_dict.pop('features_selected', [])
    input_df = pd.DataFrame([input_dict])
    input_df['zip_code'] = input_df['zip_code'].map(province_cap)
    purpose_features = ['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement',
                        'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other',
                        'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding']
    for feature in purpose_features:
        if feature == 'purpose_other':
            input_df[feature] = 1
        else:
            input_df[feature] = 0
    input_df['disbursement_method_DirectPay'] = 0
    # 0 I fondi vengono accreditati direttamente al richiedente (prestito personale classico)
    # 1 I fondi vengono inviati direttamente ai creditori per estinguere debiti esistenti (tipico nei debt consolidation loan)
    from joblib import load
    scaler = load('../scaler.pkl')
    input_scaled  = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled , columns=input_df.columns)
    return input_scaled, input_scaled_df, features_selected

def get_features_importance():
    model = lgb.Booster(model_file='lgbm_model.txt')
    importance = model.feature_importance(importance_type='gain') 
    # importance_type = 'split' o 'gain'. 
    # con split di interpreta quante volte una feature è stata usata per fare uno split
    # con gain quanto ha contribuito in termini di riduzione della loss
    feature_names = model.feature_name()
    feature_importance_dict = dict(zip(feature_names, importance))
    return feature_importance_dict

def get_xai(input_scaled_df):
    model = lgb.Booster(model_file='lgbm_model.txt')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled_df)
    return shap_values

@app.post("/valutazione_merito_creditizio")
def get_prob(richiesta: Input):
    model = lgb.Booster(model_file='lgbm_model.txt')
    input_scaled, input_scaled_df, features_selected = preprocess_input(richiesta)
    predictions = model.predict(input_scaled)
    prob = predictions[0] * 100
    shap_values = get_xai(input_scaled_df)
    if not isinstance(shap_values, list):
        shap_values = shap_values.tolist()[0]
    else:
        shap_values = shap_values[0].tolist()[0]
    shap_dict = dict(zip(input_scaled_df.columns, shap_values))
    feature_importance = get_features_importance()
    shap_dict_filtered = {k: v for k, v in shap_dict.items() if k in feature_importance}
    shap_dict_sorted = dict(
        sorted(
            shap_dict_filtered.items(),
            key=lambda item: feature_importance[item[0]],
            reverse=True
        )
    )
    if len(features_selected) > 0:
        shap_dict_sorted = {k: v for k, v in shap_dict_sorted.items() if k in features_selected}
        feature_importance = {k: v for k, v in feature_importance.items() if k in features_selected}
    return {"prob": prob, "shap_values": shap_dict_sorted, "feature_importance": feature_importance}
