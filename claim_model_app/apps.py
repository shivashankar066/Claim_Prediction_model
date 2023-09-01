from django.apps import AppConfig
from configparser import ConfigParser
import json
import pickle
import os

config = ConfigParser()
config.read(os.path.join("claim_model_app", "config", "config.ini"))

cat_cos_list = config['path']['cat_cols_list']
with open(cat_cos_list, 'r') as f:
    categorical_columns = json.load(f)
    categorical_columns = json.loads(categorical_columns)

label_encoders = {}
for col in categorical_columns:
    file_name = os.path.join('claim_model_app', 'config', '{}_label_encoder.pkl'.format(col))
    with open(file_name, 'rb') as f:
        encoder = pickle.load(f)
        label_encoders[col] = encoder


class ClaimModelAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "claim_model_app"

    categorical_columns = categorical_columns
    label_encoders = label_encoders

    # Top Denied Cpt
    top_denied_cpt_dic_path = config['path']['top_denied_cpt_path']
    with open(top_denied_cpt_dic_path, 'r') as f:
        top_denied_cpt = json.load(f)
    # Top Denied Payer
    top_denied_payer_dic_path = config['path']['top_denied_payer_path']
    with open(top_denied_payer_dic_path, 'r') as f:
        top_denied_payer = json.load(f)

    payer_mapping_path = config['path']['payer_mapping_path']
    with open(payer_mapping_path, 'r') as f:
        payer_mapping = json.load(f)

    # Unique Cpt
    unique_cpt_path = config['path']['unique_cpt_path']
    with open(unique_cpt_path, 'r') as f:
        unique_cpt = json.load(f)

    most_frequent_payer = config['data']['most_frequent_payer']
    # load XGBoost Model
    xgboost_model_path = config['path']['xgboost_model_path']
    with open(xgboost_model_path, 'rb') as f:
        print("model loaded")
        xgb_model = pickle.load(f)
        xgb_model_fit_feature_order = xgb_model.feature_names_in_
        print("Length of Training model", len(xgb_model_fit_feature_order))
        xgb_model.verbose = False
