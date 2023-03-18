from flask import Flask, jsonify
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import shap
import imblearn

app = Flask(__name__)

def load_pickle(filename):
    with open(f'./{filename}.pickle', 'rb') as f:
        return pickle.load(f)

df = load_pickle('data_cleaned_sample')
df = df.set_index('SK_ID_CURR')
df_cli = df[[c for c in df.columns if c not in ['TARGET']]]
list_clients = list(df_cli.index)

# Pour la prédiction d'un client

pipeline = load_pickle('lgbm_model')
scaler = StandardScaler()
scaler_df = pd.DataFrame(scaler.fit_transform(df_cli), 
                        columns=df_cli.columns,
                        index=df_cli.index)
model = pipeline.steps[1][1]

threshold = 0.7

#Pour l'interprétation
explain_shap = load_pickle('explainer')
nb_client_explain_global = 1000

@app.route('/')
@app.route('/clients/',  methods=['GET'])
def clients():
    response = jsonify(list_clients)
    return response

# Pour un client selectionné
@app.route('/client/<id>',  methods=['GET'])
def client(id):
    data_client = df_cli[df_cli.index == int(id)]
    response = data_client.to_dict(orient='index')
    return jsonify(response)

# Pour la prédiction
@app.route('/predict/<id>',  methods=['GET'])
def predict(id):
    data_cli = scaler_df[scaler_df.index==int(id)]
    y_prob = model.predict_proba(data_cli)[:, 1][0]
    y_predict = int((y_prob > threshold) * 1)
    response = {"id_": id, 
        "threshold": threshold, 
        "predict": y_predict,
        "probability": y_prob}
    return jsonify(response)

#Interprétation locale
@app.route('/explain/<id>',  methods=['GET'])
def explain_id(id):
    data_cli = scaler_df[scaler_df.index==int(id)]
    shap_values = explain_shap.shap_values(data_cli, check_additivity=False).tolist()
    expected_value = explain_shap.expected_value
    response = {"id_": id,
                "explain_data_id": data_cli.to_dict(),  # renvoie les données pour l'interprétabilité locale
                "shap_values": shap_values, # renvoie les valeurs de Shapely
                "expected_value":  expected_value # renvoie la valeur de base
            }
    return jsonify(response)

#Interprétation globale
@app.route('/explain/',  methods=['GET'])
def explain_all():
    data_1000_cli = scaler_df.sample(n=nb_client_explain_global, random_state=42)
    shap_values = explain_shap.shap_values(data_1000_cli, check_additivity=False).tolist()
    expected_value = explain_shap.expected_value
    response = {"explain_data": data_1000_cli.to_dict(),  # renvoie les données pour l'interprétabilité locale
                "shap_values": shap_values, # renvoie les valeurs de Shapely
                "expected_value": expected_value  # renvoie la valeur de base
            }
    return jsonify(response)
    
if __name__ == '__main__':
    app.run()
