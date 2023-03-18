import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

URL = "https://test-projet-v0.herokuapp.com"

st.set_page_config(page_title="Prêt à dépenser", page_icon=":credit_card:", layout="wide")

col1, mid, col2 = st.columns([120,10,25])
with col1:
    st.title("Dashboard : SCORING CREDIT :credit_card:")
with col2:
    st.image('./logo.jpg', width=150)

st.write("Ce dashboard à destination des conseillers permettra d'avoir de manière simple"
" et rapide le score d'octroi d'un crédit pour un client donnée."
" Il contiendra des informations relatives à un client, le score, l'interprétation de ce score "
"ainsi que ses informations comparées à l'ensemble des clients")
st.caption("Pour le projet prendre pour exemple les clients suivants 'vide', '100000', '100002', '100011' ou '331040'")

@st.cache_data()
def get_list_clients():
    """API - load list of clients""" 
    response = requests.get(f'{URL}/clients/')
    return list(response.json())

@st.cache_data()
def data_clients():
    """API - load le dataframe clients""" 
    response = requests.get(f'{URL}/client/')
    return response.json()
 
@st.cache_data(show_spinner=False)
def get_data_from_customer(id):
    response = requests.get(f"{URL}/client/{id}")
    return response.json()

def display_customer_data(element):
    r = get_data_from_customer(element)  # on fait appel à l'API
    st.subheader(f"Voici les données et les résultats du client {element}")
    st.write(pd.DataFrame.from_dict(r, orient='index'))

# On obtient la gauge pour visualiser le refus ou l'accord du prêt
def gauge_plot(probability, threshold):
    value = round(probability, 2) * 100
    percent = 0.1
    # On défini les différents paramètres des sections de la jauge (bornes inf et sup et couleur)
    steps = [([0, max(0, threshold - percent)], (37, 166, 41)),
             ([threshold - percent, max(threshold - percent, threshold + percent)], (235, 159, 27)),
             #([threshold, max(threshold, threshold + percent)], (224, 120, 95)),
             ([threshold + percent, max(threshold + percent, 1)], (242, 40, 26))]

    # On utilise la librairie plotly qui propose des graphiques ultra-paramétrables
    fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=value,
            mode="gauge+number+delta",
            title={'text': f""},
            delta={'reference': threshold * 100,
               'increasing': {'color': "rgb(135, 10, 36)"},
               'decreasing': {'color': "rgb(30, 166, 93)"}},
            gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [step[0][0] * 100, step[0][1] * 100], 'color': f'rgb{str(step[1])}'}
                         for step in steps if step[0][0] != step[0][1]],
               'threshold': {'line': {'color': "black", 'width': 8}, 'thickness': 0.75, 'value': threshold * 100}}))

    return fig

@st.cache_data(show_spinner=False)
def get_predict_from_customer(id):
    response = requests.get(f"{URL}/predict/{id}")
    return response.json()

def prediction_cli(element):
    result = get_predict_from_customer(element)
    data_result = pd.DataFrame.from_dict(result, orient='index')
    if data_result[0]['predict'] == 1:
        st.subheader('Le prêt est refusé!')
    else:
        st.subheader('Le prêt est accordé!')
    st.write(f"La probabilité que le client {element} ne rembourse pas son prêt est de: {(round(data_result[0]['probability'],2))*100} %"
    f", soit {(round(data_result[0]['probability'],2))*100 - (round(data_result[0]['threshold'],2))*100} point(s)"
    " par rapport à notre seuil optimal")
    gauge = gauge_plot(data_result[0]['probability'], data_result[0]['threshold'])
    st.write(gauge)

# On récupère les résultats pour afficher l'interprétation globale et locale

 # Interprétation globale
@st.cache_data(show_spinner=False)
def get_explain_from_all():
    response = requests.get(f"{URL}/explain/")
    return response.json()

def explain_global():
    result = get_explain_from_all()
    data = pd.DataFrame.from_dict(result['explain_data'], orient='index').T
    shap_values = pd.DataFrame(result['shap_values'])
    st.subheader("Interprétation globale")
    st.write("Le graphique ci-dessous nous donne les 10 variables qui contribuent le plus à la décision d'obtention de manière globale sur 1000 clients")
    st.markdown(
        """<style>
        .streamlit-expanderContent {background:white;}
        </style>""",
        unsafe_allow_html=True,
    )  # Un peu de style CSS pour garantir un fond blanc sur les expander (graphiques SHAP mieux intégrés !)
    st_shap(shap.summary_plot(np.array(shap_values), data, plot_type="bar", max_display = 10))
    ##st.write("Si la variable est en rouge: variable qui favorise le refus")
    ##st.write("Si la variable est en bleu: variable qui favorise l'accord'")
    st.markdown(
        """<style>
        .streamlit-expanderContent {background:white;}
        </style>""",
        unsafe_allow_html=True,
    )
    st_shap(shap.summary_plot(np.array(shap_values), data, plot_type="violin", max_display = 10))

# Interprétation locale
@st.cache_data(show_spinner=False)
def get_explain_from_customer(id):
    response = requests.get(f"{URL}/explain/{id}")
    return response.json()

def explain_local(element):
    result = get_explain_from_customer(element)
    data_cli = pd.DataFrame.from_dict(result['explain_data_id'], orient='index').T
    shap_values = pd.DataFrame(result['shap_values'])
    expected_value = result['expected_value']

    interp = shap.Explanation(np.array(shap_values)[0],
                        base_values=expected_value,
                        feature_names=data_cli.columns,
                        data=data_cli.values[0])
    st.subheader("Interprétation locale")
    st.write("Le graphique ci-dessous nous donne les variables qui contribuent le plus à la décision d'obtention du prêt pour ce client")
    st.write("Si la variable est en rouge: variable qui favorise le refus")
    st.write("Si la variable est en bleu: variable qui favorise l'accord'")
    st.markdown(
        """<style>
        .streamlit-expanderContent {background:white;}
        </style>""",
        unsafe_allow_html=True,
    )  # Un peu de style CSS pour garantir un fond blanc sur les expander (graphiques SHAP mieux intégrés !)
    st_shap(shap.plots.waterfall(interp))

# Situation du client par rapport à un échantillon de 1000 clients
def feature_boxplot(data, feature, value, customer_id):
    fig = px.box(data, x=feature)
    fig.add_vline(x=value,
                  line_color="rgb(32,32, 32)",
                  annotation_text=f'Client: {customer_id}'
                  )
    return fig

def display_interp_feature(element):
    result_global = get_explain_from_all()
    data = pd.DataFrame.from_dict(result_global['explain_data'], orient='index').T

    result_local = get_explain_from_customer(element)
    data_cli = pd.DataFrame.from_dict(result_local['explain_data_id'], orient='index').T

    features = pd.DataFrame.from_dict(data).columns.tolist()
    variable = st.sidebar.selectbox("Choisir une variable", features,
                     help='Visualiser le positionnement du client par rapport à cette variable.') 
    fig = feature_boxplot(data=data,
                        feature=variable,
                        value=data_cli[variable].values[0],
                        customer_id=element)
    st.subheader("Boxplot: Représentation du client dans un échantillon de 1000 clients")
    st.plotly_chart(figure_or_data=fig, use_container_width=True)

#---- Fonction main     

client = st.sidebar.text_input(label="Saisir l'identifiant d'un client :bust_in_silhouette:")
list_cli = get_list_clients()
if client == '':
    st.subheader(f"Entrer un numéro de clients")
elif int(client) not in list_cli:
    st.subheader(f"Le client est inconnu")
else:
    display_customer_data(client)
    prediction_cli(client)
    explain_local(client)
    explain_global()
    display_interp_feature(client)