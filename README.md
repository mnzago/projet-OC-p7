# Projet 7 - Openclassroom

Ce projet à pour but de réaliser un Dashboard permettant la visualisation de l'accord ou non du prêt avec l'interprétation globale et locale

Il contient : 
- notebook.ipynb: Le notebook du nettoyage, exploration et mdélisation des données
- app.py: Le programme FLASK
- app_streamlit.py: Le programme Streamlit
- data_cleaned_sample.pickle: Le Datamart après réduction
- model.pickle: Le modèle entrainé
- explainer.pickle: l'explication des variables (Tree.Explainer)
- Procfile: Fichier qui permet à Heroku de savoir qu'on utilise gunicorn pour Flask
- runtime.txt: Fichier informant Heroku de la version de Python utilisée
- requirements.txt: Fichier contenant les librairies utilisées pour le bon fonctionnement de notre Dashboard

# Flask API: 
 
Le fichier app.py contient toutes les requêtes pour obtenir les données nécessaires au Dashboard

Contient les API suivant : 

| HTTP method | API endpoint | Description                       |
| ----------- | -------------| ----------------------------------|
| GET         | client/<id>  | Les données clients               |
| GET         | predict/<id> | La prédiction du client           |
| GET         | explain/<id> | L'interprétation locale du client |
| GET         | explain/     | L'interprétation globale du client|

Nous précisons que nous avons notre seuil (threshold) = 0.7
Et pour l'interprétation globale pour prenons un échantillon de 1000 clients

## Hebergement dans Heroku:

Nous avons besoin de plusieurs fichier pour le bon déroulement de l'hébergement: 
 - `Procfile` qui contient l'initialisation `web: gunicorn api.app:app` (gunicorn est un serveur qui permet d'indiquer à Héroku que Flask sera loger dans une application web)
 - `requirements.txt` qui contient les librairies et leur version utilisées
 - `runtime.txt` qui informe Heroku de la version Python a utilisé 

Comment pousser mes fichiers dans Heroku : 
  - Intaller Heroku CLI
  - Dans le terminal:
  - `heroku login` (il faut se connecter à son compte)
  - git init
  - git add [nom du fichier]
  - git commit -m "commentaire" (permet d'avoir le dernier commentaire lors d'une modification)
  - `heroku create [nom_application] --region eu`
  - `heroku run init`
  - `git push heroku main` (or master)

L'application sera hébergé dans Heroku à l'adresse : `https://nom_application.herokuapp.com/`

## Push des fichiers dans Github : 

 - Dans le terminal:
  - git init
  - git add [nom du fichier]
  - git commit -m "commentaire" (permet d'avoir le dernier commentaire lors d'une modification)
  - git remote add origin https://github.com/EtudiantOC/[nom du repository].git
  - git branch -M main
  - git push -u origin main

# Dashboard Streamlit: 

Le fichier app-streamlit.py contient le code qui permet la réalisation du Dashbord

Mise en lien avec Streamlit Sharing: 
 - Se créer un compte Streamlit Sharing
 - Créer un 'new app'
 - Choisir le repository de Github, le branch et le nom du fichier streamlit
 - Deployer 
 
Pour que les applications FLASK et Streamlit communique dans le fichier Streamlit il faut un URL qui va permettre de récupérer les données dans Flask
URL = https://test-projet-v0.herokuapp.com/
