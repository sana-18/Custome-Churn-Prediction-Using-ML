import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Charger le fichier CSV (remplacez "votre_fichier.csv" par le chemin de votre fichier CSV)
# Assurez-vous que le fichier est dans le même répertoire que votre script Streamlit ou fournissez le chemin complet.
csv_file_path = "telecom_customer_churn.csv"
data = pd.read_csv(csv_file_path)
numeric_features = [feature for feature in data.columns if data[feature].dtype != "object"]
categorical_features = [feature for feature in data.columns if data[feature].dtype == "object"]
dataset_copy = data.copy()


def clean_dataset(dataset):
    # Convert 'Gender' to binary values
    dataset['Gender'] = dataset['Gender'].apply(lambda row: 1 if row == "Female" else 0)

    # Convert binary columns to 0 and 1
    binary_column = dataset.drop('Gender', axis=1).nunique()[dataset.drop('Gender', axis=1).nunique() < 3].keys().to_list()
    for column in binary_column:
        dataset[column] = dataset[column].apply(lambda row: 1 if row == 'Yes' else 0)

    # Convert remaining categorical variables to numeric codes
    categorical_columns = dataset.select_dtypes(include='object').columns
    remaining_cat_vars = dataset[categorical_columns].nunique()[dataset[categorical_columns].nunique() > 2].keys().to_list()

    for column in remaining_cat_vars:
        dataset[column] = dataset[column].astype('category').cat.codes
        dataset[column] = dataset[column].replace(-1, np.nan)

    return dataset

def clean_x(dataset):
    # Exclure la colonne 'Gender' si elle est présente
    if isinstance(dataset, pd.DataFrame) and 'Gender' in dataset.columns:
        dataset.drop('Gender', axis=1, inplace=True)

    # Convert binary columns to 0 and 1
    binary_columns = dataset.nunique()[dataset.nunique() < 3].keys().to_list()
    for column in binary_columns:
        dataset[column] = dataset[column].apply(lambda row: 1 if row == 'Yes' else 0)

    # Convert remaining categorical variables to numeric codes
    categorical_columns = dataset.select_dtypes(include='object').columns
    remaining_cat_vars = dataset[categorical_columns].nunique()[dataset[categorical_columns].nunique() > 2].keys().to_list()

    for column in remaining_cat_vars:
        dataset[column] = dataset[column].astype('category').cat.codes
        dataset[column] = dataset[column].replace(-1, np.nan)

    return dataset



def clean_Y(dataset):
    dataset.replace({'Churned': 0, 'Stayed': 2, 'Joined': 1}, inplace=True)
    return dataset
   

def meanNan(data):
    data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    return data

def total_charge(dataset):
    dataset['Total Charges']=np.sqrt(dataset['Total Charges'])
    return dataset
from scipy.stats import zscore
def aberrantes(data):
    # Seuil pour les valeurs aberrantes
    seuil = 1  # Ajustez le seuil selon vos besoins

    # Calculer les Z-scores
    z_scores = zscore(data['Number of Dependents'])

    # Limiter les valeurs au-delà du seuil
    valeurs_limiter = np.clip(data['Number of Dependents'], -seuil, seuil)

    # Remplacer la colonne originale par les valeurs limitées
    data['Number of Dependents'] = valeurs_limiter
    return data
def label_encode_features(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include='object').columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data


dataset_copy.drop(['Customer ID','Total Refunds','Zip Code','Latitude', 'Longitude','Churn Category', 'Churn Reason','Offer'],axis=1,inplace=True)
dataset_copy.drop(['City','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue','Number of Referrals'],inplace=True,axis=1)
X = dataset_copy.drop(['Customer Status', 'Gender'], axis=1)
Y = dataset_copy['Customer Status']
st.set_page_config(
    page_title="telecom_customer_churn",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
def prediction(user_input):
    # Clean user input
    clean_Y(Y)
    clean_x(X)
    user_input_cleaned = clean_x(user_input.copy())
    user_input_cleaned = clean_Y(user_input_cleaned)

    # Ensure the order of columns matches the order during training
    user_input_cleaned = user_input_cleaned[X.columns]

    # Prepare the model input
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1111, stratify=Y)

    # Assume X_train and Y_train are your training data
    model = XGBClassifier(random_state=42)
    model.fit(X_train, Y_train)

    # Ensure the feature names match the order in X_train
    user_input_cleaned = user_input_cleaned[X_train.columns]

    # Make a prediction for the new customer
    prediction_result = model.predict(user_input_cleaned)

    return prediction_result


import base64


def sidebar_bg(side_bg):
    

    side_bg_ext = 'jpg'

    st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: cover;
          
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'images/menu.jpg'
sidebar_bg(side_bg)


# Styles pour les éléments
st.markdown(
    """
    <style>
   
        custom_css {
            background-color: #f0f0f0;
        }
        
        .sidebar .sidebar-content {
            background-color: #333;
            color: #fff;
        }
        body {
            font-size: 40px;
        }
        .main {
            background-color: #fff;
            padding: 20px;
            margin: 20px;
            border-radius: 10px;
            box-shadow:0 -4px 8px  5px #191970;
        }
        .custom-title {
            font-size: 30px;
            color:#191970;
            text-align: center;
        }
        
        .dataset-header {
            font-size: 24px;
            color: #000b80;
            margin-top: 20px;
        }
        .info-text {
            font-size: 18px;
            color: #555;
            margin-top: 10px;
        }
        .div{
        font-size: 24px;
        margin-top:20px;
        padding-bottom:30px;
        }
        .joined{
        text-align: center;  # Alignement horizontal au centre
     margin-top: 20px;    # Espacement en haut
        }
        
        
        
    </style>
    """,
    unsafe_allow_html=True
)


# Appliquer le style CSS à l'application Streamlit





# Options du menu

option = st.sidebar.selectbox(
    'Sélectionnez une option',
    ('Accueil', 'Dataset', 'Visualisation des données', 'Manipulation de données','Analyse de répartition','Prédiction')
)

# Contenu en fonction de l'option sélectionnée
if option == 'Accueil':
    
    st.markdown("<div class='custom-title'>Prévision de désabonnement de la clientèle dans le secteur de télécommunication</div>", unsafe_allow_html=True)
    st.markdown("<div class='div'>Bienvenue au démonstration du projet apprentissage automatique!</div>", unsafe_allow_html=True)
    
    image_path = "images/acceuil1.jpg"  # Replace with the actual path to your image file
    st.image(image_path, use_column_width=True)
elif option == 'Dataset':
    st.markdown("<div class='custom-title'>Compréhension de données</div>", unsafe_allow_html=True)
    
    
    st.markdown("<div class='dataset-header'>Voici les cinq premières lignes du dataset :</div>", unsafe_allow_html=True)
    st.write(data.head())  # Affiche les cinq premières lignes du dataset

    # Afficher les informations supplémentaires sur le dataset
    st.markdown("<div class='dataset-header'>Informations sur le Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Shape du dataset : </div>", unsafe_allow_html=True)
    st.text(data.shape)
   
    st.markdown("<div class='info-text'>Description du dataset :</div>", unsafe_allow_html=True)
    st.text(data.describe())
    
elif option == 'Visualisation des données':
    
    
    type_variable_option = st.sidebar.radio(
        
        'Sélectionnez un type de variable',
        ('Données Numériques', 'Catégorisation de données', 'Valeurs manquantes', 'valeurs abberantes')
    )

    if type_variable_option == 'Données Numériques':
        st.markdown("<div class='custom-title'>Visualisation des valeurs numériques</div>", unsafe_allow_html=True)
        selected_variable = st.selectbox('Choisissez une variable numérique', data.select_dtypes(include='number').columns)
        st.markdown(f"<div class='dataset-header'>Distribution de {selected_variable}</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        sns.barplot(data=data, x=data[selected_variable].value_counts().index, y=data[selected_variable].value_counts(), ax=ax, color='#3498db')
        st.pyplot(fig)

    elif type_variable_option == 'Catégorisation de données':
        st.markdown("<div class='custom-title'>Visualisation des valeurs catégorielles </div>", unsafe_allow_html=True)
        selected_variable = st.selectbox('Choisissez une variable catégorielle', data.select_dtypes(include='object').columns)
        st.markdown(f"<div class='dataset-header'>Distribution de {selected_variable}</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=data, x=data[selected_variable].value_counts().index, y=data[selected_variable].value_counts(), ax=ax, color='#3498db')
        st.pyplot(fig)
    elif type_variable_option == 'Valeurs manquantes':
        st.markdown("<div class='custom-title'>Visualisation de données manquantes </div>", unsafe_allow_html=True)
        msno.matrix(data)
        st.pyplot()  # Display the missingno matrix plot
    elif type_variable_option == 'valeurs abberantes':
        st.markdown("<div class='custom-title'>visualisation des valeurs abberantes avec Boxplot</div>", unsafe_allow_html=True)
        
        numeric_features = [feature for feature in data.columns if data[feature].dtype == "float64" or data[feature].dtype == "int64"]
        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        for i, subplot in zip(numeric_features, ax.flatten()):
            sns.boxplot(x='Customer Status', y=i, data=data, ax=subplot)

        st.pyplot(fig)



elif option == 'Manipulation de données':
    type_variable_option = st.sidebar.radio(
        'Sélectionnez ',
        ('Supprimez les colonnes inutiles','La variable Total charges', 'Encodage des variables categorielles', 'Traitement des NAN','Corrélation','Detection des valeurs abberantes')
    )
    if type_variable_option == 'Supprimez les colonnes inutiles':
        st.markdown("<div class='custom-title'>Évolution du Jeu de Données après Suppression de Colonnes </div>", unsafe_allow_html=True)
        
      
        st.markdown("<div class='dataset-header'>Voici les cinq premières lignes du dataset :</div>", unsafe_allow_html=True)
        st.write(dataset_copy.head())  # Affiche les cinq premières lignes du dataset

        # Afficher les informations supplémentaires sur le dataset
        st.markdown("<div class='dataset-header'>Informations sur le Dataset:</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Shape du dataset: </div>", unsafe_allow_html=True)
        st.text(dataset_copy.shape)
    elif type_variable_option == 'Encodage des variables categorielles':
        clean_dataset(dataset_copy)
       
        st.markdown("<div class='dataset-header'>Voici les cinq premières lignes du dataset:</div>", unsafe_allow_html=True)
        st.write(dataset_copy.head())  # Affiche les cinq premières lignes du dataset
        
    elif type_variable_option == 'Traitement des NAN':
        clean_dataset(dataset_copy)
        st.write("NaN values count avant le remplissage par le moyenne:")
        st.write(dataset_copy.isna().sum())
        dataset_copy = meanNan(dataset_copy)
        st.write("NaN values count after meanNan:")
        st.write(dataset_copy.isna().sum())
        st.write(dataset_copy.head())
    elif type_variable_option == 'Corrélation':
        clean_dataset(dataset_copy)
        meanNan(dataset_copy)
        
        correlation_matrix = dataset_copy.corr()
        churn_correlation = correlation_matrix['Customer Status'].sort_values(ascending=False)
        st.write(churn_correlation)
        plt.figure(figsize=(16, 12))
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Heatmap")
    
    # Utilisez st.pyplot pour afficher la figure dans l'application Streamlit
        st.pyplot()
    elif type_variable_option == 'La variable Total charges':
        st.markdown("<div class='dataset-header'>Distribution de Total Charges avant transformation</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
       
        sns.histplot(dataset_copy['Total Charges'], kde=True, ax=ax)
        st.pyplot(fig)
        st.write(dataset_copy['Total Charges'].skew())

        st.markdown("<div class='dataset-header'>Distribution de Total Charges après transformation</div>", unsafe_allow_html=True)
        total_charge(dataset_copy)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.histplot(dataset_copy['Total Charges'], kde=True, ax=ax)
        st.pyplot(fig)
        st.write(dataset_copy['Total Charges'].skew())

    elif type_variable_option == 'Detection des valeurs abberantes':
        st.markdown("<div class='custom-title'>visualisation des valeurs abberantes avant le traitement</div>", unsafe_allow_html=True)
        numerical_columns = ['Age', 'Number of Dependents', 'Tenure in Months', 'Monthly Charge', 'Total Charges']
        st.write(numerical_columns)
        num_cols = len(numerical_columns)  # 3

        # Calculer le nombre de lignes pour organiser les sous-graphiques de manière à avoir au plus 3 colonnes par ligne
        num_rows = (num_cols - 1) // 3 + 1  # 4

        # Code de traçage
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 8))

        # Aplatir les axes des sous-graphiques
        ax = ax.flatten()

        # Tracer uniquement pour les colonnes numériques spécifiées
        for i, subplot in zip(numerical_columns, ax):
            
            sns.boxplot(x='Customer Status', y=i, data=dataset_copy, ax=subplot)

        # Supprimer tous les sous-graphiques vides restants
        for empty_subplot in ax[num_cols:]:
            empty_subplot.axis('off')

        # Utilisez st.pyplot pour afficher la figure dans l'application Streamlit
        st.pyplot(fig)
        aberrantes(dataset_copy)
        # Code de tracé
        fig, ax = plt.subplots(figsize=(8, 6))

        

# Tracer le boxplot pour 'Number of Dependents'
        st.markdown("<div class='custom-title'>visualisation des valeurs abberantes aprés le traitement</div>", unsafe_allow_html=True)
        sns.boxplot(x='Customer Status', y='Number of Dependents', data=dataset_copy, ax=ax)
        plt.title('Boxplot du Nombre de Personnes à Charge par Statut du Client')
        st.pyplot(fig)
# ...

# ...

elif option == 'Analyse de répartition':
    # Ajoutez ici le code spécifique à l'étape d'analyse de répartition
    type_variable_option = st.sidebar.radio(
        'Sélectionnez ',
        ('Traitement du Déséquilibre des Données',))

    if type_variable_option == 'Traitement du Déséquilibre des Données':
        st.markdown("<div class='custom-title'>Visualisation du Déséquilibre des Données</div>", unsafe_allow_html=True)
        clean_x(X)
        clean_Y(Y)
        meanNan(X)
        total_charge(X)
        aberrantes(X)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Pie Chart
        labels = ['Stayed', 'Churned', 'Joined']
        pie_colors = ['skyblue', '#ff4d4d', 'gold']
        ax1.pie(x=dataset_copy['Customer Status'].value_counts(), labels=labels, autopct='%1.1f%%', shadow=True, colors=pie_colors)
        ax1.set_title('Customer Status Distribution Avant Oversampling (Pie Chart)')

        # Countplot
        sns.countplot(x=Y, ax=ax2)
        ax2.set_title('Customer Status Distribution Avant Oversampling (Countplot)')

        st.pyplot(fig)
        if st.button('Oversample Data'):
            st.markdown("<div class='custom-title'>Visualisation du Déséquilibre aprés le traitement</div>", unsafe_allow_html=True)
            from imblearn.over_sampling import BorderlineSMOTE
            from sklearn.impute import SimpleImputer 
            imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
            X_imputed = imputer.fit_transform(X)
        # Apply oversampling
            oversample = BorderlineSMOTE()
            X_resampled, Y_resampled = oversample.fit_resample(X_imputed, Y)

        # Create a DataFrame for the resampled data
            resampled_data = pd.DataFrame({'Customer Status': Y_resampled})

        # Plot countplot after oversampling using Streamlit's st.bar_chart
            fig, ax = plt.subplots()
            sns.countplot(x=resampled_data['Customer Status'], ax=ax, palette=['skyblue', '#ff4d4d', 'gold'])
            ax.set_title('Customer Status Distribution After Oversampling')
            ax.set_xlabel('Customer Status')
            ax.set_ylabel('Count')
            st.pyplot(fig)

    

# ...


    
            
       
    
elif option == 'Prédiction':
    st.markdown("<div class='custom-title'>Test de prédiction</div>", unsafe_allow_html=True)
   
    st.markdown("<div style='display: flex; flex-direction: row; justify-content: space-between; font-size:30px;'>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

# Place the radio buttons in the columns
    with col1:
        married = st.radio("Married", ["Yes", "No"])
   
    with col2:
        phone_service = st.radio("Phone Service", ["Yes", "No"])
    with col3:

        multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])
        
      
    
    st.markdown("<div style='display: flex; flex-direction: row; justify-content: space-between;'>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

# Place the radio buttons in the columns
    with col1:
        internet_service = st.radio("Internet Service", ["yes", "No"])
   
    with col2:
        online_security = st.radio("Online Security", ["Yes", "No"])
    with col3:

        online_backup = st.radio("Online Backup", ["Yes", "No"])
        
        
        
    st.markdown("<div style='display: flex; flex-direction: row; justify-content: space-between;'>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

# Place the radio buttons in the columns
    with col1:
        device_protection = st.radio("Device Protection Plan", ["Yes", "No"])
   
    with col2:
        premium_tech_support = st.radio("Premium Tech Support", ["Yes", "No"])
    with col3:

        unlimited_data = st.radio("Unlimited Data", ["Yes", "No"])
        
        
        
    st.markdown("<div style='display: flex; flex-direction: row; justify-content: space-between;'>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

# Place the radio buttons in the columns
    with col1:
        streaming_tv = st.radio("Streaming TV", ["Yes", "No"])
   
    with col2:
        streaming_movies = st.radio("Streaming Movies", ["Yes", "No"])
    with col3:

        streaming_music = st.radio("Streaming Music", ["Yes", "No"])
        
        
        
        
    st.markdown("<div style='display: flex; flex-direction: row; justify-content: space-between;'>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

# Place the radio buttons in the columns
    with col1:
        internet_type = st.radio("Internet Type", ["DSL", "Fiber Optic", "No"])
   
    with col2:
        contract = st.radio("Contrat", ["Month-to-Month", "One Year", "Two Year"])
    with col3:

        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
        
        
        
        

    st.markdown("</div>", unsafe_allow_html=True)
    
  
    
    
    
    st.markdown("<div style='display: flex; justify-content: center; margin-left: 600px;'>", unsafe_allow_html=True)
    payment_method = st.radio("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer (Automatic)", "Credit Card (Automatic)"])
    st.markdown("</div>", unsafe_allow_html=True)


    # Entrées pour les caractéristiques numériques du client
    age = st.slider("Âge", min_value=int(X['Age'].min()), max_value=int(X['Age'].max()), value=int(X['Age'].mean()))
    num_dependents = st.slider("Number of Dependents", min_value=int(X['Number of Dependents'].min()), max_value=int(X['Number of Dependents'].max()), value=int(X['Number of Dependents'].mean()))
    tenure_months = st.slider("Tenure in Months", min_value=int(X['Tenure in Months'].min()), max_value=int(X['Tenure in Months'].max()), value=int(X['Tenure in Months'].mean()))
    monthly_charge = st.slider("Monthly Charge", min_value=float(X['Monthly Charge'].min()), max_value=float(X['Monthly Charge'].max()), value=float(X['Monthly Charge'].mean()))
    total_charges = st.slider("Total Charges", min_value=float(X['Total Charges'].min()), max_value=float(X['Total Charges'].max()), value=float(X['Total Charges'].mean()))

    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        # Préparer les caractéristiques catégorielles pour la prédiction
        user_inputs = {
    'Married': married,
    'Phone Service': phone_service,
    'Multiple Lines': multiple_lines,
    'Internet Service': internet_service,
    'Internet Type': internet_type,
    'Online Security': online_security,
    'Online Backup': online_backup,
    'Device Protection Plan': device_protection,
    'Premium Tech Support': premium_tech_support,
    'Streaming TV': streaming_tv,
    'Streaming Movies': streaming_movies,
    'Streaming Music': streaming_music,
    'Unlimited Data': unlimited_data,
    'Contract': contract,
    'Paperless Billing': paperless_billing,
    'Payment Method': payment_method,
    'Age': age,
    'Number of Dependents': num_dependents,
    'Tenure in Months': tenure_months,
    'Monthly Charge': monthly_charge,
    'Total Charges': total_charges,
}

# Convert the dictionary values to a NumPy array
        user_input_array = np.array([list(user_inputs.values())]).reshape(1, -1)

# Create a DataFrame with an index
        user_input_df = pd.DataFrame(user_input_array, columns=user_inputs.keys())
        
    # Now you can use user_input_df for prediction
        prediction_result = prediction(user_input_df)

    # Display the prediction result
        
        if prediction_result == 1:
            st.markdown("<div class='joined'>le client nouveau ici!</dv>", unsafe_allow_html=True)
           
            st.image('images/joined.jpg', use_column_width=True)
        elif prediction_result == 2:
            st.markdown("<div class='joined'>le client va rester!</div>", unsafe_allow_html=True)
            st.image('images/stayed.jpg', use_column_width=True)
             
        else: 
            st.markdown("<div class='joined'>Le client va se désabonner !</div>", unsafe_allow_html=True)
            st.image('images/churned.jpg', use_column_width=True)

  
    

   
