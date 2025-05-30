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

# Load the CSV file 
# Make sure the file is in the same directory as your Streamlit script or provide the full path.
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
    # Exclude the 'Gender' column if it is present
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
    # Threshold for outliers
    seuil = 1  # Adjust the threshold according to your needs

    # Calculate Z-scores
    z_scores = zscore(data['Number of Dependents'])

    # Limit values beyond the threshold
    valeurs_limiter = np.clip(data['Number of Dependents'], -seuil, seuil)

    # Replace the original column with the limited values
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


# Styles for elements
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
        text-align: center;  # Horizontal center alignment
     margin-top: 20px;    # Top spacing
        }
        
        
        
    </style>
    """,
    unsafe_allow_html=True
)

# Menu options

option = st.sidebar.selectbox(
    'Select an option',
    ('Home', 'Dataset', 'Data Visualization', 'Data Manipulation','Distribution Analysis','Prediction')
)

# Content based on selected option
if option == 'Home':
    
    st.markdown("<div class='custom-title'>Customer Churn Prediction in Telecommunications Sector</div>", unsafe_allow_html=True)
    st.markdown("<div class='div'>Welcome to the machine learning project demonstration!</div>", unsafe_allow_html=True)
    
    image_path = "images/acceuil1.jpg"  # Replace with the actual path to your image file
    st.image(image_path, use_column_width=True)
elif option == 'Dataset':
    st.markdown("<div class='custom-title'>Data Understanding</div>", unsafe_allow_html=True)
    
    
    st.markdown("<div class='dataset-header'>Here are the first five rows of the dataset:</div>", unsafe_allow_html=True)
    st.write(data.head())  # Display the first five rows of the dataset

    # Display additional information about the dataset
    st.markdown("<div class='dataset-header'>Dataset Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Dataset shape:</div>", unsafe_allow_html=True)
    st.text(data.shape)
   
    st.markdown("<div class='info-text'>Dataset description:</div>", unsafe_allow_html=True)
    st.text(data.describe())
    
elif option == 'Data Visualization':
    
    
    type_variable_option = st.sidebar.radio(
        
        'Select a variable type',
        ('Numeric Data', 'Data Categorization', 'Missing Values', 'Outliers')
    )

    if type_variable_option == 'Numeric Data':
        st.markdown("<div class='custom-title'>Numeric Values Visualization</div>", unsafe_allow_html=True)
        selected_variable = st.selectbox('Choose a numeric variable', data.select_dtypes(include='number').columns)
        st.markdown(f"<div class='dataset-header'>Distribution of {selected_variable}</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        sns.barplot(data=data, x=data[selected_variable].value_counts().index, y=data[selected_variable].value_counts(), ax=ax, color='#3498db')
        st.pyplot(fig)

    elif type_variable_option == 'Data Categorization':
        st.markdown("<div class='custom-title'>Categorical Values Visualization</div>", unsafe_allow_html=True)
        selected_variable = st.selectbox('Choose a categorical variable', data.select_dtypes(include='object').columns)
        st.markdown(f"<div class='dataset-header'>Distribution of {selected_variable}</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=data, x=data[selected_variable].value_counts().index, y=data[selected_variable].value_counts(), ax=ax, color='#3498db')
        st.pyplot(fig)
    elif type_variable_option == 'Missing Values':
        st.markdown("<div class='custom-title'>Missing Data Visualization</div>", unsafe_allow_html=True)
        msno.matrix(data)
        st.pyplot()  # Display the missingno matrix plot
    elif type_variable_option == 'Outliers':
        st.markdown("<div class='custom-title'>Outliers Visualization with Boxplot</div>", unsafe_allow_html=True)
        
        numeric_features = [feature for feature in data.columns if data[feature].dtype == "float64" or data[feature].dtype == "int64"]
        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        for i, subplot in zip(numeric_features, ax.flatten()):
            sns.boxplot(x='Customer Status', y=i, data=data, ax=subplot)

        st.pyplot(fig)



elif option == 'Data Manipulation':
    type_variable_option = st.sidebar.radio(
        'Select',
        ('Remove Unnecessary Columns','The Total Charges Variable', 'Categorical Variables Encoding', 'NAN Treatment','Correlation','Outliers Detection')
    )
    if type_variable_option == 'Remove Unnecessary Columns':
        st.markdown("<div class='custom-title'>Dataset Evolution after Columns Removal</div>", unsafe_allow_html=True)
        
      
        st.markdown("<div class='dataset-header'>Here are the first five rows of the dataset:</div>", unsafe_allow_html=True)
        st.write(dataset_copy.head())  # Display the first five rows of the dataset

        # Display additional information about the dataset
        st.markdown("<div class='dataset-header'>Dataset Information:</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Dataset shape:</div>", unsafe_allow_html=True)
        st.text(dataset_copy.shape)
    elif type_variable_option == 'Categorical Variables Encoding':
        clean_dataset(dataset_copy)
       
        st.markdown("<div class='dataset-header'>Here are the first five rows of the dataset:</div>", unsafe_allow_html=True)
        st.write(dataset_copy.head())  # Display the first five rows of the dataset
        
    elif type_variable_option == 'NAN Treatment':
        clean_dataset(dataset_copy)
        st.write("NaN values count before filling with mean:")
        st.write(dataset_copy.isna().sum())
        dataset_copy = meanNan(dataset_copy)
        st.write("NaN values count after meanNan:")
        st.write(dataset_copy.isna().sum())
        st.write(dataset_copy.head())
    elif type_variable_option == 'Correlation':
        clean_dataset(dataset_copy)
        meanNan(dataset_copy)
        
        correlation_matrix = dataset_copy.corr()
        churn_correlation = correlation_matrix['Customer Status'].sort_values(ascending=False)
        st.write(churn_correlation)
        plt.figure(figsize=(16, 12))
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Heatmap")
    
    # Use st.pyplot to display the figure in the Streamlit application
        st.pyplot()
    elif type_variable_option == 'The Total Charges Variable':
        st.markdown("<div class='dataset-header'>Total Charges distribution before transformation</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
       
        sns.histplot(dataset_copy['Total Charges'], kde=True, ax=ax)
        st.pyplot(fig)
        st.write(dataset_copy['Total Charges'].skew())

        st.markdown("<div class='dataset-header'>Total Charges distribution after transformation</div>", unsafe_allow_html=True)
        total_charge(dataset_copy)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.histplot(dataset_copy['Total Charges'], kde=True, ax=ax)
        st.pyplot(fig)
        st.write(dataset_copy['Total Charges'].skew())

    elif type_variable_option == 'Outliers Detection':
        st.markdown("<div class='custom-title'>Outliers visualization before treatment</div>", unsafe_allow_html=True)
        numerical_columns = ['Age', 'Number of Dependents', 'Tenure in Months', 'Monthly Charge', 'Total Charges']
        st.write(numerical_columns)
        num_cols = len(numerical_columns)  # 3

        # Calculate the number of rows to organize subplots to have at most 3 columns per row
        num_rows = (num_cols - 1) // 3 + 1  # 4

        # Plotting code
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 8))

        # Flatten subplot axes
        ax = ax.flatten()

        # Plot only for the specified numeric columns
        for i, subplot in zip(numerical_columns, ax):
            
            sns.boxplot(x='Customer Status', y=i, data=dataset_copy, ax=subplot)

        # Remove all remaining empty subplots
        for empty_subplot in ax[num_cols:]:
            empty_subplot.axis('off')

        # Use st.pyplot to display the figure in the Streamlit application
        st.pyplot(fig)
        aberrantes(dataset_copy)
        # Plotting code
        fig, ax = plt.subplots(figsize=(8, 6))

        

# Plot the boxplot for 'Number of Dependents'
        st.markdown("<div class='custom-title'>Outliers visualization after treatment</div>", unsafe_allow_html=True)
        sns.boxplot(x='Customer Status', y='Number of Dependents', data=dataset_copy, ax=ax)
        plt.title('Boxplot of Number of Dependents by Customer Status')
        st.pyplot(fig)

elif option == 'Distribution Analysis':
    # Add here the specific code for the distribution analysis step
    type_variable_option = st.sidebar.radio(
        'Select',
        ('Data Imbalance Treatment',))

    if type_variable_option == 'Data Imbalance Treatment':
        st.markdown("<div class='custom-title'>Data Imbalance Visualization</div>", unsafe_allow_html=True)
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
        ax1.set_title('Customer Status Distribution Before Oversampling (Pie Chart)')

        # Countplot
        sns.countplot(x=Y, ax=ax2)
        ax2.set_title('Customer Status Distribution Before Oversampling (Countplot)')

        st.pyplot(fig)
        if st.button('Oversample Data'):
            st.markdown("<div class='custom-title'>Imbalance visualization after treatment</div>", unsafe_allow_html=True)
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
    
elif option == 'Prediction':
    st.markdown("<div class='custom-title'>Prediction Test</div>", unsafe_allow_html=True)
   
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
        contract = st.radio("Contract", ["Month-to-Month", "One Year", "Two Year"])
    with col3:

        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
        
        
        
        

    st.markdown("</div>", unsafe_allow_html=True)
    
  
    
    
    
    st.markdown("<div style='display: flex; justify-content: center; margin-left: 600px;'>", unsafe_allow_html=True)
    payment_method = st.radio("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer (Automatic)", "Credit Card (Automatic)"])
    st.markdown("</div>", unsafe_allow_html=True)


    # Inputs for customer numeric characteristics
    age = st.slider("Age", min_value=int(X['Age'].min()), max_value=int(X['Age'].max()), value=int(X['Age'].mean()))
    num_dependents = st.slider("Number of Dependents", min_value=int(X['Number of Dependents'].min()), max_value=int(X['Number of Dependents'].max()), value=int(X['Number of Dependents'].mean()))
    tenure_months = st.slider("Tenure in Months", min_value=int(X['Tenure in Months'].min()), max_value=int(X['Tenure in Months'].max()), value=int(X['Tenure in Months'].mean()))
    monthly_charge = st.slider("Monthly Charge", min_value=float(X['Monthly Charge'].min()), max_value=float(X['Monthly Charge'].max()), value=float(X['Monthly Charge'].mean()))
    total_charges = st.slider("Total Charges", min_value=float(X['Total Charges'].min()), max_value=float(X['Total Charges'].max()), value=float(X['Total Charges'].mean()))

    # Button to perform prediction
    if st.button("Predict"):
        # Prepare categorical features for prediction
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
            st.markdown("<div class='joined'>The customer is new here!</dv>", unsafe_allow_html=True)
           
            st.image('images/joined.jpg', use_column_width=True)
        elif prediction_result == 2:
            st.markdown("<div class='joined'>The customer will stay!</div>", unsafe_allow_html=True)
            st.image('images/stayed.jpg', use_column_width=True)
             
        else: 
            st.markdown("<div class='joined'>The customer will churn!</div>", unsafe_allow_html=True)
            st.image('images/churned.jpg', use_column_width=True)

  
    

   
