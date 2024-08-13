import streamlit as st
import pandas as pd
import pickle

# Load pre-trained model and individual label encoders from pickle files
@st.cache_data
def load_model_and_encoders():
    with open('knn_regression_model.pkl', 'rb') as model_file:
        knn = pickle.load(model_file)
    with open('label_encoder_state.pkl', 'rb') as file:
        state_code_encoder = pickle.load(file)
    with open('label_encoder_source.pkl', 'rb') as file:
        source_name_encoder = pickle.load(file)
    with open('label_encoder_tobacco.pkl', 'rb') as file:
        tobacco_encoder = pickle.load(file)
    with open('label_encoder_planid.pkl', 'rb') as file:
        planid_encoder = pickle.load(file)
    return knn, state_code_encoder, source_name_encoder, tobacco_encoder, planid_encoder

# Load model and encoders
knn, state_code_encoder, source_name_encoder, tobacco_encoder, planid_encoder = load_model_and_encoders()

# Streamlit interface
st.title('Health Insurance Premium Prediction using KNN Regression')

st.write("### Predict Premium")

# User inputs for all features
business_year = st.number_input('Business Year', min_value=2000, max_value=2100, value=2023)
state_code = st.text_input('State Code')
issuer_id = st.text_input('Issuer ID')
source_name = st.text_input('Source Name')
version_num = st.number_input('Version Number', min_value=1, value=1)
issuer_id2 = st.text_input('Issuer ID2')
plan_id = st.text_input('Plan ID')
rating_area_id = st.number_input('Rating Area ID', min_value=0)
tobacco = st.selectbox('Tobacco', ['Yes', 'No'])
age = st.number_input('Age', min_value=0, value=30)
row_number = st.number_input('Row Number', min_value=0, value=1)
rate_duration = st.number_input('Rate Duration (in days)', min_value=0, value=30)

# Convert inputs to DataFrame in the correct order
input_data = pd.DataFrame({
    'BusinessYear': [business_year],
    'StateCode': [state_code],
    'IssuerId': [issuer_id],
    'SourceName': [source_name],
    'VersionNum': [version_num],
    'IssuerId2': [issuer_id2],
    'PlanId': [plan_id],
    'RatingAreaId': [rating_area_id],
    'Tobacco': [tobacco],
    'Age': [age],
    'RowNumber': [row_number],
    'RateDuration': [rate_duration]
})

# Encode inputs using the loaded encoders
def safe_transform(column, encoder):
    """Transform the column safely, handling unseen labels."""
    known_classes = set(encoder.classes_)
    column = column.apply(lambda x: x if x in known_classes else known_classes.pop())
    return encoder.transform(column)

try:
    input_data['StateCode'] = safe_transform(input_data['StateCode'], state_code_encoder)
    input_data['Tobacco'] = safe_transform(input_data['Tobacco'], tobacco_encoder)
    input_data['PlanId'] = safe_transform(input_data['PlanId'], planid_encoder)
    input_data['SourceName'] = safe_transform(input_data['SourceName'], source_name_encoder)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Predict
if st.button('Predict Premium'):
    prediction = knn.predict(input_data)
    st.write(f"Predicted Premium: {prediction[0]}")
