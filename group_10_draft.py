# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
model_path = os.path.join(os.path.dirname(__file__), "diagnosis_trained_test.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load model and scaler
with open(model_path, "rb") as file:
    model_data = pickle.load(file)

loaded_model = model_data["model"]
y_test = model_data["y_test"]
y_predict_test = model_data["y_predict_test"]
classification_rep = model_data["classification_report"]
conf_matrix = model_data["confusion_matrix"]

# Load the scaler separately
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Function for diabetes prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    standardized_data = scaler.transform(input_data_as_numpy_array)
    prediction = loaded_model.predict(standardized_data)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'


def main():
    st.title('🩺 Diabetes Prediction App')
    st.markdown("### A user-friendly tool to predict diabetes based on health parameters.")
    st.subheader("📊 Understanding the Input Variables")
    st.markdown("This model uses health indicators from a Kaggle dataset to predict diabetes:")
    st.markdown("- **Pregnancies**: Number of times pregnant")
    st.markdown("- **Glucose**: Plasma glucose concentration")
    st.markdown("- **BloodPressure**: Diastolic blood pressure (mm Hg)")
    st.markdown("- **BMI**: Body Mass Index")
    st.markdown("- **DiabetesPedigreeFunction**: Diabetes likelihood based on family history")
    st.markdown("- **Age**: Age in years")
    st.subheader("📝 Enter Your Health Details")
    Pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=17, value=1)
    Glucose = st.slider('Glucose Level', min_value=40, max_value=150, value=100)
    BloodPressure = st.slider('Blood Pressure', min_value=10, max_value=150, value=80)
    BMI = st.slider('BMI', min_value=15.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    Age = st.slider('Age', min_value=1, max_value=100, value=30)

    if st.button('Predict'):
        input_data = [Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)
    
    st.subheader("📈 Model Evaluation")
    if 'y_test' in globals() and 'y_predict_test' in globals():
        cm = confusion_matrix(y_test, y_predict_test)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    else:
        st.warning("Model evaluation metrics unavailable. Ensure 'y_test' and 'y_predict_test' are defined.")

if __name__ == '__main__':
    main()
