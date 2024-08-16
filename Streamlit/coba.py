import streamlit as st
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
# from imblearn.under_sampling import RandomUnderSampler
# Memuat dataset CSV ke dalam DataFrame
df = pd.read_csv("Medicaldataset.csv")

label_encoder = LabelEncoder()
label_encoder.fit(df["Result"])
class_mapping = dict(zip(label_encoder.classes_,
                label_encoder.transform(label_encoder.classes_)))
print(class_mapping)
df["Result"] = label_encoder.transform(df["Result"])
 
# Fungsi untuk menampilkan data dalam aplikasi Streamlit
def show_data():
    st.title("Data Analysis with Streamlit")
    st.write("Data from CSV:")
    st.write(df)  # Menampilkan data dalam bentuk tabel

# Panggil fungsi untuk menampilkan data
show_data()
# Fungsi untuk memuat model dan scaler berdasarkan pilihan pengguna
def load_model_and_scaler(model_choice):
    model_file = f'gaussian_nb_model_{model_choice}.pkl'
    scaler_file = f'scaler_{model_choice}.pkl'
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def main():
    st.title("Prediksi Penyakit Jantung")
    st.write("Isi formulir di bawah untuk melakukan prediksi.")

# Pilihan model
    model_choice = st.selectbox("Pilih Model", ["A", "B", "C", "D"])

    # Input form
    age = st.number_input("Umur", min_value=0, max_value=150, value=58)
    gender = st.radio("Jenis Kelamin", options=["Pria", "Wanita"])
    heart_rate = st.number_input("Heart Rate", min_value=0, value=61)
    sys_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=112)
    dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=58)
    blood_sugar = st.number_input("Gula Darah", min_value=0, value=87)
    ck_mb = st.number_input("CK-MB", min_value= None, max_value=None, value=1.83,format="%.4f")
    troponin = st.number_input("Troponin", min_value=None, max_value=None, value=0.004, format="%.4f")
    st.write(ck_mb)
    st.write(troponin)

    # Konversi gender menjadi nilai numerik
    gender = 1 if gender == "Pria" else 0

    # Prediksi
    if st.button("Prediksi"):
        model, scaler = load_model_and_scaler(model_choice)
        result = predict_heart_disease(model, scaler, age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
        st.success(f"Prediksi: {'Positif' if result == 1 else 'Negatif'}")

def predict_heart_disease(model, scaler, age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
    # Format input data
    input_data = np.array([[age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin]])
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict with the model
    prediction = model.predict(input_data_scaled)
    print(f"Input data: {input_data}")
    print(f"Scaled input data: {input_data_scaled}")
    print(f"Prediction: {prediction}")
    return prediction[0]

if __name__ == "__main__":
    main()

# def main():
#     st.title("Prediksi Penyakit Jantung")
#     st.write("Isi formulir di bawah untuk melakukan prediksi.")

#     # Pilihan model
#     model_choice = st.selectbox("Pilih Model", ["A", "B", "C", "D"])

#     # Input form
#     age = st.number_input("Umur", min_value=0, max_value=150, value=55)
#     gender = st.radio("Jenis Kelamin", options=["Pria", "Wanita"])
#     heart_rate = st.number_input("Heart Rate", min_value=0, value=64)
#     sys_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=160)
#     dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=77)
#     blood_sugar = st.number_input("Gula Darah", min_value=0, value=270)
#     ck_mb = st.number_input("CK-MB", min_value=0, value=199)
#     troponin = st.number_input("Troponin", min_value=0, value=3)

#     # Konversi gender menjadi nilai numerik
#     gender = 1 if gender == "Pria" else 0

#     # Prediksi
#     if st.button("Prediksi"):
#         model, scaler = load_model_and_scaler(model_choice)
#         result = predict_heart_disease(model, scaler, age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
#         st.success(f"Prediksi: {'Positif' if result == 1 else 'Negatif'}")


# def predict_heart_disease(model, scaler, age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
#     # Format input data
#     input_data = np.array([[age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin]])
    
#     # Scale input data
#     input_data_scaled = scaler.transform(input_data)
    
#     # Predict with the model
#     prediction = model.predict(input_data_scaled)
#     print(f"Input data: {input_data}")
#     print(f"Scaled input data: {input_data_scaled}")
#     print(f"Prediction: {prediction}")
#     return prediction[0]

# if __name__ == "__main__":
#     main()