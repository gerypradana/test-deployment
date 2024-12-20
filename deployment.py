import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl')

# Load dataset untuk informasi tambahan
df = pd.read_csv('fifa_eda.csv')

# Hitung rata-rata
overall_mean = df['Overall'].mean()
potential_mean = df['Potential'].mean()
value_mean = df['Value'].mean()

# Streamlit UI
st.title("FIFA Star Player Prediction")
st.write("Masukkan nilai Overall, Potential, dan Value untuk melihat jumlah pemain yang Star dan Not Star.")

# Input user
overall = st.number_input("Overall", min_value=0, max_value=100, step=1)
potential = st.number_input("Potential", min_value=0, max_value=100, step=1)
value = st.number_input("Value (dalam juta)", min_value=0, step=1)

if st.button("Prediksi"):
    # Tentukan apakah input lebih besar dari rata-rata
    is_star = (overall > overall_mean) and (potential > potential_mean) and (value > value_mean)
    
    if is_star:
        st.write("Pemain ini bisa dianggap sebagai Star berdasarkan input yang diberikan.")
    else:
        st.write("Pemain ini tidak memenuhi kriteria untuk dianggap sebagai Star berdasarkan input yang diberikan.")

    # Filter data berdasarkan input
    filtered_df = df[(df['Overall'] >= overall) & 
                     (df['Potential'] >= potential) & 
                     (df['Value'] >= value)]
    
    if not filtered_df.empty:
        # Normalisasi data yang difilter
        filtered_features = filtered_df[['Overall', 'Potential', 'Value']]
        filtered_features_scaled = scaler.transform(filtered_features)
        
        # Prediksi Star untuk data yang difilter
        filtered_df['Star'] = rf_model.predict(filtered_features_scaled)
        
        # Hitung jumlah Star dan Not Star
        star_count = filtered_df['Star'].sum()
        not_star_count = len(filtered_df) - star_count
        
        st.write(f"Jumlah pemain yang Star: {star_count}")
        st.write(f"Jumlah pemain yang Not Star: {not_star_count}")

        # Tampilkan nilai Star yang diprediksi
        st.write("Nilai Star yang diprediksi:", filtered_df['Star'].value_counts())

        # Tampilkan data pemain Star jika ada
        star_players = filtered_df[filtered_df['Star'] == 1]
        if not star_players.empty:
            st.subheader("Detail Pemain Star")
            st.dataframe(star_players)

        # Tampilkan data pemain Not Star jika ada
        not_star_players = filtered_df[filtered_df['Star'] == 0]
        if not not_star_players.empty:
            st.subheader("Detail Pemain Not Star")
            st.dataframe(not_star_players)
    else:
        st.write("Tidak ada pemain yang memenuhi kriteria Star.")







