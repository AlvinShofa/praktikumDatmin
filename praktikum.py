
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load data
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
df = df[df["Location Level"] == "Province"]

# Sidebar
st.sidebar.title("COVID-19 Dashboard")
selected_view = st.sidebar.radio("Pilih Tampilan", ["Peta Clustering", "Tren Harian", "Prediksi Kasus"])

# Fitur clustering
features = ["Total Cases per Million", "Total Deaths per Million", 
            "Population Density", "Growth Factor of New Cases"]
data = df[features].dropna()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

df_filtered = df.loc[data.index].copy()
df_filtered['Cluster'] = clusters

# PETA CLUSTERING
if selected_view == "Peta Clustering":
    st.title("Clustering Wilayah di Indonesia")
    fig = px.scatter_geo(df_filtered,
                         locations="Location ISO Code",
                         locationmode="ISO-3",
                         color="Cluster",
                         hover_name="Province",
                         title="Hasil Clustering Provinsi di Indonesia")
    st.plotly_chart(fig)
    st.subheader("Karakteristik Tiap Cluster")
    st.dataframe(df_filtered.groupby("Cluster")[features].mean())

# TREND KASUS HARIAN
elif selected_view == "Tren Harian":
    st.title("Tren Kasus Harian Nasional")
    df_nasional = df[df["Location"] == "Indonesia"]
    daily_cols = ["New Cases", "New Deaths", "New Recovered"]
    df_nasional["Date"] = pd.to_datetime(df_nasional["Date"])
    df_nasional = df_nasional.sort_values("Date")

    fig = px.line(df_nasional, x="Date", y=daily_cols, title="Tren Harian Kasus, Kematian, dan Sembuh")
    st.plotly_chart(fig)

# PREDIKSI KASUS
elif selected_view == "Prediksi Kasus":
    st.title("Prediksi Jumlah Total Kasus (Supervised Learning)")

    # Ambil fitur prediktor
    df_pred = df[["Total Deaths", "Total Recovered", "Population Density", "Case Fatality Rate", "Total Cases"]].dropna()
    X = df_pred[["Total Deaths", "Total Recovered", "Population Density", "Case Fatality Rate"]]
    y = df_pred["Total Cases"]

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Masukkan Nilai untuk Prediksi")
    death = st.number_input("Total Deaths", min_value=0)
    recovered = st.number_input("Total Recovered", min_value=0)
    density = st.number_input("Population Density", min_value=0.0)
    fatality = st.number_input("Case Fatality Rate", min_value=0.0)

    pred = model.predict([[death, recovered, density, fatality]])[0]
    st.success(f"Prediksi Total Kasus: {int(pred):,}")
