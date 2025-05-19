# praktikum.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Risk Classification", layout="centered")
st.title("🦠 COVID-19 Risk Level Classification Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")

    # Filter hanya Indonesia
    df = df[df['Location ISO Code'] == 'ID']

    # Agregasi per lokasi
    df = df.groupby('Location').agg({
        'Total Cases': 'max',
        'Total Deaths': 'max',
        'Total Recovered': 'max',
        'Population Density': 'max',
        'Case Fatality Rate': 'max'
    }).dropna()

    # Hitung nilai maksimum total kasus
    max_val = df['Total Cases'].max()

    # Tentukan bins secara dinamis berdasarkan nilai maksimum
    # Jika data terlalu kecil, gunakan hanya 2 kategori
    if max_val <= 1000:
        bins = [0, max_val + 1]
        labels = ['Low']
    elif max_val <= 10000:
        bins = [0, 1000, max_val + 1]
        labels = ['Low', 'Medium']
    else:
        bins = [0, 1000, 10000, max_val + 1]
        labels = ['Low', 'Medium', 'High']

    # Pastikan bins naik dan unik
    bins = sorted(set(bins))

    # Buat label sesuai jumlah interval
    if len(labels) != len(bins) - 1:
        labels = [f"Level {i+1}" for i in range(len(bins)-1)]

    # Assign kategori risiko
    df['Risk Level'] = pd.cut(df['Total Cases'], bins=bins, labels=labels, include_lowest=True)

    return df

# Ambil data
df = load_data()
st.subheader("📋 Data Overview")
st.dataframe(df)

# Cek jumlah kelas
if df['Risk Level'].nunique() < 2:
    st.error("⚠️ Data tidak memiliki cukup variasi kelas untuk klasifikasi. Silakan periksa data atau ubah batas kategori.")
    st.stop()

# Fitur dan Target
X = df[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
y = df['Risk Level']

# Balancing dengan SMOTE
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
except ValueError as e:
    st.error(f"❌ Gagal melakukan SMOTE: {e}")
    st.stop()

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi Model
st.header("📊 Model Evaluation")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
st.pyplot(fig)

# Distribusi Prediksi
st.subheader("📈 Distribusi Klasifikasi")
st.bar_chart(pd.Series(y_pred).value_counts())

# Grafik Tren Harian (Opsional)
st.header("📅 Daily Cases Trend")
try:
    df_raw = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    df_raw = df_raw[df_raw["Location"] == "Indonesia"]
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_trend = df_raw.groupby('Date')['New Cases'].sum()
    st.line_chart(df_trend)
except Exception as e:
    st.warning(f"Gagal menampilkan grafik tren: {e}")
