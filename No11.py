import pickle
import streamlit as st
import pandas as pd
import numpy as np

# --- Konfigurasi Halaman Streamlit (HARUS JADI YANG PERTAMA) ---
st.set_page_config(
    page_title="Prediksi Harga Mobil Sederhana",
    page_icon="🚗",
    layout="wide"
)

# --- 1. Memuat Model Machine Learning ---
try:
    model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))
    st.success("Model prediksi berhasil dimuat.")
except FileNotFoundError:
    st.error("Error: File 'model_prediksi_harga_mobil.sav' tidak ditemukan. Pastikan file model ada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# --- 2. Judul Aplikasi ---
st.title("🚗 Aplikasi Prediksi Harga Mobil Sederhana")
st.write("Aplikasi ini memprediksi harga mobil berdasarkan Highway MPG, Curbweight, dan Horsepower.")

# --- 3. Tampilan Dataset ---
st.header("📊 Dataset Mobil")
st.write("Berikut adalah sebagian dari dataset yang digunakan untuk pelatihan model.")

try:
    df1 = pd.read_csv('CarPrice.csv')
    st.dataframe(df1.head()) # Tampilkan beberapa baris pertama dataset
    st.write(f"Dataset memiliki {df1.shape[0]} baris dan {df1.shape[1]} kolom.")

    # Pastikan kolom numerik dan tidak ada NaN yang mengganggu plotting/prediksi
    df1['highwaympg'] = pd.to_numeric(df1['highwaympg'], errors='coerce')
    df1['curbweight'] = pd.to_numeric(df1['curbweight'], errors='coerce')
    df1['horsepower'] = pd.to_numeric(df1['horsepower'], errors='coerce')
    df1.dropna(subset=['highwaympg', 'curbweight', 'horsepower'], inplace=True) # Hapus baris dengan NaN

except FileNotFoundError:
    st.warning("File 'CarPrice.csv' tidak ditemukan. Bagian dataset tidak dapat ditampilkan.")
    df1 = pd.DataFrame() # Buat DataFrame kosong agar kode selanjutnya tidak error
except Exception as e:
    st.error(f"Error saat memuat CarPrice.csv: {e}")
    df1 = pd.DataFrame()


# --- 4. Grafik Fitur Kunci (jika df1 berhasil dimuat dan tidak kosong) ---
if not df1.empty:
    st.header("📈 Visualisasi Fitur Kunci")
    st.write("Grafik di bawah menunjukkan distribusi fitur-fitur utama.")

    # Grafik Highway-mpg
    st.subheader("Grafik Highway MPG")
    if 'highwaympg' in df1.columns:
        st.line_chart(df1['highwaympg'])
    else:
        st.info("Kolom 'highwaympg' tidak ditemukan di dataset.")

    # Grafik Curbweight
    st.subheader("Grafik Curbweight")
    if 'curbweight' in df1.columns:
        st.line_chart(df1['curbweight'])
    else:
        st.info("Kolom 'curbweight' tidak ditemukan di dataset.")

    # Grafik Horsepower
    st.subheader("Grafik Horsepower")
    if 'horsepower' in df1.columns:
        st.line_chart(df1['horsepower'])
    else:
        st.info("Kolom 'horsepower' tidak ditemukan di dataset.")
else:
    st.info("Tidak dapat menampilkan grafik karena dataset 'CarPrice.csv' tidak dimuat atau kosong.")

# --- 5. Input Nilai untuk Prediksi ---
st.header("🔍 Prediksi Harga Mobil")
st.write("Masukkan nilai fitur-fitur berikut untuk mendapatkan estimasi harga mobil:")

col1, col2, col3 = st.columns(3)

with col1:
    highwaympg = st.number_input("Highway MPG:", min_value=0.0, max_value=100.0, value=30.0, help="Miles per gallon di jalan tol.")
with col2:
    curbweight = st.number_input("Curbweight (lbs):", min_value=500.0, max_value=6000.0, value=2500.0, help="Berat kosong mobil dalam pound.")
with col3:
    horsepower = st.number_input("Horsepower (hp):", min_value=0.0, max_value=500.0, value=100.0, help="Tenaga kuda mesin.")


# --- 6. Tombol Prediksi ---
if st.button("Prediksi Harga"):
    try:
        # Pastikan urutan fitur sesuai dengan urutan saat model dilatih!
        features = np.array([[highwaympg, curbweight, horsepower]])
        car_prediction = model.predict(features)

        # Mengonversi prediksi ke format Rupiah
        harga_mobil_float = float(car_prediction[0][0])
        harga_mobil_formatted = f"Rp {harga_mobil_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        st.success(f"**Prediksi Harga Mobil: {harga_mobil_formatted}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.write("Pastikan input numerik benar dan model dimuat dengan sukses.")

st.markdown("---")
st.caption("Aplikasi Prediksi Harga Mobil Sederhana © 2025. Dibuat dengan Streamlit. Oleh mhmdfarouqq")
