import streamlit as st
import pandas as pd
import pickle

# ===============================
# 🎯 LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        with open("model/model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# ===============================
# 🎨 APP UI
# ===============================
st.set_page_config(page_title="Promotion Eligibility Predictor", page_icon="📊", layout="centered")

st.title("📊 Employee Promotion Eligibility Prediction")
st.write("Upload file **CSV** berisi data karyawan untuk memprediksi apakah mereka layak dipromosikan.")

# ===============================
# 📂 FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload file data karyawan (CSV atau Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Baca file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("📋 Data yang diupload:")
        st.dataframe(data.head())

        # Pastikan model sudah dimuat
        if model is not None:
            st.subheader("🔮 Hasil Prediksi:")
            preds = model.predict(data)
            data["Promotion_Eligible_Prediction"] = preds

            # Tampilkan hasil
            st.dataframe(data)

            # Unduh hasil prediksi
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name="promotion_predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("Model belum berhasil dimuat.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file atau memproses data: {e}")
else:
    st.info("Silakan upload file CSV atau Excel terlebih dahulu.")

# ===============================
# 👣 FOOTER
# ===============================
st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh dwiputri-git | Streamlit Deployment Demo")
