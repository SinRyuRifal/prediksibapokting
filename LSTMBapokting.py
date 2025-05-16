import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Bapokting", layout="wide")

# Judul
st.title("📈 Prediksi Harga Bahan Pokok (LSTM)")

# Kolom yang digunakan
selected_columns = [
    'Beras Premium', 'Beras Medium', 'Bawang Merah', 'Bawang Putih Bonggol',
    'Cabai Merah Keriting', 'Daging Sapi Murni', 'Cabai Rawit Merah',
    'Daging Ayam Ras', 'Telur Ayam Ras', 'Gula Konsumsi',
    'Minyak Goreng Kemasan'
]

# Unduh template
st.sidebar.header("📥 Template & Upload")
st.sidebar.markdown("Dataset minimal berisi 30 baris data")

with st.sidebar:
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    template_df = pd.DataFrame(columns=["Tanggal"] + selected_columns)
    csv = convert_df_to_csv(template_df)
    st.download_button("📄 Unduh Template CSV", data=csv, file_name="template_bapokting.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📤 Unggah File CSV", type=["csv", "xlsx"])

# Proses jika file diunggah
if uploaded_file is not None:
    # Baca file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=["Tanggal"])
        else:
            df = pd.read_excel(uploaded_file, parse_dates=["Tanggal"])
        df.set_index("Tanggal", inplace=True)

        # Validasi kolom dan panjang data
        if not all(col in df.columns for col in selected_columns):
            st.error("❌ File harus mengandung kolom berikut: " + ", ".join(selected_columns))
        elif len(df) < 30:
            st.error("❌ Data minimal harus berisi 30 baris!")
        else:
            st.success("✅ Data berhasil diunggah dan valid!")

            # Lanjutkan prediksi
            data = df[selected_columns].dropna()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            sequence_length = 30
            input_sequence = scaled_data[-sequence_length:][np.newaxis, :, :]

            model = load_model("lstm_bapokting.keras")
            n_future = 30
            predictions_scaled = []

            for _ in range(n_future):
                next_pred = model.predict(input_sequence, verbose=0)[0]
                predictions_scaled.append(next_pred)
                input_sequence = np.vstack([input_sequence[0, 1:], next_pred])[np.newaxis, :, :]

            predictions_scaled = np.array(predictions_scaled)
            predictions = scaler.inverse_transform(predictions_scaled)
            predictions = np.round(predictions).astype(int)

            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future)
            forecast_df = pd.DataFrame(predictions, columns=selected_columns, index=future_dates)

            st.subheader("📊 Hasil Prediksi 30 Hari ke Depan")
            st.dataframe(forecast_df)

            # Unduh hasil prediksi
            output = BytesIO()
            forecast_df.to_excel(output)
            st.download_button(
                "⬇️ Unduh Hasil Prediksi",
                data=output.getvalue(),
                file_name="hasil_prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Visualisasi
            st.subheader("📉 Visualisasi Prediksi")
            fig, axes = plt.subplots(6, 2, figsize=(20, 25))
            axes = axes.flatten()

            for i, col in enumerate(selected_columns):
                ax = axes[i]
                ax.plot(df.index[-sequence_length:], df[col][-sequence_length:], label="Aktual", color="blue", marker='o')
                ax.plot(forecast_df.index, forecast_df[col], label="Prediksi", color="orange", marker='o')
                ax.set_title(col)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)
                ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file data untuk memulai prediksi.")
