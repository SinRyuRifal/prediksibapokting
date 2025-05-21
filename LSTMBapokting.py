import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Prediksi Harga Bapokting", layout="wide")
st.title("📈 Prediksi Harga Bahan Pokok (LSTM)")

SELECTED_COLUMNS = [
    'Beras Premium', 'Beras Medium', 'Bawang Merah', 'Bawang Putih Bonggol',
    'Cabai Merah Keriting', 'Daging Sapi Murni', 'Cabai Rawit Merah',
    'Daging Ayam Ras', 'Telur Ayam Ras', 'Gula Konsumsi',
    'Minyak Goreng Kemasan'
]

class LSTMBapoktingPredictor:
    def __init__(self, file):
        self.file = file
        self.df = None
        self.model = load_model("lstm_bapokting2.keras")
        self.scaler = MinMaxScaler()
        self.n_future = 30
        self.sequence_length = 7

    def load_data(self):
        if self.file.name.endswith(".csv"):
            self.df = pd.read_csv(self.file, parse_dates=["Tanggal"])
        else:
            self.df = pd.read_excel(self.file, parse_dates=["Tanggal"])
        self.df.set_index("Tanggal", inplace=True)

    def validate_data(self):
        if not all(col in self.df.columns for col in SELECTED_COLUMNS):
            return False, "File harus mengandung kolom berikut: " + ", ".join(SELECTED_COLUMNS)
        if len(self.df) < 30:
            return False, "Data minimal harus berisi 30 baris!"
        return True, "Data berhasil diunggah dan valid!"

    def predict(self):
        data = self.df[SELECTED_COLUMNS].dropna()
        scaled_data = self.scaler.fit_transform(data)
        input_seq = scaled_data[-self.sequence_length:][np.newaxis, :, :]

        preds_scaled = []
        for _ in range(self.n_future):
            next_pred = self.model.predict(input_seq, verbose=0)[0]
            preds_scaled.append(next_pred)
            input_seq = np.vstack([input_seq[0, 1:], next_pred])[np.newaxis, :, :]

        preds = self.scaler.inverse_transform(np.array(preds_scaled))
        preds = np.round(preds).astype(int)
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=self.n_future)
        return pd.DataFrame(preds, columns=SELECTED_COLUMNS, index=future_dates)

    def plot_predictions(self, forecast_df):
        total_plots = len(SELECTED_COLUMNS)
        rows = (total_plots + 1) // 2  # 2 kolom per baris
        fig, axes = plt.subplots(rows, 2, figsize=(20, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(SELECTED_COLUMNS):
            ax = axes[i]
            ax.plot(self.df.index[-self.sequence_length:], self.df[col][-self.sequence_length:], label="Aktual", color="blue", marker='o')
            ax.plot(forecast_df.index, forecast_df[col], label="Prediksi", color="orange", marker='o')
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
            ax.legend()

        # Nonaktifkan sisa axes jika ada
        for j in range(total_plots, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return fig

st.sidebar.header("📥 Template & Upload")
st.sidebar.markdown("Dataset minimal berisi 30 baris data")

with st.sidebar:
    def download_template():
        template_df = pd.DataFrame(columns=["Tanggal"] + SELECTED_COLUMNS)
        output = BytesIO()
        template_df.to_excel(output, index=False)
        output.seek(0)
        return output

    st.download_button(
        label="📄 Unduh Template Excel",
        data=download_template(),
        file_name="template_bapokting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.file_uploader("📤 Unggah File", type=["csv", "xlsx"])

# Proses prediksi
if uploaded_file is not None:
    try:
        predictor = LSTMBapoktingPredictor(uploaded_file)
        predictor.load_data()
        valid, message = predictor.validate_data()

        if not valid:
            st.error(message)
        else:
            st.success(message)

            forecast_df = predictor.predict()

            st.subheader("📊 Hasil Prediksi 30 Hari ke Depan")
            st.dataframe(forecast_df)

            output = BytesIO()
            forecast_df.to_excel(output)
            st.download_button(
                "⬇️ Unduh Hasil Prediksi",
                data=output.getvalue(),
                file_name="hasil_prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("📉 Visualisasi Prediksi")
            fig = predictor.plot_predictions(forecast_df)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file data untuk memulai prediksi.")
