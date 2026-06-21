import gradio as gr
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load models and scaler
model_rf = joblib.load('model_rf.pkl')
model_kmeans = joblib.load('model_kmeans.pkl')
scaler = joblib.load('scaler.pkl')

def predict(n, p, k, suhu, kelembapan, ph, curah_hujan):
    # Data input mentah
    input_data = [[n, p, k, suhu, kelembapan, ph, curah_hujan]]
    
    # 1. Prediksi jenis tanaman dengan Random Forest
    rf_pred = model_rf.predict(input_data)[0].upper()
    
    # 2. Prediksi cluster dengan K-Means
    input_scaled = scaler.transform(input_data)
    kmeans_pred = model_kmeans.predict(input_scaled)[0]
    
    cluster_meaning = {
        0: "Lahan Lembap/Basah",
        1: "Lahan Kering/Panas",
        2: "Lahan Sedang/Stabil"
    }

    hasil_cluster = f"Cluster {kmeans_pred} ({cluster_meaning.get(kmeans_pred, 'Tidak diketahui')})"
    
    return rf_pred, hasil_cluster

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Nitrogen (N)", value=90),
        gr.Number(label="Fosfor (P)", value=42),
        gr.Number(label="Kalium (K)", value=43),
        gr.Number(label="Suhu (°C)", value=20.8),
        gr.Number(label="Kelembapan (%)", value=82.0),
        gr.Number(label="pH Tanah", value=6.5),
        gr.Number(label="Curah Hujan (mm)", value=202.9),
    ],
    outputs=[
        gr.Textbox(label="🌱 Prediksi Rekomendasi Tanaman (Random Forest)"),
        gr.Textbox(label="🧩 Kategori Lahan (K-Means Cluster)")
    ],
    title="🌾 Smart Farming Predictor",
    description="Masukkan parameter kandungan tanah dan cuaca untuk memprediksi rekomendasi tanaman sekaligus mengetahui kategori lahan (cluster).",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)