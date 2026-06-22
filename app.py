import gradio as gr
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load models and scaler
rf_data = joblib.load('random_forest_crop_recommendation.pkl')
model_rf = rf_data['model']

kmeans_data = joblib.load('kmeans_land_clustering.pkl')
model_kmeans = kmeans_data['model']
scaler = kmeans_data['scaler']
cluster_label_map = kmeans_data.get('cluster_label_map', {})

def predict(n, p, k, suhu, kelembapan, ph, curah_hujan):
    # Data input mentah
    input_data = [[n, p, k, suhu, kelembapan, ph, curah_hujan]]
    
    # 1. Prediksi jenis tanaman dengan Random Forest
    rf_pred = model_rf.predict(input_data)[0].upper()
    
    # 2. Prediksi cluster dengan K-Means
    input_scaled = scaler.transform(input_data)
    kmeans_pred = model_kmeans.predict(input_scaled)[0]
    
    label_tipe_lahan = cluster_label_map.get(kmeans_pred, "Tidak diketahui")
    hasil_cluster = f"Cluster {kmeans_pred} ({label_tipe_lahan})"
    
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