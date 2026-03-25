# Weather Prediction App

Aplikasi prediksi cuaca berbasis Machine Learning menggunakan Random Forest Classifier untuk memprediksi kemungkinan hujan di Australia besok.

## Deskripsi Project

project ini merupakan tugas Exploratory Data Analysis (EDA) yang menganalisis dataset cuaca Australia dan membangun model prediksi hujan. Aplikasi web dibangun dengan Streamlit untuk visualisasi data dan prediksi interaktif.

### Dataset
- **Sumber**: [Kaggle - Weather Dataset (Rattle Package)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Periode**: ~10 tahun observasi cuaca harian
- **Lokasi**: Berbagai stasiun cuaca di Australia
- **Data Source**: [Bureau of Meteorology Australia](http://www.bom.gov.au/climate/data)

## Fitur Utama

### 1. Dashboard
- **Summary Dataset**: Informasi total kota, baris, dan kolom
- **Probability of Rain**: Visualisasi proporsi hari hujan vs cerah
- **Correlation Analysis**: Heatmap korelasi antar fitur (raw & cleaned)
- **Feature Importance**: Fitur paling berpengaruh terhadap prediksi hujan
- **Analysis by City**: Analisis curah hujan dan suhu per kota

### 2. Rain Prediction
- Input data cuaca (suhu, kelembaban, tekanan, kecepatan angin, dll)
- Prediksi probabilitas hujan besok
- Visualisasi hasil prediksi dengan indikator warna:
  - Hijau: Probabilitas rendah (< 40%)
  - Kuning: Probabilitas sedang (40-50%)
  - Merah: Probabilitas tinggi (> 50%)

## Tech Stack

| Library | Versi | Kegunaan |
|---------|--------|----------|
| Python | 3.x | Bahasa pemrograman |
| Streamlit | 1.53.1 | Web app framework |
| Pandas | 2.3.3 | Data manipulation |
| NumPy | 2.4.2 | Numerical computing |
| Scikit-learn | 1.8.0 | Machine Learning |
| Plotly | 6.5.2 | Visualisasi interaktif |
| Joblib | 1.5.3 | Model serialization |

## Struktur Folder

```
Tugas-EDA/
├── app.py                      # Main Streamlit application
├── train_model.py              # Script training model Random Forest
├── processing_data.ipynb       # Jupyter notebook EDA & preprocessing
├── requirements.txt            # Python dependencies
├── README.md                    # Dokumentasi project
├── dataset/
│   ├── raw_weatherAUS.csv         # Dataset mentah
│   ├── weather_cleaned.css        # Dataset hasil cleaning
│   ├── for_trained_weatherAUS.csv # Dataset siap training
│   └── feature_importance.csv     # Hasil feature importance
└── model/
    ├── rain_prediction_model.joblib          # Model terlatih
    ├── rain_prediction_model_accuracy.joblib # Paket model + metrics
    └── le_location.pkl                      # Label encoder lokasi
```

## Model Machine Learning

### Algorithm: Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    class_weight='balanced',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

### Metrics
- **Accuracy**: ~85%
- **F1 Score**: Balanced precision & recall
- **Feature Engineering**:
  - DeltaPressure (selisih tekanan)
  - DeltaTemp (selisii suhu)
  - DeltaHumidity (selisii kelembaban)
  - One-hot encoding untuk arah angin

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/KHLLS/Tugas-EDA.git
cd Tugas-EDA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## Key Insights

1. **Humidity3pm** adalah fitur paling berpengaruh terhadap prediksi hujan
2. **MaxTemp** berkorelasi negatif dengan kemungkinan hujan
3. Australia merupakan benua kering - hujan hanya terjadi ~22% dari waktu
4. Kolom dengan korelasi >= 90% telah dihapus untuk menghindar redundansi

## Notebook Processing

File `processing_data.ipynb` berisi:
- Exploratory Data Analysis (EDAA
- Data cleaning & handling missing values
- Feature engineering
- Data preparation untuk training

## Author

**Kahlil** - [GitHub](https://github.com/KHLLS)

## License

Data source: Copyright Commonwealth of Australia 2010, Bureau of Meteorology.
