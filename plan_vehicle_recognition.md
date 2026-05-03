# 🚗 Vehicle Recognition Project Plan

## 📌 Overview
Proyek ini bertujuan untuk mengklasifikasikan jenis kendaraan (mobil, motor, truk) menggunakan teknik **pengolahan citra** dan **machine learning (CNN)**.

---

## 🎯 Objective
- Mengimplementasikan konsep dasar pengolahan citra
- Menggunakan algoritma dari matkul lain (Machine Learning / CNN)
- Membandingkan performa dengan dan tanpa preprocessing

---

## ⚙️ Scope
### Input:
- Gambar kendaraan (dataset)

### Output:
- Label kendaraan:
  - Mobil
  - Motor
  - Truk

### Mode:
- ✅ Utama: Image-based recognition
- 🔁 Opsional: Webcam real-time demo

---

## 📦 Dataset
### Kriteria:
- Minimal 1000 gambar
- 3 kelas (mobil, motor, truk)
- Format: JPG/PNG

### Struktur folder:
```
dataset/
  train/
    car/
    motorbike/
    truck/
  test/
    car/
    motorbike/
    truck/
```

---

## 🧠 System Architecture
```
Input Image → Preprocessing → CNN Model → Prediction
```

---

## 🖼️ Image Processing (Wajib)
Langkah preprocessing:
1. Resize (128x128)
2. Normalisasi pixel
3. (Eksperimen):
   - Grayscale
   - Edge Detection (Canny)

Tujuan:
- Mengurangi noise
- Menyederhanakan fitur

---

## 🤖 Model (CNN)
### Arsitektur sederhana:
- Conv2D
- MaxPooling
- Flatten
- Dense
- Output Softmax (3 kelas)

---

## 🔁 Workflow
1. Load dataset
2. Preprocessing gambar
3. Split train & test
4. Training model
5. Evaluasi model
6. Testing dengan gambar baru

---

## 🧪 Experiment Plan
### Perbandingan:
1. Tanpa preprocessing
2. Dengan preprocessing (grayscale / edge)

### Metric:
- Accuracy
- Loss

---

## 🧾 Pseudocode
```
load dataset
for each image:
    resize image
    normalize pixel

train CNN model

for each test image:
    predict class

calculate accuracy
```

---

## 👥 Team Division (4 orang)
1. Dataset & preprocessing
2. Model training (CNN)
3. Testing & evaluation
4. Dokumentasi & presentasi

---

## ⏱️ Timeline (2 minggu)
### Minggu 1:
- Kumpulkan dataset
- Implement preprocessing
- Training awal

### Minggu 2:
- Evaluasi
- Perbandingan hasil
- Dokumentasi
- Persiapan presentasi

---

## 🎥 Demo Plan
### Minimal:
- Input gambar → output label

### Opsional:
- Webcam real-time detection

---

## 📄 Deliverables
- Source code
- Laporan (PDF)
- Dokumentasi hasil pengujian

---

## ⚠️ Risk & Mitigation
| Risk | Solution |
|------|---------|
| Dataset kurang | Tambah dari sumber lain |
| Akurasi rendah | Tambah data / tuning model |
| Waktu mepet | Fokus ke image-based saja |

---

## 🚀 Tools
- Python
- OpenCV
- TensorFlow / Keras
- Jupyter Notebook / VSCode

---

## ✅ Success Criteria
- Model bisa klasifikasi minimal 3 jenis kendaraan
- Ada perbandingan preprocessing
- Program berjalan tanpa error
- Ada hasil evaluasi yang jelas

---

## 🧠 Notes
- Fokus ke hasil, bukan fitur berlebihan
- Jangan over-engineer (hindari sistem real-time kompleks)
- Pastikan laporan kuat di analisis hasil