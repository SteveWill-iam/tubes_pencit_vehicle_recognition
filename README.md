# 🚗 Vehicle Recognition System (CNN Manual)

Proyek ini adalah sistem klasifikasi jenis kendaraan (Bus, Car, Motorcycle, Truck) yang dibangun menggunakan arsitektur **Convolutional Neural Network (CNN)** secara manual untuk memenuhi Tugas Besar mata kuliah Pengolahan Citra.

## 📌 Fitur Utama
- **Arsitektur Manual**: Dibangun dari nol tanpa *Transfer Learning* untuk transparansi perhitungan parameter.
- **Data Augmentation**: Mengatasi dataset kecil dengan teknik *Random Flip*, *Rotation*, dan *Zoom* langsung di dalam model.
- **Optimasi Otomatis**: Menggunakan *Early Stopping* untuk mencegah overfitting dan *ReduceLROnPlateau* sebagai "rem otomatis" saat akurasi mentok.
- **Input Resolusi Tinggi**: Mendukung citra RGB 224x224 untuk detail fitur yang lebih baik.

## 🛠️ Arsitektur Model
Model menggunakan 4 Blok Konvolusi dengan spesifikasi:
1. **Conv2D (32 filter)** + MaxPooling
2. **Conv2D (64 filter)** + MaxPooling
3. **Conv2D (128 filter)** + MaxPooling
4. **Conv2D (128 filter)** + MaxPooling
5. **GlobalAveragePooling2D**: Menggantikan `Flatten()` untuk mengurangi jumlah parameter dan menjaga model tetap ringan.
6. **Dense Layer (128 units)** dengan Dropout 0.5.
7. **Softmax Output**: 4 Kelas (Bus, Car, Motorcycle, Truck).

## 🚀 Cara Penggunaan

### 1. Persiapan Lingkungan
Pastikan kamu sudah menginstal Python 3.9+ dan library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset
Pastikan folder `dataset/` sudah terisi dengan struktur:
- `dataset/train/` (Folder berisi sub-folder kelas)
- `dataset/test/` (Folder berisi sub-folder kelas)

*(Atau jalankan saja `python datasets.py` untuk mengunduh dan menyusun folder secara otomatis menggunakan Kagglehub)*

### 3. Training & Visualisasi & Prediksi Gambar
Jalankan `main.py` untuk:
1. Menampilkan **Visualisasi Cara Kerja (Preprocessing)** pada 1 contoh gambar menggunakan Matplotlib (tutup jendela grafiknya untuk lanjut).
2. Memulai proses pembelajaran (Training CNN). Model terbaik akan otomatis tersimpan sebagai `best_vehicle_model.keras`.
3. Di akhir proses, melakukan prediksi otomatis pada gambar tes (`test_images/red-car-cropped.jpg`).

```bash
python main.py
```

## 📝 Catatan Eksperimen
Berdasarkan hasil pengujian, model mencapai akurasi stabil di angka ~62%. Terdapat tantangan pada citra kendaraan dengan desain modern (seperti Mitsubishi Xpander) yang sering terdeteksi sebagai Motorcycle atau Bus karena distorsi aspek rasio (resize ke 224x224) dan kemiripan pola visual pada bagian grille. Hal ini menjadi catatan untuk pengembangan dataset yang lebih variatif di masa depan.