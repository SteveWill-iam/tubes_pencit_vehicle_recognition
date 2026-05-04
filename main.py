import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Grayscale Manual (Dibiarkan untuk modul Visualisasi)
def manual_grayscale(image_rgb):
    tinggi, lebar, _ = image_rgb.shape
    gray_img = np.zeros((tinggi, lebar))
    for i in range(tinggi):
        for j in range(lebar):
            r, g, b = image_rgb[i, j]
            gray_img[i, j] = int((int(r) + int(g) + int(b)) / 3)
    return gray_img

#Edge deteksi manual (Dibiarkan untuk modul Visualisasi)
def manual_edge_detection(gray_img):
    tinggi, lebar = gray_img.shape
    hasil_edge = np.zeros((tinggi, lebar))
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
    for i in range(1, tinggi - 1):
        for j in range(1, lebar - 1):
            area = gray_img[i-1:i+2, j-1:j+2]
            gx = np.sum(area * kernel_x)
            gy = np.sum(area * kernel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            hasil_edge[i, j] = magnitude
    return hasil_edge

def visualize_preprocessing(img_path):
    print(f"\n[VISUALISASI] Menampilkan visualisasi cara kerja pada gambar: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Gagal memuat gambar untuk visualisasi.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    print("[VISUALISASI] Menghitung Grayscale manual...")
    gray = manual_grayscale(img_resized)
    print("[VISUALISASI] Menghitung Edge Detection manual...")
    edges = manual_edge_detection(gray)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('1. Original RGB (224x224)')
    plt.imshow(img_resized)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('2. Manual Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('3. Manual Edge Detection (Sobel)')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.suptitle('Demonstrasi Preprocessing Manual (Cara Kerja Dasar Visi Komputer)\nTUTUP JENDELA INI UNTUK MELANJUTKAN TRANING', fontsize=16)
    plt.tight_layout()
    plt.show()  # Program akan berhenti (pause) sampai jendela matplotlib ditutup
    print("\n[VISUALISASI] Jendela ditutup. Melanjutkan ke proses Training CNN...")

if __name__ == "__main__":
    print("=== 🚗 Vehicle Recognition System (CNN) ===")
    
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    test_images_dir = 'test_images'
    
<<<<<<< Updated upstream
    # 1. Visualisasi Cara Kerja (Ambil 1 contoh gambar)
=======
    # Deteksi kelas dari struktur folder dataset/train
>>>>>>> Stashed changes
    try:
        class_list = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_names = {i: name for i, name in enumerate(class_list)}
    except Exception as e:
        print(f"Gagal mendeteksi kelas dari folder {train_dir}.")
        class_names = {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'truck'} # Fallback
    
<<<<<<< Updated upstream
    # 2. Persiapan Data (Data Augmentation & Pipeline)
    print("\n[DATA] Mempersiapkan Dataset dengan Data Augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # Normalisasi pixel 0-1
        rotation_range=20,        # Rotasi acak
        width_shift_range=0.2,    # Geser horizontal
        height_shift_range=0.2,   # Geser vertikal
        horizontal_flip=True,     # Flip arah
        zoom_range=0.2            # Zoom in/out acak
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Simpan nama kelas (mapping index -> string)
    class_names = {v: k for k, v in train_generator.class_indices.items()}
=======
>>>>>>> Stashed changes
    print(f"Kelas terdeteksi: {class_names}")

    print("\nPilih mode operasi:")
    print("[1] Train model baru")
    print("[2] Load model yang sudah ada")
    mode_choice = input("Masukkan pilihan (1/2): ").strip()
    
<<<<<<< Updated upstream
    # 3. Membangun Arsitektur CNN Manual (Sesuai README, Tanpa Transfer Learning)
    print("\n[MODEL] Membangun Arsitektur CNN manual...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Regularization untuk mencegah overfitting (menghafal buta)
        layers.Dense(4, activation='softmax') # 4 Target kelas
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # 4. Fitur Auto-Rem (Callbacks)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    
    # 5. Training
    print("\n[TRAINING] Memulai proses pembelajaran...")
    history = model.fit(
        train_generator,
        epochs=30, 
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Simpan Model Terbaik
    model.save('best_vehicle_model.keras')
    print("\n✅ Model terbaik berhasil disimpan sebagai 'best_vehicle_model.keras'")
=======
    if mode_choice == '1':
        # 1. Visualisasi Cara Kerja (Ambil 1 contoh gambar)
        try:
            if class_list:
                contoh_dir = os.path.join(train_dir, class_list[0])
                contoh_gambar = os.path.join(contoh_dir, os.listdir(contoh_dir)[0])
                visualize_preprocessing(contoh_gambar)
        except Exception as e:
            print(f"Gagal melakukan visualisasi: {e}")
        
        # 2. Persiapan Data (Data Augmentation & Pipeline)
        print("\n[DATA] Mempersiapkan Dataset dengan Data Augmentation...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,           # Normalisasi pixel 0-1
            rotation_range=20,        # Rotasi acak
            width_shift_range=0.2,    # Geser horizontal
            height_shift_range=0.2,   # Geser vertikal
            horizontal_flip=True,     # Flip arah
            zoom_range=0.2            # Zoom in/out acak
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        validation_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        # 3. Membangun Arsitektur CNN Manual (Sesuai README, Tanpa Transfer Learning)
        print("\n[MODEL] Membangun Arsitektur CNN manual...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), # Regularization untuk mencegah overfitting (menghafal buta)
            layers.Dense(len(class_names), activation='softmax') # Target kelas otomatis sesuai jumlah kelas
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
                      
        model.summary()
        
        # 4. Fitur Auto-Rem (Callbacks)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        
        # 5. Training
        print("\n[TRAINING] Memulai proses pembelajaran...")
        history = model.fit(
            train_generator,
            epochs=30, 
            validation_data=validation_generator,
            callbacks=[early_stop, reduce_lr]
        )
        
        # Simpan Model Terbaik -> ganti nama model jika ingin membuat model baru
        model.save('best_vehicle_model_55mb_30ep.keras')
        print("\n✅ Model terbaik berhasil disimpan")
        
    elif mode_choice == '2':
        model_files = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('.h5')]
        
        if not model_files:
            print("❌ Tidak ada model tersimpan (.keras atau .h5) di direktori ini. Jalankan ulang dan pilih opsi [1] Train.")
            exit()
            
        print("\nModel yang tersedia:")
        for i, m_file in enumerate(model_files):
            print(f"[{i+1}] {m_file}")
            
        try:
            m_index = int(input("Pilih nomor model yang ingin dimuat: ").strip()) - 1
            if m_index < 0 or m_index >= len(model_files):
                raise ValueError("Nomor di luar pilihan.")
                
            selected_model = model_files[m_index]
            print(f"\n[LOADING] Memuat model '{selected_model}'...")
            model = tf.keras.models.load_model(selected_model)
            print("✅ Model berhasil dimuat.")
        except Exception as e:
            print(f"❌ Pilihan tidak valid atau gagal load: {e}")
            exit()
            
    else:
        print("❌ Mode tidak valid. Program berhenti.")
        exit()
>>>>>>> Stashed changes
    
    # 6. Testing pada Banyak Gambar Baru
    if os.path.exists(test_images_dir) and os.listdir(test_images_dir):
        print(f"\n[PREDIKSI MULTIPEL] Memulai prediksi pada semua gambar di folder '{test_images_dir}'...")
        
        for img_name in os.listdir(test_images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue # Hiraukan file selain gambar
                
            img_path = os.path.join(test_images_dir, img_name)
            print(f"\n--- Memproses gambar: '{img_name}' ---")
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gagal membaca gambar {img_name}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            
            predictions = model.predict(img_array, verbose=0)[0]
            
            hasil_index = np.argmax(predictions)
            hasil_label = class_names[hasil_index]
            keyakinan = predictions[hasil_index] * 100
            
            print("HASIL SOFTMAX:")
            for i, prob in enumerate(predictions):
                print(f"  - {class_names[i].capitalize():<12} : {prob * 100:.2f}%")
                
            print(f"➤ KESIMPULAN: Mesin mendeteksi '{img_name}' sebagai '{hasil_label.upper()}' dengan probabilitas {keyakinan:.2f}%!")
    else:
        print(f"\n⚠️ [Peringatan] Folder '{test_images_dir}' tidak ditemukan atau kosong.")