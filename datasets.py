import os
import shutil
import random
import kagglehub

def setup_dataset():
    # Folder tujuan kita
    base_dir = 'dataset'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # Kelas yang ingin kita pakai (pastikan lowercase untuk pencocokan)
    target_classes = ['car', 'truck', 'motorcycle', 'bus']
    split_ratio = 0.8 # 80% Train, 20% Test
    
    # Cek apakah dataset sudah ada
    if os.path.exists(base_dir) and os.listdir(base_dir):
        print(" Folder 'dataset' sudah terisi. Skip download dan penyiapan data.")
        return

    print(" Mendownload dataset dari Kaggle...")
    source_path = kagglehub.dataset_download("kunalmadan/vehicle-multi-classification")
    print(f" Dataset terdownload di cache Kaggle: {source_path}")

    print(" Menyiapkan folder train dan test...")
    # Cari folder yang berisi gambar-gambar (biasanya di subfolder tertentu setelah didownload)
    dataset_source_dir = source_path
    for root, dirs, files in os.walk(source_path):
        if any(c.lower() in [d.lower() for d in dirs] for c in target_classes):
            dataset_source_dir = root
            break

    # Proses pemindahan dan split
    for class_name in os.listdir(dataset_source_dir):
        if class_name.lower() not in target_classes:
            continue # Skip pesawat dan kapal
            
        class_source_path = os.path.join(dataset_source_dir, class_name)
        if not os.path.isdir(class_source_path):
            continue
            
        images = os.listdir(class_source_path)
        random.shuffle(images) # Acak urutan gambar
        
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        # Buat folder tujuan jika belum ada
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Copy file ke folder Train
        for img in train_images:
            shutil.copy(os.path.join(class_source_path, img), 
                        os.path.join(train_dir, class_name, img))
            
        # Copy file ke folder Test
        for img in test_images:
            shutil.copy(os.path.join(class_source_path, img), 
                        os.path.join(test_dir, class_name, img))
                        
    print(" Dataset berhasil disiapkan dan di-split (80/20)!")

# Jalankan fungsi saat file ini dipanggil langsung
if __name__ == "__main__":
    setup_dataset()