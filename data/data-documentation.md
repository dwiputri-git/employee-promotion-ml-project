# 📑 Data Documentation - Employee Promotion ML Project

## 1. Dataset Overview
- **Nama file:** [employee-promotion.csv](employee-promotion.csv)
- **Sumber data:** Internal HR (diunggah ke Google Drive)
- **Jumlah data:** 1000 baris × 10 kolom
- **Tujuan penggunaan:**  
  Dataset ini digunakan untuk membangun model machine learning yang memprediksi apakah seorang karyawan layak akan mendapatkan promosi berdasarkan faktor-faktor kuantitatif.
  
## 2. Deskripsi Kolom

| Kolom                  | Deskripsi                                                    | Tipe Data | 
| :-------------------   | :-----------------------------------------                   | :-------- | 
| Employee_ID            | ID unik untuk tiap karyawan                                  | object    | 
| Age                    | Usia karyawan                                                | float     |
| Years_at_Company       | Lama bekerja di perusahaan (tahun)                           | float     |
| Performance_Score      | Skor performa tahunan (1–5)                                  | float     |
| Leadership_Score       | Skor kepemimpinan (0–100)                                    | float     |
| Training_Hours         | Jumlah jam pelatihan yang diikuti karyawan                   | float     |
| Projects_Handled       | Jumlah proyek yang pernah ditangani                          | float     |
| Peer_Review_Score      | Skor penilaian rekan kerja (0–100)                           | float     |
| Current_Position_Level | Level jabatan saat ini (Junior, Mid, Senior, Lead)           | object    |
| Promotion_Eligible     | Label target: 1 = eligible dipromosikan, 0 = tidak eligible  | float     |

### 3. Initial Data Check
- **Missing Values:**
- Ditemukan nilai kosong pada hampir semua kolom (kecuali Employee_ID)

- **Duplicate Data:**
-  Tidak ditemukan duplikasi berdasarkan pemeriksaan awal.

- **Data Type Consistency:**
- Kolom numerik (Age, Years_at_Company, dll.) sesuai dengan tipe data float.
- Kolom kategori (Current_Position_Level) bertipe object.
- Kolom Promotion_Eligible bertipe float (0.0 dan 1.0), akan dikonversi ke int pada preprocessing.

### 4. Reproducibility Data
Untuk memastikan hasil pengecekan ini bisa direproduksi kembali, jalankan notebook berikut:
📄 [01_data_intro.ipynb](01_data_intro.ipynb)

### Kesimpulan Awal
- Dataset memiliki beberapa missing value yang perlu ditangani saat preprocessing.
- Tidak ada data duplikat.
- Perlu konversi tipe data Promotion_Eligible menjadi integer.
- Dataset siap diproses lebih lanjut untuk tahap EDA.
