# Smart Library Monitoring: Deteksi Ketersediaan Kursi, Barang Tertinggal, dan Heatmap Trafik Rak Buku Berbasis YOLO11

Sistem pemantauan perpustakaan berbasis Computer Vision (**YOLO11 Fine-Tuned**) dan **Streamlit** untuk menganalisis okupansi meja baca, deteksi barang tertinggal (*abandoned object / idle seat*), dan kepadatan lorong rak buku (*foot traffic heatmap*) secara *real-time* dari 3 sudut kamera CCTV Perpustakaan Universitas Prasetiya Mulya.

Repository: `https://github.com/Earthen8/prasmul-library-cv`

---

## 1. Tujuan Tugas & Deliverables UAS

Berdasarkan panduan **UAS Cloud Computing (Semester Ganjil 2026-1)**, proyek ini bertujuan untuk:
1. Mengidentifikasi masalah nyata di sekitar kampus yang sulit ditangani oleh model YOLO pra-latih COCO (**M0**).
2. Mengumpulkan, menganotasi, dan memverifikasi dataset primer sendiri bebas kebocoran data (*data leakage*).
3. Melakukan *fine-tuning* terukur dari **M0 (YOLO11s Pretrained)** $\rightarrow$ **M1 (Baseline Fine-Tuning)** $\rightarrow$ **M2 (Model Akhir)** beserta analisis peningkatan akurasi dan kecepatan.
4. Membangun aplikasi streaming/demo *real-time* sederhana yang dapat dijalankan oleh dosen dan tim.

**Deliverables Kelompok:**
* **Laporan Teknis (PDF, 15–25 halaman)** — Bab 1 s/d Bab 5 + Lampiran A–G.
* **PPT Presentasi**.
* **Video Presentasi & Demo (maksimal 5 menit)** — menampilkan perbandingan nyata **M0 vs M2** berdampingan.
* **Aplikasi Demo Streamlit** di repository GitHub ini.

---

## 2. Permasalahan Utama & Solusi yang Dicapai (3 Angle CCTV)

| ID Kamera | Profil ROI | Permasalahan Nyata | Solusi Computer Vision yang Diterapkan |
| :--- | :--- | :--- | :--- |
| **CCTV01** | `Rak_1` | 1) Mahasiswa kesulitan melihat meja sofa kosong di sisi kiri.<br>2) Staf perpustakaan perlu mengetahui baris rak buku mana yang paling padat/sering dikunjungi untuk penataan *layout*.<br>3) Area balkon lantai 2 (kanan atas) ikut tertangkap kamera. | • **Seat Occupancy & Idle Timer** pada 3 zona meja kiri (`Meja_01`–`03`).<br>• **Heatmap Baris-per-Baris Rak** pada 5 poligon diagonal (`Baris_Rak_01`–`05`) dengan pencocokan `floor_point` + `center` agar tetap terdeteksi meski badan terhalang rak (*head-only blindspot*).<br>• **Ignore Zone** (`Area_Diabaikan_Lantai2`) untuk menyaring deteksi lantai 2. |
| **CCTV02** | `Rak_2` | Area meja baca di sisi kanan sering penuh atau terisi barang tanpa pemilik, dengan sudut pandang miring dan pencahayaan jendela terang. | • **Seat Occupancy & Abandoned Object** pada poligon meja baca kanan (`Meja_01`, `Meja_02`) dengan normalisasi pencahayaan **CLAHE**. |
| **CCTV03** | `Mezanin_1` | Mahasiswa membuang waktu mencari meja baca kosong di Mezanin saat jam sibuk, serta adanya praktik *booking* kursi memakai tas/laptop yang ditinggal lama. | • **Seat Occupancy & Abandoned Object Timer** berbasis *frame rate* video pada 4 zona meja (`Meja_Depan_01`, `Meja_Tengah_02`, `Meja_Belakang_03`, `Meja_Bundar_Kanan`). |

---

## 3. Bagaimana Kami Mencapainya (Data, Eksperimen, & Hasil Resmi)

### A. Kualitas & Verifikasi Dataset (`scripts/periksa_dataset.py`)
Seluruh data berasal dari **100% data primer** rekaman CCTV perpustakaan (hari Senin, Selasa, Rabu pada kondisi pagi sepi, siang sibuk, dan sore) dengan pemisahan sesi/hari agar **bebas kebocoran data**:

| Split | Jumlah Gambar | Background | Objek `person` (`0`) | Objek `belonging` (`1`) | Error Label |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train/`** | **636** | 100 | 4.582 | 615 | 0 |
| **`valid/`** | **154** | 7 | 1.257 | 201 | 0 |
| **`test/`** *(sesi terpisah)* | **220** | 0 | 2.511 | 179 | 0 |
| **TOTAL** | **1.010** *(target `>= 1.000`)* | **107** *(10,6%)* | **8.350** *(target `>= 300`)* | **995** *(target `>= 300`)* | **0** |

* **Kebocoran Exact (MD5):** `0 pasang`
* **Kebocoran Near-Duplicate (16×16 dHash):** `0 pasang`

---

### B. Ringkasan Eksperimen Fine-Tuning (`results/tabel_percobaan_uas.csv`)
Training dijalankan di **Kaggle GPU (`T4 x2`)** menggunakan `scripts/train_uas.py` (Ultralytics `8.4.215`, `epochs=100`, `patience=30`, 2 random seed `0` & `42`):

| Eksperimen | Base Model | `imgsz` | Seed | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 | Catatan Perubahan |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_seed0` | `yolo11s.pt` | 640 | 0 | 0.7832 | 0.8211 | **0.8451** | 0.4569 | Baseline M1 |
| `m1_seed42` | `yolo11s.pt` | 640 | 42 | 0.7955 | 0.7920 | 0.8421 | 0.4572 | Baseline M1 (seed 42) |
| `iter1_aug_seed0` | `yolo11s.pt` | 640 | 0 | **0.8349** | 0.7920 | 0.8395 | 0.4563 | Iterasi 1: Augmentasi `hsv_v=0.5` |
| `iter1_aug_seed42` | `yolo11s.pt` | 640 | 42 | 0.8142 | 0.7924 | 0.8437 | 0.4585 | Iterasi 1: Augmentasi `hsv_v=0.5` |
| **`iter2_imgsz_seed0`** | `yolo11s.pt` | 960 | 0 | 0.7843 | 0.8159 | 0.8402 | **0.4596** | **Iterasi 2 (`imgsz=960`) — Dipilih sbg M2** |
| `iter2_imgsz_seed42` | `yolo11s.pt` | 960 | 42 | 0.8222 | 0.7761 | 0.8427 | 0.4529 | Iterasi 2 (`imgsz=960`, seed 42) |
| `iter3_m2_seed0` | `yolo11m.pt` | 960 | 0 | 0.7947 | 0.7945 | 0.8413 | 0.4563 | Iterasi 3: Kapasitas `yolo11m` |
| `iter3_m2_seed42` | `yolo11m.pt` | 960 | 42 | 0.8061 | 0.7875 | 0.8376 | 0.4589 | Iterasi 3: Kapasitas `yolo11m` |

> **Pemilihan M2:** `iter2_imgsz_seed0` (`yolo11s.pt`, `imgsz=960`) dipilih sebagai **M2 (`best.pt`)** berdasarkan **mAP@50-95 validasi tertinggi (`0.4596`)** serta bobot yang jauh lebih ringan dan cepat dibanding `yolo11m`.

---

### C. Hasil Evaluasi Resmi pada Set Test (`results/perbandingan_m0_m1_m2.csv`)
Diujikan pada **220 gambar `dataset/test/` (2.690 objek)** dengan `conf=0.25`, `imgsz=960` di CPU lokal (`AMD Ryzen 7 8840HS`):

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Kecepatan | `person` mAP@50 | `belonging` mAP@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 (`yolo11s.pt` COCO)** | 0.4203 | 0.1828 | **0.1743** | 0.1049 | 129.9 ms (7.7 FPS) | 0.3487 | 0.0000 |
| **M1 (`weights/m1_best.pt`)** | 0.6153 | 0.6127 | 0.5297 | 0.3146 | 125.3 ms (8.0 FPS) | 0.6385 | 0.4209 |
| **M2 (`best.pt`)** | **0.7768** | **0.6121** | **0.5968** | **0.3719** | **125.0 ms (8.0 FPS)** | **0.6722** | **0.5214** |
| **Selisih M0 $\rightarrow$ M2** | **+0.3565 (+84,8%)** | **+0.4293 (+234,8%)** | **+0.4225 (+242,4%)** | **+0.2670 (+254,5%)** | **-4.9 ms** | **+0.3235 (+92,8%)** | **+0.5214** |

---

### D. Lokasi Artefak untuk Penyusunan Laporan & PPT
* **Tabel Percobaan & Evaluasi:** `results/tabel_percobaan_uas.csv`, `results/perbandingan_m0_m1_m2.csv`
* **Kurva PR, Confusion Matrix, & F1 Curve Resmi:**
  * M0: `runs/detect/results/val/m0_pretrained/`
  * M1: `runs/detect/results/val/m1_baseline_ft/`
  * M2: `runs/detect/results/val/m2_final/`
* **Galeri Gambar Bukti Kegagalan & Analisis Kesalahan (`results/galleries/`):**
  * Kegagalan M0 (**16 gambar**): `results/galleries/m0_failures/`
  * Kesalahan M2 (**24 gambar**): `results/galleries/m2_errors/`
  * Perbandingan M0 gagal vs M2 berhasil (**12 gambar**): `results/galleries/m0_fail_m2_success/`
  * Ringkasan CSV galeri: `results/galleries/gallery_summary.csv`

---

## 4. Panduan Instalasi & Demo di Localhost (Untuk Anggota Tim)

Ikuti langkah berikut di laptop masing-masing untuk menjalankan demo aplikasi Streamlit di `http://localhost:8501`:

### Langkah 1: Clone Repository & Setup Environment
```powershell
git clone https://github.com/Earthen8/prasmul-library-cv.git
cd prasmul-library-cv

# Buat virtual environment & install library
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Langkah 2: Pasang Bobot Model (`best.pt`) & Video Demo (`demo/cut-demo/`)
Karena file `.mp4` dan `.pt` berukuran besar tidak disimpan di GitHub (`.gitignore`), silakan **download dari Google Drive kelompok** lalu letakkan di lokasi berikut:
1. **Model Fine-Tuned M2:** Letakkan file `best.pt` langsung di folder utama proyek (`prasmul-library-cv/best.pt`).
2. **Video Klip Demo:** Masukkan seluruh video klip pendek dari Google Drive ke dalam folder **`demo/cut-demo/`** sesuai kameranya:
   * `demo/cut-demo/CCTV01/` (`Cut_CCTV01_1.mp4` – `Cut_CCTV01_5.mp4`)
   * `demo/cut-demo/CCTV02/` (`Cut_CCTV02_1.mp4` – `Cut_CCTV02_5.mp4`)
   * `demo/cut-demo/CCTV03/` (`Cut_CCTV03_1.mp4` – `Cut_CCTV03_5.mp4`)

*(Opsional: Jika ingin mencoba mode Cloud API Roboflow sebagai cadangan, salin `.env.example` menjadi `.env` dan isi `ROBOFLOW_API_KEY`).*

### Langkah 3: Jalankan Aplikasi Demo di Localhost
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Browser akan otomatis terbuka di **`http://localhost:8501`**.

### Langkah 4: Cara Menggunakan Demo di Localhost
1. **Mode Monitoring Real-Time (Streaming Video):**
   * Di sidebar kiri, pilih **Mode Tampilan Demo UAS** $\rightarrow$ **`Monitoring Real-Time (Single Model)`**.
   * Pastikan **Model Deteksi (Backend)** di **`Lokal YOLO11 Fine-Tuned (best.pt / yolo11s.pt)`**.
   * Klik **Browse files** $\rightarrow$ pilih salah satu klip dari **`demo/cut-demo/CCTV01/`**, **`CCTV02/`**, atau **`CCTV03/`** *(aplikasi otomatis mendeteksi profil kamera dari nama file)*.
   * Atur **Confidence** di kisaran **`0.25`** dan **Frame Skipping** di **`5`**, lalu klik **Mulai Stream**.
2. **Mode Perbandingan Berdampingan (M0 vs M2 untuk Demo UAS):**
   * Di sidebar kiri, pilih **`Perbandingan M0 vs M2 (Demo UAS Berdampingan)`**.
   * Pilih kamera (`CCTV01` / `CCTV02` / `CCTV03`) lalu unggah 1 gambar demo (`.jpg` / `.png`) untuk melihat hasil **M0 (kiri)** vs **M2 (kanan)** lengkap dengan jumlah deteksi dan kecepatan inferensi (`ms` & `FPS`).

### Langkah 5: Verifikasi Unit Test (Opsional)
Untuk memastikan seluruh modul dan skrip UAS berjalan normal di laptop kamu:
```powershell
.venv\Scripts\python.exe -m tests.verify_app
```
Pastikan seluruh **10/10 pengujian** berstatus `[PASS]`.
