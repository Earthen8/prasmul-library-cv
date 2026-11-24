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

---

## 5. Data Teknis Siap-Pakai untuk Laporan UAS

Bagian ini berisi seluruh data teknis hasil pelatihan, evaluasi, dan pengujian kode yang dapat langsung disalin ke dalam template laporan kelompok (`Abstrak`, `Bab 1`, `Bab 3`, `Bab 4`, dan `Bab 5`).

### 5.1 Ringkasan Angka untuk Abstrak
* **Masalah Baseline (M0):** Model pretrained COCO `yolo11s.pt` hanya mencapai **mAP@0.5 = 0.1743** (`person`: **0.3487**, `belonging`: **0.0000** karena tidak ada padanan kelas `belonging` di COCO dan objek CCTV berukuran kecil/tertutup meja).
* **Dataset & Metode:** **1.010 gambar** CCTV bersih (107 *background* / 10.6%) dengan total **9.345 anotasi** (**8.350 `person`**, **995 `belonging`**) dari sesi terpisah; dianotasi di Roboflow dengan dua kelas (`0 = person`, `1 = belonging`) dan di-*split* menjadi Train/Valid/Test tanpa kebocoran data.
* **Peningkatan Kinerja (Set Test = 220 gambar, `conf=0.25`, `imgsz=960`):**
  * **mAP@0.5 (semua kelas):** naik dari **0.1743 (M0)** $\rightarrow$ **0.5297 (M1)** $\rightarrow$ **0.5968 (M2)** (+0.4225 dari M0, +0.0671 dari M1).
  * **Precision (semua kelas):** naik dari **0.4203 (M0)** $\rightarrow$ **0.6153 (M1)** $\rightarrow$ **0.7768 (M2)**.
  * **Recall (semua kelas):** naik dari **0.1828 (M0)** $\rightarrow$ **0.6127 (M1)** $\rightarrow$ **0.6121 (M2)**.
  * **mAP@0.5:0.95 (semua kelas):** naik dari **0.1049 (M0)** $\rightarrow$ **0.3146 (M1)** $\rightarrow$ **0.3719 (M2)**.
  * **Kecepatan Inferensi CPU (`AMD Ryzen 7 8840HS`):** **125.03 ms/gambar (~8.00 FPS)** untuk M2.

---

### 5.2 Bahan Bab 1 (Batasan, Tabel 1.1, & Tabel 1.2)

**Batasan Teknis:**
* **Kelas yang dideteksi:** `0 = person` (pengunjung perpustakaan) dan `1 = belonging` (barang bawaan seperti tas, laptop, buku di meja/kursi).
* **Dataset bersih:** **1.010 gambar** (`train`: 636, `valid`: 154, `test`: 220) dari rekaman CCTV Perpustakaan Universitas Prasetiya Mulya pada sesi terpisah (sesi `Rabu` dan `Selasa` khusus di set `test`).
* **Perangkat & Lingkungan:** Anotasi di **Roboflow**, pelatihan model **YOLO11** (`Ultralytics`) di **Kaggle Notebook (GPU T4 x2)**, serta aplikasi antarmuka berbasis **Streamlit** (`app.py`).
* **Batasan Ruang Lingkup:** Sistem melakukan deteksi objek (`person` dan `belonging`), status kursi berbasis zona ROI, pelacakan *dwell-time* sederhana untuk barang tertinggal, serta *heatmap* trafik tanpa pengenalan wajah (*face recognition*) atau identifikasi identitas individu.

**Tabel 1.1 — Kinerja M0 pada Set Test (`dataset/test/`, 220 gambar, `imgsz=960`, `conf=0.25`)**

| Model | Kelas | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| M0 (`yolo11s.pt`, pretrained COCO) | Semua (rata-rata 2 kelas) | 0.4203 | 0.1828 | 0.1743 | 0.1049 |
| M0 (`yolo11s.pt`, pretrained COCO) | `person` (orang) | 0.8407 | 0.3656 | 0.3487 | 0.2098 |
| M0 (`yolo11s.pt`, pretrained COCO) | `belonging` (barang) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Gambar 1.1 — Contoh Kegagalan M0:**  
Gunakan gambar dari folder `results/galleries/m0_failures/` (16 sampel: 11 `objek_terlewat` seperti `Rabu_Meja3_001.jpg`, `Rabu_Meja3_008.jpg`, `Rabu_Meja3_016.jpg`; 5 `kotak_tidak_pas` seperti `Rabu_Meja3_009.jpg`, `Rabu_Meja3_011.jpg`) atau perbandingan langsung di `results/galleries/m0_fail_m2_success/` (12 sampel).

**Tabel 1.2 — Hipotesis Penyebab Kegagalan M0**

| Jenis Kegagalan | Contoh File (`results/galleries/m0_failures/`) | Hipotesis Penyebab | Perbaikan yang Diuji |
| :--- | :--- | :--- | :--- |
| Objek terlewat (*False Negative*) pada `person` | `Rabu_Meja3_008.jpg` (7/8 terlewat), `Rabu_Meja3_016.jpg` (8/8 terlewat) | Sudut CCTV tinggi/lebar membuat orang di meja jauh tampak sangat kecil dan terhalang meja/rak (*occlusion*), berbeda dengan dataset COCO. | Fine-tuning domain Perpustakaan Prasmul (M1) + peningkatan resolusi input `imgsz=960` (M2). |
| Kelas `belonging` gagal total (0.0000) | `Rabu_Meja3_001.jpg`, `Rabu_Meja3_002.jpg`, `Rabu_Meja3_003.jpg` | Dataset COCO tidak memiliki label tunggal `belonging` untuk barang tertinggal di meja CCTV. | Pelatihan ulang *custom 2-class* (`person` & `belonging`) melalui M1 dan M2 (`imgsz=960`). |
| Kotak tidak pas (*Localization Error*) | `Rabu_Meja3_009.jpg`, `Rabu_Meja3_010.jpg`, `Rabu_Meja3_011.jpg` | Proporsi tubuh pengunjung yang duduk terpotong meja menghasilkan *bounding box* tidak presisi pada model umum COCO. | Fine-tuning dengan anotasi konsisten dari rekaman CCTV target dan resolusi 960px. |

---

### 5.3 Bahan Bab 3 (Tabel 3.1, 3.3, 3.4, 3.5, & 3.6)

**Tabel 3.1 — Rincian Pengumpulan & Audit Dataset Bersih (`scripts/periksa_dataset.py`)**

| Aspek | Keterangan Teknis |
| :--- | :--- |
| Alat & Sumber | Rekaman 3 kamera CCTV Perpustakaan Prasmul (`CCTV01`, `CCTV02`, `CCTV03`); diekstrak menjadi frame gambar. |
| Lokasi | Perpustakaan Universitas Prasetiya Mulya. |
| Kondisi | Sudut pandang CCTV dari atas (*top-down/high-angle*), area meja baca dan rak buku, kepadatan bervariasi, termasuk **107 gambar *background* (10.6%)** tanpa objek. |
| Jumlah Bersih | **1.010 gambar**; **8.350 objek `person`** (kelas 0) dan **995 objek `belonging`** (kelas 1); total **9.345 anotasi** dengan **0 error label**. |

**Tabel 3.3 — Pembagian Data (Split Strategy & Pencegahan Kebocoran Data)**

| Split | Keterangan Sesi | Jumlah Gambar | Background | Objek `person` | Objek `belonging` | Total Objek | Persentase Gambar |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | Sesi terpisah dari test | 636 | 100 | 4.582 | 615 | 5.197 | 62.97% |
| **Validation** | Sesi terpisah dari test | 154 | 7 | 1.257 | 201 | 1.458 | 15.25% |
| **Test** | Sesi terpisah (`Rabu` & `Selasa`) | 220 | 0 | 2.511 | 179 | 2.690 | 21.78% |
| **Total** | — | **1.010** | **107 (10.6%)** | **8.350** | **995** | **9.345** | **100%** |

*Pencegahan Kebocoran Data:* Pemisahan set `test` dilakukan berdasarkan sesi rekaman (`Rabu` dan `Selasa`), bukan acak per frame. Verifikasi otomatis menggunakan `scripts/periksa_dataset.py` berbasis *16x16 dHash* memastikan **0 duplikat persis (*exact leak*)** dan **0 kemiripan tinggi (*near leak*)** antar split.

**Tabel 3.4 — Rancangan Model M0, M1, dan M2**

| Model | Deskripsi | Perubahan dari Model Sebelumnya | Hipotesis yang Diuji |
| :--- | :--- | :--- | :--- |
| **M0** | `yolo11s.pt` pretrained COCO tanpa pelatihan ulang | — (Baseline) | Model umum COCO gagal mendeteksi objek kecil/tertutup meja dan tidak mengenal kelas `belonging`. |
| **M1** | `yolo11s.pt` di-*fine-tune* dengan dataset CCTV Prasmul (`imgsz=640`, konfigurasi default) | Pelatihan ulang 2 kelas (`person`, `belonging`) dengan bobot awal COCO | Fine-tuning pada domain target meningkatkan akurasi kedua kelas secara signifikan. |
| **M2** | `yolo11s.pt` di-*fine-tune* dengan resolusi input lebih tinggi (`imgsz=960`, `iter2_imgsz_seed0`) | Peningkatan `imgsz` dari `640` $\rightarrow$ `960` | Resolusi `960` membantu fitur spasial objek kecil (`belonging` dan `person` jarak jauh) sehingga mAP@0.5 meningkat tanpa mengganti arsitektur menjadi `yolo11m`. |

**Tabel 3.5 — Lingkungan dan Konfigurasi Pelatihan & Evaluasi**

| Komponen | Spesifikasi |
| :--- | :--- |
| **Platform Anotasi** | Roboflow (ekspor format YOLO 2 kelas) |
| **Platform Pelatihan** | Kaggle Notebook (`GPU T4 x2`) |
| **Lingkungan Evaluasi Lokal** | CPU `AMD Ryzen 7 8840HS w/ Radeon 780M Graphics`, `Python 3.12.7`, `PyTorch 2.15.0+cpu`, `Ultralytics 8.4.215` |
| **Varian Model** | `yolo11s.pt` (M0, M1, Iterasi 1, Iterasi 2 / M2) dan `yolo11m.pt` (Iterasi 3 pembanding) |
| **Hiperparameter Utama M2** | `epochs=100`, `patience=30`, `batch=8`, `imgsz=960`, `lr0=0.01`, `optimizer=auto`, `cls_remap=true` |
| **Seed Reproduktibilitas** | `seed=0` dan `seed=42` untuk setiap skenario |

**Tabel 3.6 — Ringkasan Percobaan Lengkap (Validasi Kaggle `tabel_percobaan_uas.csv`)**

| ID Eksperimen | Model | Perubahan / Skenario | Seed | Epoch Selesai | `imgsz` | Waktu Latih | Precision (Val) | Recall (Val) | mAP@0.5 (Val) | mAP@0.5:0.95 (Val) | Catatan |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_seed0` | `yolo11s.pt` | Baseline FT | 0 | 100 | 640 | 17.5 mnt | 0.7832 | 0.8211 | 0.8451 | 0.4569 | **Model M1 (`weights/m1_best.pt`)** |
| `m1_seed42` | `yolo11s.pt` | Baseline FT | 42 | 98 | 640 | 17.4 mnt | 0.7955 | 0.7920 | 0.8421 | 0.4572 | Uji reproduktibilitas M1 |
| `iter1_aug_seed0` | `yolo11s.pt` | Augmentasi | 0 | 100 | 640 | 17.7 mnt | 0.8349 | 0.7920 | 0.8395 | 0.4563 | Precision val naik, mAP@0.5 sedikit turun |
| `iter1_aug_seed42` | `yolo11s.pt` | Augmentasi | 42 | 100 | 640 | 17.8 mnt | 0.8142 | 0.7924 | 0.8437 | 0.4585 | Uji reproduktibilitas Iterasi 1 |
| `iter2_imgsz_seed0` | `yolo11s.pt` | Resolusi `960` | 0 | 100 | 960 | 35.7 mnt | 0.7843 | 0.8159 | 0.8402 | **0.4596** | **Model M2 Final (`best.pt`)** — mAP@0.5:0.95 val tertinggi |
| `iter2_imgsz_seed42` | `yolo11s.pt` | Resolusi `960` | 42 | 97 | 960 | 34.6 mnt | 0.8222 | 0.7761 | 0.8427 | 0.4529 | Uji reproduktibilitas Iterasi 2 |
| `iter3_m2_seed0` | `yolo11m.pt` | Model `yolo11m` + `960` | 0 | 100 | 960 | 78.8 mnt | 0.7947 | 0.7945 | 0.8413 | 0.4563 | Waktu latih >2x lebih lama, mAP tidak lebih baik dari `yolo11s` 960 |
| `iter3_m2_seed42` | `yolo11m.pt` | Model `yolo11m` + `960` | 42 | 100 | 960 | 78.9 mnt | 0.8061 | 0.7875 | 0.8376 | 0.4589 | Uji reproduktibilitas Iterasi 3 |

---

### 5.4 Bahan Bab 4 (Tabel 4.1, 4.2, Kontribusi Perbaikan, & Kecepatan)

**Tabel 4.1 — Perbandingan Kinerja pada Set Test (`dataset/test/` = 220 gambar, `imgsz=960`, `conf=0.25`)**

| Model | Kelas | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inferensi (ms/gambar) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** (`yolo11s.pt`) | Semua | 0.4203 | 0.1828 | 0.1743 | 0.1049 | 129.92 | 7.70 |
| **M0** (`yolo11s.pt`) | `person` | 0.8407 | 0.3656 | 0.3487 | 0.2098 | — | — |
| **M0** (`yolo11s.pt`) | `belonging` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |
| **M1** (`weights/m1_best.pt`) | Semua | 0.6153 | 0.6127 | 0.5297 | 0.3146 | 125.34 | 7.98 |
| **M1** (`weights/m1_best.pt`) | `person` | 0.8057 | 0.6722 | 0.6385 | 0.4091 | — | — |
| **M1** (`weights/m1_best.pt`) | `belonging` | 0.4249 | 0.5531 | 0.4209 | 0.2202 | — | — |
| **M2** (`best.pt`) | Semua | **0.7768** | 0.6121 | **0.5968** | **0.3719** | **125.03** | **8.00** |
| **M2** (`best.pt`) | `person` | **0.8600** | 0.6655 | **0.6722** | **0.4461** | — | — |
| **M2** (`best.pt`) | `belonging` | **0.6936** | **0.5587** | **0.5214** | **0.2977** | — | — |

**Bahan Gambar Bab 4:**
* **Kurva Pelatihan & Confusion Matrix:** Tersedia di `results/kaggle_output/prasmul-library-cv/runs/detect/runs/uas/m1_seed0/` dan `iter2_imgsz_seed0/` (`results.png`, `confusion_matrix.png`, `BoxPR_curve.png`).
* **Perbandingan Visual M0 vs M2:** Tersedia 12 gambar berdampingan di `results/galleries/m0_fail_m2_success/` (`Rabu_Meja3_005.jpg`, `008.jpg`, `016.jpg`, `019.jpg`, `021.jpg`, `025.jpg`, `026.jpg`, `027.jpg`, `028.jpg`, `029.jpg`, `030.jpg`, `031.jpg`).

**Tabel 4.2 — Analisis Kesalahan Model Terbaik M2 (24 Sampel dari `results/galleries/m2_errors/`)**

| Kategori Kesalahan | Jumlah Sampel | Contoh File (`results/galleries/m2_errors/`) | Dugaan Penyebab |
| :--- | :---: | :--- | :--- |
| Objek terlewat (`objek_terlewat` / *False Negative*) | **13** | `Rabu_Meja3_001.jpg`, `Rabu_Meja3_002.jpg`, `Rabu_Meja3_008.jpg`, `Rabu_Meja3_016.jpg` | Oklusi parsial oleh meja/rak/pengunjung lain serta objek `belonging` berukuran sangat kecil di sudut jauh kamera. |
| Kotak tidak pas (`kotak_tidak_pas` / *Localization Error*) | **7** | `Rabu_Meja3_012.jpg`, `Rabu_Meja3_013.jpg`, `Rabu_Meja3_015.jpg`, `Rabu_Meja3_017.jpg` | Batas tubuh atau barang yang saling berimpitan menyebabkan IoU bergeser atau kotak kurang presisi. |
| Deteksi palsu (`deteksi_palsu` / *False Positive*) | **4** | `Rabu_Meja3_005.jpg`, `Rabu_Meja3_009.jpg`, `Rabu_Meja3_010.jpg`, `Rabu_Meja3_011.jpg` | Tekstur kursi, bayangan, atau benda di meja menyerupai ciri visual `belonging` atau bagian tubuh pengunjung. |
| **Total Sampel Galeri Error M2** | **24** | Lihat `results/galleries/gallery_summary.csv` | — |

**Tabel Kontribusi Tiap Perbaikan (Tabel 4.3 / Bagian 4.3)**

| Perubahan | mAP@0.5 Sebelum | mAP@0.5 Sesudah | Selisih | Biaya (Waktu Latih & Kecepatan Inferensi) |
| :--- | :---: | :---: | :---: | :--- |
| **M0 $\rightarrow$ M1:** Fine-tuning `yolo11s.pt` (`imgsz=640`, dataset Prasmul) | 0.1743 | 0.5297 | **+0.3554** | Waktu latih 17.5 menit di GPU T4 x2; inferensi CPU 125.34 ms (7.98 FPS) |
| **M1 $\rightarrow$ M2:** Peningkatan resolusi pelatihan `imgsz=640` $\rightarrow$ `960` | 0.5297 | 0.5968 | **+0.0671** | Waktu latih 35.7 menit di GPU T4 x2; inferensi CPU tetap ~125.03 ms (8.00 FPS) |

*Catatan Pembahasan Kecepatan:* Pada perangkat CPU lokal (`AMD Ryzen 7 8840HS`), M2 berjalan pada **125.03 ms/gambar (~8.00 FPS)**. Meskipun berada di bawah ambang video penuh 15–30 FPS tanpa GPU, aplikasi Streamlit mengatasi hal ini secara praktis dengan parameter **Frame Skipping** (default `5` frame) sehingga pemantauan ruang baca tetap responsif secara *near real-time*.

---

### 5.5 Poin Teknis Bab 5 (Kesimpulan & Keterbatasan)

* **Kesimpulan 1 (Baseline M0):** Pada set test (220 gambar), model pretrained COCO M0 hanya mencapai **mAP@0.5 = 0.1743** (`person` 0.3487, `belonging` 0.0000), membuktikan model umum tidak memadai untuk pemantauan CCTV Perpustakaan Prasmul.
* **Kesimpulan 2 (M1 & M2):** Fine-tuning M1 meningkatkan **mAP@0.5** menjadi **0.5297 (+0.3554)**, dan peningkatan resolusi M2 (`imgsz=960`) menaikkannya lagi menjadi **0.5968 (+0.0671 terhadap M1; +0.4225 total terhadap M0)**, dengan lonjakan besar pada kelas `belonging` (mAP@0.5 dari **0.0000 $\rightarrow$ 0.4209 $\rightarrow$ 0.5214**, precision **0.6936**).
* **Keterbatasan Teknis Dataset & Model:**
  * Ketidakseimbangan kelas (*class imbalance*): jumlah anotasi `person` (**8.350**) jauh lebih besar daripada `belonging` (**995**).
  * Kasus oklusi meja/rak dan objek kecil di kejauhan masih menyumbang **13 dari 24 sampel** error utama pada galeri M2.
  * Inferensi di CPU non-GPU mencapai **~8.00 FPS** sehingga bergantung pada *frame skipping* atau akselerasi GPU bila ingin memproses banyak kamera sekaligus secara penuh.
  * *(Catatan untuk anggota tim: Data `Tabel 4.4` hasil hitung manual video demo dan pengisian daftar pustaka/lampiran non-teknis dapat dilengkapi setelah menjalankan demo localhost bersama).*

