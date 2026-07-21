# Klasifikasi Aksara Jawa Berbasis YOLO26

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20(ROCm)-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

Klasifikasi 20 karakter dasar **Aksara Jawa** menggunakan arsitektur
**YOLO26 (nano-classification)** yang di-*fine-tune* dengan PyTorch. Selain
melatih dan mengevaluasi model, repositori ini fokus pada **visualisasi dan
interpretasi filter konvolusi**: merekam evolusi bobot kernel di setiap *epoch*,
membedah transformasi matriks pada tiap blok arsitektur, dan menganalisis
pergeseran probabilitas *softmax* pada karakter yang bentuknya mirip.

> Repositori kode untuk skripsi *"Klasifikasi Aksara Jawa Berbasis YOLO26 dengan
> Visualisasi dan Interpretasi Filter Konvolusi"* — Harry Mardika (50422657).

## Daftar Isi

- [Klasifikasi Aksara Jawa Berbasis YOLO26](#klasifikasi-aksara-jawa-berbasis-yolo26)
  - [Daftar Isi](#daftar-isi)
  - [Tentang Proyek](#tentang-proyek)
  - [Fitur Utama](#fitur-utama)
  - [Struktur Proyek](#struktur-proyek)
  - [Kebutuhan Sistem](#kebutuhan-sistem)
  - [Instalasi](#instalasi)
  - [Cara Penggunaan](#cara-penggunaan)
  - [Struktur Dataset](#struktur-dataset)
  - [Hasil \& Output](#hasil--output)
  - [Konfigurasi](#konfigurasi)
  - [Troubleshooting](#troubleshooting)
  - [Referensi](#referensi)
  - [Lisensi \& Kontributor](#lisensi--kontributor)

## Tentang Proyek

Aksara Jawa punya banyak karakter dengan guratan yang mirip (misalnya `ta`,
`nya`, `dha`), sehingga rawan tertukar saat diklasifikasi otomatis. Proyek ini
mengambil *backbone* klasifikasi YOLO26 nano yang sudah dilatih di ImageNet,
lalu mengganti *layer* linear terakhirnya agar mengeluarkan 20 kelas Aksara Jawa
dan melatih ulang seluruh jaringan pada citra aksara.

Yang membedakan proyek ini dari klasifikasi biasa adalah sisi *interpretability*.
Selama pelatihan, bobot setiap *layer* `Conv2d` direkam per *epoch* ke
`kernel_history.pkl`, distribusi probabilitas 20 kelas untuk sekumpulan citra
acuan dicatat sepanjang waktu, dan bobot/gradien/*feature map* dikirim ke
TensorBoard. Setelah pelatihan, kode membedah aliran tensor dari `[3, 224, 224]`
sampai vektor keputusan `[20]` blok demi blok (Conv → C3k2 → C2PSA → Classify)
untuk menjelaskan *bagaimana* dan *mengapa* model sampai pada keputusannya.

## Fitur Utama

- **Preprocessing dataset otomatis** — unduh dataset via `kagglehub`, *auto-crop*
  latar putih dengan Otsu thresholding, lalu *resize* + *padding* (letterbox)
  tanpa distorsi ke citra persegi 224×224 PNG *lossless*, diparalelkan ke seluruh
  core CPU.
- **Fine-tuning YOLO26** — mengganti *head* linear ImageNet (1000 kelas) menjadi
  20 kelas Aksara Jawa; pelatihan dengan AdamW, `CosineAnnealingLR`, *mixed
  precision* (AMP), *class weighting* untuk data tidak seimbang, *gradient
  clipping*, dan *early stopping*.
- **Evaluasi** — `classification_report` (precision/recall/F1 per kelas) dan
  *confusion matrix* pada data uji.
- **Perekaman evolusi kernel** — menyimpan bobot seluruh *layer* `Conv2d` di
  tiap *epoch* ke `kernel_history.pkl`, lalu mengekspor evolusi nilai matriks
  kernel per *layer* ke berkas teks (`kernels/*.txt`).
- **Visualisasi "Probability Shift"** — melacak probabilitas *softmax* 20 kelas
  pada citra acuan sepanjang *epoch* untuk melihat model menggeser keyakinannya
  menjauh dari kelas yang salah.
- **Bedah arsitektur blok-per-blok** — memasang *forward hook* pada 11 blok dan
  memvisualisasikan *feature map* tiap tahap, dari ekstraksi tepi sampai
  mekanisme *attention* (C2PSA) dan *Classify head*.
- **Logging TensorBoard** — *scalar* (loss/akurasi/LR), histogram bobot &
  gradien, grid citra bobot & *feature map*, serta graf arsitektur model.
- **Graf komputasi** — merender graf komputasi model dengan `torchviz`.

## Struktur Proyek

```text
aksara-jawa/
├── preprocess.ipynb        # Unduh + preprocessing dataset (crop, resize, pad)
├── data.ipynb              # EDA: distribusi kelas dataset
├── model.ipynb             # Training, evaluasi, probability-shift, bedah blok
├── kernel.ipynb            # Muat kernel_history.pkl → tabel & ekspor kernels/*.txt
├── tensorboard.ipynb       # Menampilkan log TensorBoard
├── preprocess.txt          # Ekspor skrip dari preprocess.ipynb
├── model.py                # Ekspor skrip dari model.ipynb
├── kernel.txt              # Ekspor skrip dari kernel.ipynb
├── architecture.md         # Ringkasan arsitektur (output torchinfo.summary)
├── yolo26n-cls.pt          # Bobot pretrained YOLO26 nano-classification
├── kernel_history.pkl      # Rekaman bobot kernel seluruh Conv2d per epoch (~60 MB)
├── Guide.pdf               # Panduan / catatan pendukung
├── data/                   # Dataset (di-.gitignore)
│   ├── v3/v3/{train,val}/<kelas>/   # Sumber mentah dari kagglehub
│   ├── prediction/prediction/       # Sumber mentah data uji (folder datar)
│   └── final/                       # Hasil preprocessing (dipakai training)
│       ├── train/<kelas>/*.png
│       ├── val/<kelas>/*.png
│       └── test/*.png               # Folder datar, label dari nama file
├── model/
│   └── best_model.pt       # Checkpoint model terbaik (disimpan saat val acc naik)
├── kernels/                # Evolusi nilai kernel per layer (*.txt)
├── images/                 # Graf komputasi & plot progres pelatihan
└── runs/                   # Log TensorBoard (yolo26-aksara_<timestamp>/)
```

> Folder `data/`, seluruh `*.pdf`, `yolo26n-cls.pt`, dan `kernels/` tercantum di
> `.gitignore`.

## Kebutuhan Sistem

| Komponen        | Versi / Keterangan                                             |
| --------------- | ------------------------------------------------------------- |
| Python          | 3.12                                                          |
| GPU             | AMD GPU dengan **ROCm 7.2** (kode diuji pada backend ROCm)     |
| PyTorch         | 2.11.0+rocm7.2 (`torch.cuda.*` dipakai sebagai abstraksi ROCm) |
| Graphviz        | Paket sistem `graphviz` (dibutuhkan `torchviz` untuk render)   |
| Package manager | `micromamba` / `conda` (opsional) + `pip`                      |

> **Catatan GPU:** Kode ditulis untuk AMD ROCm, tetapi tetap berjalan di CPU —
> `device` dipilih otomatis (`cuda` jika tersedia, jika tidak `cpu`). Untuk
> pengguna NVIDIA, pasang build PyTorch CUDA yang sesuai; API di kode tidak
> perlu diubah.

Dependensi utama yang benar-benar dipakai di kode:

| Library                                | Kegunaan                                  |
| -------------------------------------- | ----------------------------------------- |
| `torch`, `torchvision`                 | Model, training, transforms, DataLoader   |
| `ultralytics`                          | Memuat arsitektur YOLO26 (`yolo26n-cls`)  |
| `torchinfo`                            | Ringkasan arsitektur & bentuk tensor      |
| `torchviz`                             | Render graf komputasi (butuh `graphviz`)  |
| `tensorboard`                          | Logging metrik, histogram, citra          |
| `scikit-learn`                         | `classification_report`, confusion matrix |
| `opencv-python`                        | Auto-crop & resize/pad preprocessing      |
| `Pillow`                               | Membaca citra pada dataset prediksi       |
| `kagglehub`                            | Mengunduh dataset dari Kaggle             |
| `numpy`, `pandas`                      | Manipulasi matriks & tabel kernel         |
| `matplotlib`, `seaborn`                | Plot metrik, probability shift, EDA       |
| `tqdm`                                 | Progress bar pelatihan                    |
| `jupyter` / `ipython`                  | Menjalankan notebook                      |

## Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/harrymardika/aksara-jawa.git
cd aksara-jawa

# 2. Aktifkan environment (contoh dengan micromamba)
micromamba activate pytorch
# atau buat environment baru:
# micromamba create -n pytorch python=3.12
# micromamba activate pytorch

# 3. Pasang PyTorch + torchvision untuk ROCm (sesuaikan dengan GPU Anda)
#    Lihat: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Pasang dependensi lain yang dipakai
pip install ultralytics torchinfo torchviz tensorboard \
            scikit-learn opencv-python pillow kagglehub \
            numpy pandas matplotlib seaborn tqdm jupyter

# 5. Paket sistem untuk render graf komputasi (torchviz)
#    Fedora : sudo dnf install graphviz
#    Ubuntu : sudo apt install graphviz
```

Bobot pretrained `yolo26n-cls.pt` sudah disertakan di repositori, sehingga tidak
perlu diunduh terpisah. Dataset disiapkan melalui langkah preprocessing di bawah.

## Cara Penggunaan

Seluruh alur kerja dijalankan melalui **Jupyter Notebook** (tidak ada antarmuka
*command-line*/argparse). Berkas `.py`/`.txt` di root adalah hasil ekspor skrip
dari notebook dan berguna sebagai rujukan pembacaan kode. Jalankan notebook
secara berurutan:

```bash
jupyter notebook   # atau: jupyter lab
```

**1. Preprocessing dataset — `preprocess.ipynb`**
Mengunduh dataset `phiard/aksara-jawa` via `kagglehub`, lalu meng-*crop*,
me-*resize*, dan mem-*pad* seluruh citra ke 224×224 PNG. Direktori diatur di
blok `__main__`:

```python
INPUT_DATA_DIR  = "data/v3/v3"                  # sumber train/val mentah
OUTPUT_DATA_DIR = "data/final"                  # hasil train/val (dipakai training)
TEST_INPUT_DIR  = "data/prediction/prediction"  # sumber data uji (folder datar)
TEST_OUTPUT_DIR = "data/final/test"             # hasil data uji
FINAL_RESOLUTION = 224
```

**2. Eksplorasi data — `data.ipynb`** *(opsional)*
Menampilkan distribusi jumlah citra per kelas.

**3. Pelatihan & analisis — `model.ipynb`**
Memuat `yolo26n-cls.pt`, mengganti *head* linear menjadi 20 kelas, lalu melatih
model. Selama pelatihan otomatis dihasilkan: checkpoint `model/best_model.pt`,
graf komputasi `images/model_computation_graph.png`, log TensorBoard di `runs/`,
dan rekaman kernel `kernel_history.pkl`. Sel-sel berikutnya di notebook yang sama
menjalankan **evaluasi** (`classification_report` + *confusion matrix* pada
`data/final/test`), plot **probability shift**, dan **bedah 11 blok** arsitektur.

**4. Interpretasi kernel — `kernel.ipynb`**
Memuat `kernel_history.pkl`, menampilkan evolusi nilai matriks kernel sebagai
tabel Pandas berwarna, dan mengekspor evolusi per *layer* ke `kernels/*.txt`
(diparalelkan). *Layer* perwakilan yang dianalisis, mis. `model.0.conv`,
`model.2.m.0.cv1.conv`, `model.9.m.0.attn.qkv.conv`, `model.10.conv.conv`.

**5. Memantau pelatihan — TensorBoard**

```bash
tensorboard --logdir runs
```

## Struktur Dataset

Skrip pelatihan (`model.ipynb`/`model.py`) memakai
`torchvision.datasets.ImageFolder` untuk **train** dan **val**, sehingga setiap
kelas harus berupa subfolder:

```text
data/final/
├── train/
│   ├── ba/  ca/  da/  dha/  ga/  ha/  ja/  ka/  la/  ma/
│   └── na/  nga/ nya/ pa/  ra/  sa/  ta/  tha/ wa/  ya/
├── val/
│   └── (struktur subfolder kelas yang sama seperti train)
└── test/
    └── ba17.png  ca2.png  da5.png  ...   (folder DATAR, tanpa subfolder)
```

Terdapat **20 kelas**: `ba, ca, da, dha, ga, ha, ja, ka, la, ma, na, nga, nya,
pa, ra, sa, ta, tha, wa, ya`.

Data **test** disusun berbeda: file diletakkan langsung dalam satu folder, dan
label diambil dari awalan huruf nama berkas lewat regex (`ba17.png` → `ba`,
`nga220.pred.png` → `nga`). Awalan yang tidak termasuk 20 kelas akan diberi
peringatan dan dilewati.

## Hasil & Output

| Output                                   | Lokasi                                    |
| ---------------------------------------- | ----------------------------------------- |
| Checkpoint model terbaik                 | `model/best_model.pt`                     |
| Rekaman bobot kernel per epoch           | `kernel_history.pkl`                      |
| Evolusi nilai kernel per layer           | `kernels/*.txt`                           |
| Graf komputasi model                     | `images/model_computation_graph.png`      |
| Plot akurasi & loss pelatihan            | ditampilkan di notebook (`images/`)       |
| Plot *probability shift* (per & 20 kelas)| ditampilkan di notebook                   |
| *Classification report* & confusion matrix | ditampilkan di notebook                 |
| Log TensorBoard                          | `runs/yolo26-aksara_<timestamp>/`         |

`model/best_model.pt` adalah *dictionary* berisi `epoch`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, dan `best_acc`.

**Hasil pelatihan (berhenti di epoch 10 karena early stopping):**

| Metrik                          | Nilai   |
| -------------------------------- | ------- |
| Peak validation accuracy         | 99.58%  |
| Train accuracy (epoch terbaik)   | 99.44%  |
| Train loss (epoch terbaik)       | 0.0230  |
| Val loss (epoch terbaik)         | 2.0896  |

**Classification report pada data `data/final/test`** (25 citra, 20 kelas):

| Metrik        | Nilai  |
| ------------- | ------ |
| Accuracy      | 96.00% |
| Macro avg F1  | 0.9733 |
| Weighted avg F1 | 0.9627 |

Sebagian besar kelas mencapai precision/recall/F1 sempurna (1.0000); pengecualian
utama adalah kelas `ca` (precision 0.5000, recall 1.0000, F1 0.6667, dari hanya
1 sampel uji) dan `ma` (precision 1.0000, recall 0.6667, F1 0.8000, dari 3
sampel uji) — mengindikasikan set uji per kelas yang sangat kecil sehingga satu
kesalahan berdampak besar pada metrik kelas tersebut.

> Angka di atas berasal dari satu *run* pelatihan tersimpan di `model.ipynb`.
> Karena inisialisasi dan augmentasi bersifat acak, hasil run Anda sendiri bisa
> sedikit berbeda.

## Konfigurasi

Parameter diubah langsung di dalam notebook (tidak ada file konfigurasi
terpisah). Parameter penting di `model.ipynb`:

| Parameter                | Nilai                        | Keterangan                          |
| ------------------------ | ---------------------------- | ----------------------------------- |
| `batch_size`             | `32`                         | Ukuran batch train & val            |
| `num_epochs`             | `30`                         | Jumlah epoch maksimum               |
| Optimizer                | `AdamW(lr=7e-5, weight_decay=1e-3)` | Optimizer                    |
| Scheduler                | `CosineAnnealingLR(T_max=30)`| Penjadwalan learning rate           |
| `patience`               | `3`                          | Early stopping (epoch tanpa perbaikan) |
| `log_image_interval`     | `5`                          | Interval epoch untuk log citra ke TensorBoard |
| `max_norm`               | `1.0`                        | Gradient clipping                   |
| Class weights            | dihitung dari `train_dataset.targets` | Menangani ketidakseimbangan kelas |
| Input size / Normalize   | `224×224`, mean/std ImageNet | Transform train & val               |
| Augmentasi (train)       | `RandomRotation(15, fill=255)`, `RandomAffine(translate=0.1, scale=(0.8,1.2), fill=255)` | Rotasi & pergeseran ringan |

Path dataset di `model.ipynb`: `train_dir = "./data/final/train"`,
`val_dir = "./data/final/val"`, `pred_dir = "./data/final/test"`; bobot dimuat
dari `model/best_model.pt`.

Resolusi & path preprocessing diatur di blok `__main__` `preprocess.ipynb`
(lihat [Cara Penggunaan](#cara-penggunaan)).

## Troubleshooting

- **AMD GPU / ROCm.** Kode memakai API `torch.cuda.*`, `torch.autocast(device_type="cuda")`,
  dan `torch.amp.GradScaler("cuda")` sebagai abstraksi ROCm — ini normal pada
  build PyTorch ROCm dan tidak berarti butuh perangkat NVIDIA. Cek deteksi GPU
  dengan `torch.version.hip` dan `torch.cuda.is_available()` (dicetak di sel
  pertama). Tanpa GPU, model otomatis jatuh ke CPU.
- **`torchviz` gagal me-render graf.** Pastikan paket sistem `graphviz`
  terpasang (`dnf`/`apt install graphviz`), bukan hanya modul Python-nya.
- **File kernel tidak ditemukan.** `kernel.ipynb` mencari `kernel_history.pkl`,
  lalu fallback ke `kernel_history_all.pkl`. File ini baru ada setelah
  `model.ipynb` selesai melatih dan menyimpannya.

## Referensi

- Ultralytics YOLO — https://docs.ultralytics.com
- Dataset: `phiard/aksara-jawa` (Kaggle, via `kagglehub`)
- `Guide.pdf` dan `architecture.md` — dokumen pendukung dalam repositori

## Lisensi & Kontributor

Dirilis di bawah lisensi **MIT** — lihat berkas [LICENSE](LICENSE).

- **Penulis:** Harry Mardika (NPM 50422657)
- **Dosen Pembimbing:** Dr. Ericks Rachmat Swedia, ST., MMSI.
