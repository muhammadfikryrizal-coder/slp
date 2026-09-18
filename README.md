# Assignment 1 — Single Layer Perceptron (Iris)

Single Layer Perceptron untuk klasifikasi biner data Iris:
**Iris-setosa = 0** vs **Iris-versicolor = 1**.

Tugas dikerjakan dua kali dengan cara berbeda — spreadsheet (Google Sheets) dan
Python — dan keduanya menghasilkan angka yang **identik**.

## Konfigurasi

| Parameter | Nilai |
|---|---|
| Fungsi aktivasi | Sigmoid, `g(z) = 1 / (1 + e^-z)` |
| Loss | Mean Squared Error, `(g(z) - target)^2` |
| Optimizer | Gradient descent per sampel (stochastic) |
| Learning rate | 0.1 |
| Bobot awal | bias = teta1 = teta2 = teta3 = teta4 = 0.5 |
| Epoch | 5 |
| Data training | 80 sampel (baris 1–40 dan 51–90) |
| Data validation | 20 sampel (baris 41–50 dan 91–100) |

Pada validasi bobot **tidak** di-update. Bobot diambil dari hasil training pada
akhir epoch yang bersangkutan, lalu dipakai konstan untuk seluruh 20 sampel.

## Hasil

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 1 | 0.449889 | 52.50% | 0.328951 | 50.00% |
| 2 | 0.037452 | 95.00% | 0.247289 | 50.00% |
| 3 | 0.024372 | 97.50% | 0.175892 | 50.00% |
| 4 | 0.017357 | 97.50% | 0.119381 | 85.00% |
| 5 | 0.012740 | 98.75% | 0.081581 | 100.00% |

Bobot akhir: `bias = 0.257720`, `teta1 = -0.241802`, `teta2 = -0.416903`,
`teta3 = 1.122214`, `teta4 = 0.824059`.

Loss training dan validasi sama-sama menurun di setiap epoch, jadi tidak ada
indikasi overfitting.

## Isi repositori

| Berkas | Keterangan |
|---|---|
| `slp_iris.py` | Implementasi SLP — jawaban utama tugas no. 2 |
| `iris.csv` | 100 baris data Iris (setosa + versicolor) |
| `build_slp_spreadsheet.py` | Mengisi template `PMM-TemplateSLP.xlsx` jadi spreadsheet lengkap |
| `make_charts_gsheet.py` | Menggambar grafik dari nilai yang dihitung spreadsheet |
| `make_ppt.py` | Menyusun slide presentasi 7 halaman |
| `slp_history.csv` | Rekap loss & accuracy per epoch |
| `chart_accuracy_python.png`, `chart_loss_python.png` | Grafik dari Python |
| `chart_accuracy_gsheet.png`, `chart_loss_gsheet.png` | Grafik dari spreadsheet |
| `PMM-SLP-Iris-Filled.xlsx` | Spreadsheet SLP yang sudah terisi penuh |
| `Assignment1_SLP.pptx` | Slide presentasi |

## Cara menjalankan

```bash
pip install numpy matplotlib openpyxl python-pptx pillow
python slp_iris.py
```

Untuk membangun ulang spreadsheet, grafik, dan slide:

```bash
python build_slp_spreadsheet.py PMM-TemplateSLP.xlsx PMM-SLP-Iris-Filled.xlsx
python make_charts_gsheet.py PMM-SLP-Iris-Filled.xlsx
python make_ppt.py --nama "Nama Lengkap" --nim "NIM" --gsheet "URL" --github "URL"
```

## Pemetaan rumus spreadsheet ke kode

| Kolom | Rumus spreadsheet | Fungsi Python |
|---|---|---|
| N | `=I5+J5*C5+K5*D5+L5*E5+F5*M5` | `forward()` |
| O | `=1/(1+EXP(-N5))` | `sigmoid()` |
| P | `=IF(O5>0.5,1,0)` | `pred = 1.0 if g > 0.5 else 0.0` |
| Q | `=O5-G5` | `error = g - target` |
| R | `=Q5^2` | `sse_list.append(error ** 2)` |
| S | `=IF(P5=G5,1,0)` | `correct_list.append(...)` |
| T | `=2*(O5-G5)*(1-O5)*O5` | `gradient()` elemen pertama |
| U–X | `=T5*C5` … `=T5*F5` | `gradient()` elemen berikutnya |
| I–M | `=I5-$K$2*T5` … | `w = w - lr * gradient(...)` |
