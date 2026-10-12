import re
import numpy as np
import pandas as pd
import matplotlib

DI_NOTEBOOK = "get_ipython" in globals()     # True bila dijalankan di Jupyter / Colab
if not DI_NOTEBOOK:
    # Skrip biasa: grafik hanya disimpan ke PNG, jadi dipakai backend tanpa jendela (Agg).
    # Ini mencegah galat tkinter "main thread is not in main loop" yang sering muncul di Windows.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn import set_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    display                                  
except NameError:
    def display(x): print(x)                 


# ------------------------------------------------------------------------------
# Fungsi bantu dan pengaturan
# ------------------------------------------------------------------------------
def bagian(judul, utama=True):
    # Cetak judul langkah dengan garis pemisah agar hasil running mudah dibaca
    garis = ("=" if utama else "-") * 80
    print(f"\n{garis}\n{judul}\n{garis}")

def simpan_gambar(nama):
    plt.savefig(nama, dpi=130, bbox_inches="tight")
    if DI_NOTEBOOK:
        plt.show()
    plt.close()

def tampil_transpose(tabel):
    t = tabel.T.copy()
    t.columns = [f"Baris {i}" for i in tabel.index]      
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else f"{v:.3f}"
        return str(v)
    print(t.apply(lambda kolom: kolom.map(fmt)).to_string(justify="right"))

set_config(transform_output="pandas")        
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
SEED = 42
WARNA = {"LGG": "#2F6DB5", "GBM": "#E8832A"}


# ------------------------------------------------------------------------------
# Langkah 1. Pengumpulan data
# ------------------------------------------------------------------------------
bagian("Langkah 1. Pengumpulan data")
# encoding="utf-8-sig" membuang karakter BOM tersembunyi pada nama kolom pertama ('Grade')
df_raw = pd.read_csv("TCGA_GBM_LGG_Mutations_all.csv", encoding="utf-8-sig")
df = df_raw.copy()
print("Ukuran data (baris, kolom):", df.shape)
print("Setiap baris = 1 pasien; kolom pertama 'Grade' = target.")
display(df.head())
df.info()
bagian("5 data teratas (data asli)", utama=False);  tampil_transpose(df_raw.head(5))
bagian("5 data terbawah (data asli)", utama=False); tampil_transpose(df_raw.tail(5))


# ------------------------------------------------------------------------------
# Langkah 2. Jenis data dan data terstruktur
# ------------------------------------------------------------------------------
bagian("Langkah 2. Jenis data dan data terstruktur")
ringkas = pd.DataFrame({"tipe_terbaca": df.dtypes.astype(str),
                        "jumlah_nilai_unik": df.nunique(),
                        "contoh_nilai": [df[c].iloc[0] for c in df.columns]})
display(ringkas)


# ------------------------------------------------------------------------------
# Langkah 3. Pembersihan data tahap A (aturan tetap, tanpa statistik)
# ------------------------------------------------------------------------------
bagian("Langkah 3. Pembersihan data tahap A (aturan tetap, tanpa statistik)")

# 3.1 Missing value tersembunyi
bagian("3.1 Missing value tersembunyi", utama=False)
print("Missing menurut isna():", int(df.isna().sum().sum()), " <- menyesatkan, nilai kosong ditulis sebagai teks")
PLACEHOLDER = ["--", "not reported"]
ph = df.isin(PLACEHOLDER).sum()
tabel_missing = pd.DataFrame({"jumlah": ph[ph > 0], "persen": (ph[ph > 0] / len(df) * 100).round(2)})
print("\nMissing value tersembunyi (placeholder teks):")
display(tabel_missing)
n_bermasalah = int(df[["Gender", "Age_at_diagnosis", "Race"]].isin(PLACEHOLDER).any(axis=1).sum())
print("Baris dengan data klinis tidak lengkap:", n_bermasalah, "dari", len(df))

# Penanda kosong diseragamkan menjadi NaN (pengisiannya di Langkah 5, setelah data dibagi)
df = df.replace({"--": np.nan, "not reported": np.nan})
print("\nNaN setelah penyeragaman:")
print(df.isna().sum()[lambda s: s > 0])


# 3.2 Duplikat (dibersihkan sebelum split agar baris yang sama tidak jatuh ke data latih dan uji)
bagian("3.2 Duplikat", utama=False)
print("\nBaris identik          :", int(df.duplicated().sum()))
print("Case_ID (pasien) ganda :", int(df["Case_ID"].duplicated().sum()))
df = df.drop_duplicates()          # tidak ada yang terhapus, tetapi langkahnya tetap dijalankan
print("Ukuran setelah hapus duplikat:", df.shape)


# 3.3 Inkonsistensi: ejaan, kapitalisasi, spasi, dan tipe data
bagian("3.3 Inkonsistensi: ejaan, kapitalisasi, spasi, dan tipe data", utama=False)
for c in ["Gender", "Race", "Primary_Diagnosis"]:
    s = df[c].dropna()
    n_asli, n_seragam = s.nunique(), s.str.strip().str.lower().nunique()
    print(f"{c:18s}: {n_asli} nilai unik, {n_seragam} setelah strip+huruf kecil ->",
          "konsisten" if n_asli == n_seragam else "ADA variasi")
print("\nNilai unik Grade :", sorted(df["Grade"].unique()))
print("Nilai unik Gender:", sorted(df["Gender"].dropna().unique()))
print("Nilai unik Race  :", sorted(df["Race"].dropna().unique()))
print("Nilai unik 20 gen:", sorted(set(np.concatenate([df[c].unique() for c in df.columns[7:]]))))

# Usia tersimpan sebagai teks ("51 years 108 days") -> angka desimal dalam tahun
usia_teks = df["Age_at_diagnosis"].dropna()
print("\nFormat usia yang tidak sesuai pola penuh:")
print(usia_teks[~usia_teks.str.fullmatch(r"\d+ years \d+ days")].value_counts())

def parse_age(s):
    if pd.isna(s):
        return np.nan
    m = re.fullmatch(r"(\d+) years(?: (\d+) days)?", s.strip())
    return np.nan if m is None else int(m.group(1)) + int(m.group(2) or 0) / 365     # tahun + hari/365

df["Age_at_diagnosis"] = df["Age_at_diagnosis"].apply(parse_age).astype(float)
df.columns = df.columns.str.lower()             # nama kolom diseragamkan: huruf kecil
print("\nUsia sekarang bertipe:", df["age_at_diagnosis"].dtype, "| rentang:",
      round(df["age_at_diagnosis"].min(), 1), "-", round(df["age_at_diagnosis"].max(), 1), "tahun")


# 3.4 Fitur tidak relevan dan indikasi kebocoran data (leakage)
bagian("3.4 Fitur tidak relevan dan kebocoran data", utama=False)
print("\nProject vs Grade:")
display(pd.crosstab(df["project"], df["grade"]))
print("Primary_Diagnosis vs Grade:")
display(pd.crosstab(df["primary_diagnosis"].fillna("(kosong)"), df["grade"]))
y_tmp = (df["grade"] == "GBM").astype(int)
skor = cross_val_score(KNeighborsClassifier(5), pd.get_dummies(df[["project"]]).astype(int), y_tmp,
                       cv=StratifiedKFold(10, shuffle=True, random_state=SEED))
print("Akurasi CV hanya dari kolom Project: %.4f  <- kebocoran target" % skor.mean())

GEN = list(df.columns[7:])                       # 20 kolom status mutasi gen
df = df.drop(columns=["project", "primary_diagnosis", "case_id"])
print("Jumlah fitur:", df.shape[1] - 1, "| kolom:", list(df.columns))


# 3.5 Memahami data: statistik deskriptif
bagian("3.5 Memahami data: statistik deskriptif", utama=False)
dist = df["grade"].value_counts().to_frame("jumlah")
dist["persen"] = (dist["jumlah"] / len(df) * 100).round(2)
print("\nDistribusi kelas:")
display(dist)

# a. Usia saat diagnosis (fitur rasio)
tabel_umur = df.groupby("grade")["age_at_diagnosis"].describe().round(1)
tabel_umur.loc["Semua"] = df["age_at_diagnosis"].describe().round(1)
print("\nStatistik usia per kelas:")
display(tabel_umur)

batas = [0, 29.99, 39.99, 49.99, 59.99, 69.99, 120]
label = ["<30", "30-39", "40-49", "50-59", "60-69", ">=70"]
kel = pd.cut(df["age_at_diagnosis"], batas, labels=label)
tabel_bin = pd.crosstab(kel, df["grade"])
tabel_bin["persen_GBM"] = (tabel_bin["GBM"] / tabel_bin.sum(axis=1) * 100).round(1)
print("Kelompok usia dan proporsi GBM:")
display(tabel_bin)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for g in ["LGG", "GBM"]:
    ax[0].hist(df.loc[df.grade == g, "age_at_diagnosis"].dropna(), bins=range(10, 95, 5), alpha=.65, label=g, color=WARNA[g])
ax[0].set_xlabel("Usia saat diagnosis (tahun)"); ax[0].set_ylabel("Jumlah pasien"); ax[0].legend(); ax[0].set_title("Sebaran usia")
ax[1].boxplot([df.loc[df.grade == g, "age_at_diagnosis"].dropna() for g in ["LGG", "GBM"]])
ax[1].set_xticklabels(["LGG", "GBM"])           
ax[1].set_ylabel("Usia (tahun)"); ax[1].set_title("Boxplot usia per kelas")
plt.tight_layout(); simpan_gambar("fig_umur.png")

# b. Jenis kelamin dan ras
tabel_gender = pd.crosstab(df["gender"], df["grade"])
p_gender = chi2_contingency(tabel_gender)[1]
tabel_gender["persen_GBM"] = (tabel_gender["GBM"] / tabel_gender.sum(axis=1) * 100).round(1)
print("Gender vs Grade (p chi-square = %.3f):" % p_gender)
display(tabel_gender)

tabel_race = pd.crosstab(df["race"], df["grade"])
tabel_race["persen_GBM"] = (tabel_race["GBM"] / tabel_race.sum(axis=1) * 100).round(1)
print("\nRace vs Grade (kategori sangat kecil, uji chi-square tidak andal):")
display(tabel_race)

# c. Status mutasi 20 gen
baris = []
for g in GEN:
    mut = df[g] == "MUTATED"
    tab = pd.crosstab(mut, df["grade"])
    baris.append([g.upper(), int(mut.sum()),
                  round(mut[df.grade == "LGG"].mean() * 100, 1), round(mut[df.grade == "GBM"].mean() * 100, 1),
                  round((mut & (df.grade == "GBM")).sum() / mut.sum() * 100, 1), chi2_contingency(tab)[1]])
tabel_gen = pd.DataFrame(baris, columns=["gen", "jumlah_MUTATED", "%_mutasi_LGG", "%_mutasi_GBM",
                                         "%_GBM_dari_yg_bermutasi", "p_chi2"])
tabel_gen["selisih"] = (tabel_gen["%_mutasi_GBM"] - tabel_gen["%_mutasi_LGG"]).round(1)
tabel_gen = tabel_gen.sort_values("jumlah_MUTATED", ascending=False).reset_index(drop=True)
tabel_gen_tampil = tabel_gen.copy()
tabel_gen_tampil["p_chi2"] = tabel_gen_tampil["p_chi2"].map(lambda p: "<0.001" if p < 0.001 else round(p, 3))
print("\nStatus mutasi 20 gen:")
display(tabel_gen_tampil)

urut = tabel_gen.reindex(tabel_gen["selisih"].abs().sort_values().index)
y_pos = np.arange(len(urut)); h = 0.4
plt.figure(figsize=(8, 7))
plt.barh(y_pos + h / 2, urut["%_mutasi_LGG"], h, label="LGG", color=WARNA["LGG"])
plt.barh(y_pos - h / 2, urut["%_mutasi_GBM"], h, label="GBM", color=WARNA["GBM"])
plt.yticks(y_pos, urut["gen"]); plt.xlabel("% pasien bermutasi"); plt.legend(loc="lower right")
plt.title("Frekuensi mutasi 20 gen: LGG vs GBM (urut menurut selisih)")
plt.tight_layout(); simpan_gambar("fig_gen.png")

# Jumlah gen yang bermutasi per pasien
jml = (df[GEN] == "MUTATED").sum(axis=1)
tabel_jml = jml.groupby(df["grade"]).describe().round(2)
print("\nJumlah gen bermutasi per pasien:")
display(tabel_jml)
print("Pasien tanpa mutasi pada ke-20 gen:", int((jml == 0).sum()))


# ------------------------------------------------------------------------------
# Langkah 4. Pembagian dataset (data latih : data uji = 80% : 20%)
# ------------------------------------------------------------------------------
bagian("Langkah 4. Pembagian dataset (data latih : data uji = 80% : 20%)")
TARGET = "grade"
X = df.drop(columns=TARGET)
y = (df[TARGET] == "GBM").astype(int)            # LGG = 0, GBM = 1
# Berkas asli terurut menurut kelas (LGG dulu, GBM kemudian). train_test_split mengacak baris (shuffle=True)
# dan stratify=y menjaga proporsi kelas pada data latih dan data uji.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y, shuffle=True)
print("\nData latih:", X_train.shape, "| Data uji:", X_test.shape)
print("Proporsi GBM -> seluruh: %.3f | latih: %.3f | uji: %.3f" % (y.mean(), y_train.mean(), y_test.mean()))
print("Latih ->", y_train.map({0: "LGG", 1: "GBM"}).value_counts().to_dict(),
      "| Uji ->", y_test.map({0: "LGG", 1: "GBM"}).value_counts().to_dict())


# ------------------------------------------------------------------------------
# Langkah 5. Pembersihan data tahap B (statistik dipelajari dari data latih)
# ------------------------------------------------------------------------------
bagian("Langkah 5. Pembersihan data tahap B (statistik dari data latih)")

# 5.1 Missing value: median (numerik) dan modus (kategorikal), dihitung dari data latih
bagian("5.1 Missing value", utama=False)
print("\nMissing pada data latih :", X_train.isna().sum()[lambda s: s > 0].to_dict())
print("Missing pada data uji   :", X_test.isna().sum()[lambda s: s > 0].to_dict())
nilai_isi = {"age_at_diagnosis": X_train["age_at_diagnosis"].median(),
             "gender": X_train["gender"].mode()[0],
             "race": X_train["race"].mode()[0]}
print("Nilai pengisi (dari DATA LATIH):",
      {k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in nilai_isi.items()})
X_train = X_train.fillna(nilai_isi)
X_test = X_test.fillna(nilai_isi)
print("Sisa missing -> latih:", int(X_train.isna().sum().sum()), "| uji:", int(X_test.isna().sum().sum()))


# 5.2 Outlier (IQR) pada usia; batas dihitung dari data latih
bagian("5.2 Outlier (IQR) pada usia", utama=False)
q1, q3 = X_train["age_at_diagnosis"].quantile([0.25, 0.75]); iqr = q3 - q1
batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_tr = int(((X_train["age_at_diagnosis"] < batas_bawah) | (X_train["age_at_diagnosis"] > batas_atas)).sum())
n_te = int(((X_test["age_at_diagnosis"] < batas_bawah) | (X_test["age_at_diagnosis"] > batas_atas)).sum())
print(f"\nQ1={q1:.1f}  Q3={q3:.1f}  IQR={iqr:.1f}  batas=[{batas_bawah:.1f}, {batas_atas:.1f}]")
print(f"Outlier -> latih: {n_tr} baris | uji: {n_te} baris | usia termuda: {X['age_at_diagnosis'].min():.1f}, "
      f"tertua: {X['age_at_diagnosis'].max():.1f}")
X_train["age_at_diagnosis"] = X_train["age_at_diagnosis"].clip(batas_bawah, batas_atas)   # capping (tidak mengubah apa pun bila 0 outlier)
X_test["age_at_diagnosis"] = X_test["age_at_diagnosis"].clip(batas_bawah, batas_atas)


# 5.3 Data tidak seimbang
bagian("5.3 Data tidak seimbang", utama=False)
rasio = (y == 0).sum() / (y == 1).sum()
print("\nLGG : GBM = %d : %d  (rasio %.2f : 1)" % ((y == 0).sum(), (y == 1).sum(), rasio))
print("Tergolong ringan -> tanpa resampling. Penanganan: stratify saat split, metrik F1-macro, laporan per kelas.")


# ------------------------------------------------------------------------------
# Langkah 6. Transformasi
# ------------------------------------------------------------------------------
bagian("Langkah 6. Transformasi")

# 6.1 Encoding kategorikal. Kategori ras yang jarang (< 3% data latih) digabung menjadi "lainnya".
bagian("6.1 Encoding kategorikal", utama=False)
frek_race = X_train["race"].value_counts(normalize=True)
print("\nProporsi kategori race pada data latih (%):", (frek_race * 100).round(1).to_dict())
RACE_UMUM = frek_race[frek_race >= 0.03].index.tolist()                  # ditentukan dari DATA LATIH
NAMA_RACE = {"white": "race_white", "black or african american": "race_black"}   # nama kolom ringkas
KOLOM_RACE = [NAMA_RACE.get(r, "race_" + r.replace(" ", "_")) for r in sorted(RACE_UMUM)] + ["race_lainnya"]

def encode(Xin):
    Xo = Xin.copy()
    Xo["gender_female"] = (Xo.pop("gender") == "Female").astype(int)     # Male = 0, Female = 1 (dokumentasi UCI)
    for g in GEN:
        Xo[g] = (Xo[g] == "MUTATED").astype(int)                         # NOT_MUTATED = 0, MUTATED = 1
    race = Xo.pop("race")
    race = race.where(race.isin(RACE_UMUM), "lainnya")                   # kategori jarang atau baru -> "lainnya"
    kolom = race.map(lambda r: NAMA_RACE.get(r, "race_" + r.replace(" ", "_")))
    race_oh = pd.get_dummies(kolom, dtype=int).reindex(columns=KOLOM_RACE, fill_value=0)
    return pd.concat([Xo, race_oh], axis=1)

X_train_enc, X_test_enc = encode(X_train), encode(X_test)
assert list(X_train_enc.columns) == list(X_test_enc.columns), "kolom data latih dan uji harus sama"
print("Kolom setelah encoding (%d):" % X_train_enc.shape[1]); print(list(X_train_enc.columns))
display(X_train_enc.head())


# 6.2 Normalisasi dan standarisasi (hanya usia; fitur lain sudah 0/1)
bagian("6.2 Normalisasi dan standarisasi", utama=False)
kol_usia = ["age_at_diagnosis"]
std = StandardScaler().fit(X_train_enc[kol_usia])
mu, sd = std.mean_[0], std.scale_[0]          # rata-rata dan simpangan baku usia pada data latih
mm = MinMaxScaler().fit(X_train_enc[kol_usia])

def skala(Xe, scaler):
    Xs = Xe.copy()
    if scaler is not None:
        Xs[kol_usia] = scaler.transform(Xe[kol_usia])
    return Xs

X_train_std, X_test_std = skala(X_train_enc, std), skala(X_test_enc, std)          # untuk ditampilkan
X_train_mm = skala(X_train_enc, mm)                                                 # pembanding
X_train_raw = X_train_enc                                                           # pembanding (tanpa scaling)

tabel_scaling = pd.DataFrame({
    "tanpa scaling": X_train_raw["age_at_diagnosis"].describe(),
    "normalisasi min-max": X_train_mm["age_at_diagnosis"].describe(),
    "standarisasi z-score": X_train_std["age_at_diagnosis"].describe()}).loc[["min", "max", "mean", "std"]].round(3)
print("\nUsia pada data latih menurut perlakuan scaling:")
display(tabel_scaling)


# ------------------------------------------------------------------------------
# Langkah 7. Hasil pra-pemrosesan: 5 data teratas dan 5 data terbawah
# ------------------------------------------------------------------------------
bagian("Langkah 7. Hasil pra-pemrosesan: 5 data teratas dan 5 data terbawah")

# Yang ditampilkan adalah data latih hasil pemrosesan. Urutan barisnya sudah diacak oleh train_test_split,
# jadi bukan lima baris pertama dan terakhir berkas asli (nomor baris asli tetap tertera pada nama kolom).
hasil_latih = X_train_std.copy()
hasil_latih["grade"] = y_train
print("Ukuran data latih hasil pra-proses:", hasil_latih.shape, "| total NaN:", int(hasil_latih.isna().sum().sum()),
      "| semua kolom numerik:", all(pd.api.types.is_numeric_dtype(t) for t in hasil_latih.dtypes))
print("Data uji diproses dengan aturan yang sama:", X_test_std.shape)

bagian("5 data teratas (hasil pemrosesan, data latih)", utama=False)
tampil_transpose(hasil_latih.head(5))
bagian("5 data terbawah (hasil pemrosesan, data latih)", utama=False)
tampil_transpose(hasil_latih.tail(5))
hasil_latih.round(3).to_csv("data_hasil_pemrosesan.csv", index_label="baris_asli")
print("\nSeluruh data latih hasil pemrosesan disimpan: data_hasil_pemrosesan.csv")

# 7b. Penelusuran perhitungan: nilai asli -> hasil pra-proses pada 10 baris di atas
#     usia: z = (x - rata-rata data latih) / simpangan baku data latih (dihitung manual, lalu dibandingkan)
bagian("Langkah 7b. Penelusuran perhitungan: nilai asli ke hasil pra-proses", utama=False)
print("Rata-rata usia data latih = %.4f | simpangan baku = %.4f" % (mu, sd))
indeks = list(hasil_latih.head(5).index) + list(hasil_latih.tail(5).index)
jejak = []
for i in indeks:
    teks = df_raw.loc[i, "Age_at_diagnosis"]
    tahun = parse_age(teks) if teks not in PLACEHOLDER else np.nan
    if np.isnan(tahun):
        tahun = nilai_isi["age_at_diagnosis"]                       # usia kosong diisi median data latih
    tahun = float(np.clip(tahun, batas_bawah, batas_atas))
    z_manual = (tahun - mu) / sd
    kolom_race = [c for c in KOLOM_RACE if hasil_latih.loc[i, c] == 1][0]
    jejak.append([i, teks, round(tahun, 3), round(z_manual, 3), round(hasil_latih.loc[i, "age_at_diagnosis"], 3),
                  bool(np.isclose(z_manual, hasil_latih.loc[i, "age_at_diagnosis"])),
                  df_raw.loc[i, "Gender"], int(hasil_latih.loc[i, "gender_female"]),
                  df_raw.loc[i, "Race"], kolom_race, df_raw.loc[i, "Grade"], int(hasil_latih.loc[i, "grade"])])
jejak = pd.DataFrame(jejak, columns=["baris_asli", "usia_asli", "usia_tahun", "z_manual", "z_program", "cocok",
                                     "gender_asli", "gender_female", "race_asli", "kolom_race_aktif",
                                     "grade_asli", "grade_kode"])
display(jejak)
assert jejak["cocok"].all(), "perhitungan manual harus sama dengan hasil program"

# 7c. Kesimpulan dari 5 data teratas dan 5 data terbawah
bagian("Langkah 7c. Kesimpulan 5 data teratas dan 5 data terbawah", utama=False)

def ringkas_baris(i):
    r = hasil_latih.loc[i]
    usia = r["age_at_diagnosis"] * sd + mu                           # kembalikan z-score ke satuan tahun
    ras = [c.replace("race_", "") for c in KOLOM_RACE if r[c] == 1][0]
    gen_mut = [g.upper() for g in GEN if r[g] == 1]
    return (f"Baris {i}: {'perempuan' if r['gender_female'] == 1 else 'laki-laki'}, usia {usia:.1f} tahun "
            f"(z = {r['age_at_diagnosis']:.3f}), ras {ras}, gen bermutasi: {', '.join(gen_mut) if gen_mut else 'tidak ada'}, "
            f"kelas {'GBM' if r['grade'] == 1 else 'LGG'}")

for i in indeks:
    print(ringkas_baris(i))

sepuluh = hasil_latih.loc[indeks]
n_gbm = int(sepuluh["grade"].sum()); n_lgg = len(sepuluh) - n_gbm
print(f"\nDari {len(sepuluh)} baris: {n_lgg} berkelas LGG dan {n_gbm} berkelas GBM")
for kode, nama in [(0, "LGG"), (1, "GBM")]:
    sub = sepuluh[sepuluh["grade"] == kode]
    if len(sub):
        print(f"Kelas {nama} ({len(sub)} baris): rata-rata z usia = {sub['age_at_diagnosis'].mean():.3f}; "
              f"IDH1 bermutasi pada {int(sub['idh1'].sum())} baris, TP53 pada {int(sub['tp53'].sum())}, "
              f"PTEN pada {int(sub['pten'].sum())}, EGFR pada {int(sub['egfr'].sum())}")


# ------------------------------------------------------------------------------
# Langkah 8. Klasifikasi KNN
# ------------------------------------------------------------------------------
bagian("Langkah 8. Klasifikasi KNN")
# Scaler usia berada DI DALAM pipeline sehingga pada setiap lipatan validasi silang ia hanya mempelajari
# rata-rata dan simpangan baku dari bagian latih lipatan itu (tidak bocor ke bagian validasi).
knn_pipe = Pipeline([
    ("skala", ColumnTransformer([("usia", StandardScaler(), kol_usia)], remainder="passthrough",
                                verbose_feature_names_out=False)),
    ("sel", SelectKBest(f_classif)),
    ("knn", KNeighborsClassifier())])
param_grid = {"sel__k": [5, 10, 15, 20, "all"],
              "knn__n_neighbors": list(range(1, 32, 2)),
              "knn__weights": ["uniform", "distance"],
              "knn__metric": ["euclidean", "manhattan"]}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
grid = GridSearchCV(knn_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train_enc, y_train)
best = grid.best_estimator_
print("\nParameter terbaik   :", grid.best_params_)
print("F1-macro CV terbaik : %.4f" % grid.best_score_)
fitur_dipilih = best.named_steps["sel"].get_feature_names_out().tolist()
print("Fitur yang dipakai model (%d): %s" % (len(fitur_dipilih), fitur_dipilih))

y_pred = best.predict(X_test_enc)
print("\nAkurasi baseline (selalu tebak LGG): %.4f" % (1 - y_test.mean()))
print("Akurasi KNN pada data uji          : %.4f" % accuracy_score(y_test, y_pred))
print("F1-macro KNN pada data uji         : %.4f" % f1_score(y_test, y_pred, average="macro"))
print()
print(classification_report(y_test, y_pred, target_names=["LGG (0)", "GBM (1)"]))
tabel_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Aktual LGG", "Aktual GBM"],
                        columns=["Prediksi LGG", "Prediksi GBM"])
display(tabel_cm)


# ------------------------------------------------------------------------------
# Eksperimen: pengaruh scaling dan seleksi fitur (10-fold CV pada data latih)
# ------------------------------------------------------------------------------
bagian("Eksperimen: pengaruh scaling dan seleksi fitur")
def cv_skor(**param):
    s = cross_val_score(clone(best).set_params(**param), X_train_enc, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    return round(s.mean(), 4), round(s.std(), 4)

tabel_eksp = pd.DataFrame([
    ["Tanpa scaling usia, semua fitur",                    *cv_skor(skala__usia="passthrough", sel__k="all")],
    ["Normalisasi min-max usia, semua fitur",              *cv_skor(skala__usia=MinMaxScaler(), sel__k="all")],
    ["Standarisasi usia (z-score), semua fitur",           *cv_skor(sel__k="all")],
    ["Standarisasi + seleksi 10 fitur terbaik",            *cv_skor(sel__k=10)],
    ["Standarisasi + seleksi 5 fitur terbaik",             *cv_skor(sel__k=5)],
    ["Standarisasi + konfigurasi terbaik grid search",     *cv_skor()],
], columns=["skenario", "akurasi_CV", "std"])
print("\nPengaruh scaling dan seleksi fitur (parameter KNN = hasil terbaik):")
display(tabel_eksp)


# ------------------------------------------------------------------------------
# Langkah 9. Pemeriksaan overfitting dan underfitting
# ------------------------------------------------------------------------------
bagian("Langkah 9. Pemeriksaan overfitting dan underfitting")
bp = grid.best_params_
baris = []
for k in range(1, 32, 2):
    p = clone(best).set_params(knn__n_neighbors=k)
    acc_latih = p.fit(X_train_enc, y_train).score(X_train_enc, y_train)
    acc_cv = cross_val_score(clone(best).set_params(knn__n_neighbors=k), X_train_enc, y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1).mean()
    baris.append([k, round(acc_latih, 4), round(acc_cv, 4), round(acc_latih - acc_cv, 4)])
tabel_k = pd.DataFrame(baris, columns=["k", "akurasi_latih", "akurasi_CV", "selisih"])
print("\nAkurasi latih vs validasi untuk berbagai k (parameter lain = hasil terbaik):")
display(tabel_k.set_index("k").loc[[1, 3, 5, 9, 15, 21, 29, 31]])

plt.figure(figsize=(7, 3.8))
plt.plot(tabel_k["k"], tabel_k["akurasi_latih"], marker="o", label="Akurasi latih", color=WARNA["LGG"])
plt.plot(tabel_k["k"], tabel_k["akurasi_CV"], marker="s", label="Akurasi CV (validasi)", color=WARNA["GBM"])
plt.xlabel("k (jumlah tetangga)"); plt.ylabel("Akurasi"); plt.legend(); plt.title("Akurasi latih vs validasi untuk berbagai k")
plt.tight_layout(); simpan_gambar("fig_k.png")

acc_latih_final = best.score(X_train_enc, y_train)
acc_uji_final = accuracy_score(y_test, y_pred)
acc_cv_final = tabel_k.loc[tabel_k.k == bp["knn__n_neighbors"], "akurasi_CV"].iloc[0]
print("Model akhir (k=%d): akurasi latih = %.4f | akurasi CV = %.4f | akurasi uji = %.4f"
      % (bp["knn__n_neighbors"], acc_latih_final, acc_cv_final, acc_uji_final))


# ------------------------------------------------------------------------------
# Kesimpulan
# ------------------------------------------------------------------------------
bagian("Kesimpulan")
print("1. Data: %d pasien, %d calon fitur (3 klinis + 20 gen); %d baris bernilai kosong tersembunyi (placeholder) diisi dengan statistik data latih."
      % (len(df_raw), len(GEN) + 3, n_bermasalah))
print("2. Kolom Project dan Primary_Diagnosis membocorkan target, Case_ID hanya pengenal -> ketiganya dibuang.")
print("3. Encoding: gender 0/1, 20 gen 0/1, ras one-hot (%s); usia distandarisasi." % ", ".join(KOLOM_RACE))
print("4. Lima data teratas dan lima data terbawah hasil pemrosesan: %d berkelas LGG dan %d berkelas GBM; hasil hitungan manual z-score usia sama dengan hasil program."
      % (n_lgg, n_gbm))
print("5. KNN terbaik: k=%d, jarak %s, bobot %s, %s fitur; akurasi uji %.4f (baseline %.4f), F1-macro %.4f."
      % (bp["knn__n_neighbors"], bp["knn__metric"], bp["knn__weights"], bp["sel__k"], acc_uji_final,
         1 - y_test.mean(), f1_score(y_test, y_pred, average="macro")))
print("6. Selisih akurasi latih dan validasi pada k terbaik hanya %.4f -> tidak ada tanda overfitting yang berarti."
      % abs(acc_latih_final - acc_cv_final))