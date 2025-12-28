# Tugas 4 - Data Mining: Association Rules

## Analisis Pola Pemesanan di Cafe "Kopi Senja"

---

## Halaman Judul

| | |
|---|---|
| **Judul Tugas** | Implementasi Algoritma Association Rules (Apriori dan FP-Growth) untuk Analisis Pola Pembelian |
| **Mata Kuliah** | Data Mining |
| **Nama Mahasiswa** | [Nama Mahasiswa] |
| **NIM** | [NIM] |
| **Dosen Pengampu** | [Nama Dosen] |
| **Tanggal** | 27 Desember 2025 |

---

## Abstrak

Laporan ini menyajikan implementasi dan analisis algoritma Association Rules, khususnya algoritma Apriori dan FP-Growth, pada studi kasus pemesanan menu di Cafe "Kopi Senja". Data transaksi dari 10 responden dianalisis untuk menemukan pola pembelian pelanggan. Metodologi yang digunakan meliputi pemrosesan data, penerapan algoritma Apriori dan FP-Growth secara manual serta implementasi menggunakan library mlxtend di Python. Hasil analisis menemukan beberapa association rules dengan nilai lift > 1, yang menunjukkan korelasi positif antar item. Rekomendasi bisnis dibuat berdasarkan pola yang ditemukan untuk meningkatkan strategi cross-selling, penempatan menu, dan manajemen stok.

---

## Daftar Isi

1. Pendahuluan
   - 1.1 Latar Belakang
   - 1.2 Rumusan Masalah
   - 1.3 Tujuan
2. Tinjauan Pustaka
   - 2.1 Association Rules
   - 2.2 Algoritma Apriori
   - 2.3 Algoritma FP-Growth
   - 2.4 Perbandingan Apriori vs FP-Growth
3. Metodologi
   - 3.1 Studi Kasus
   - 3.2 Dataset
   - 3.3 Preprocessing
   - 3.4 Algoritma
4. Hasil dan Pembahasan
   - 4.1 Exploratory Data Analysis
   - 4.2 Implementasi Apriori
   - 4.3 Implementasi FP-Growth
   - 4.4 Perbandingan Hasil
   - 4.5 Top 4 Association Rules
5. Interpretasi dan Rekomendasi
   - 5.1 Interpretasi Hasil
   - 5.2 Rekomendasi Bisnis
6. Kesimpulan
7. Saran
8. Referensi
9. Lampiran

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Analisis Association Rules adalah salah satu teknik penting dalam data mining yang digunakan untuk menemukan pola hubungan antar item dalam dataset transaksi. Teknik ini sangat berguna dalam berbagai industri, terutama retail dan Food & Beverage (F&B), untuk memahami perilaku pembelian pelanggan.

Dalam industri cafe dan restoran, memahami pola pembelian pelanggan sangat penting untuk:
- Meningkatkan strategi cross-selling
- Mengoptimalkan penempatan menu pada display
- Membuat paket promosi yang menarik
- Mengelola stok bahan baku lebih efisien

Dengan menerapkan teknik Association Rules, manajemen cafe dapat mengambil keputusan berbasis data yang lebih akurat dan efektif.

### 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana menemukan pola pembelian pelanggan Cafe "Kopi Senja" menggunakan algoritma Association Rules?
2. Item apa yang sering dibeli bersama oleh pelanggan?
3. Algoritma mana (Apriori atau FP-Growth) yang lebih efisien untuk dataset ini?
4. Rekomendasi bisnis apa yang dapat diberikan berdasarkan pola pembelian yang ditemukan?

### 1.3 Tujuan

Tujuan dari penelitian ini adalah:
1. Menerapkan algoritma Apriori dan FP-Growth untuk menganalisis pola pembelian pelanggan
2. Menemukan association rules yang berguna untuk meningkatkan strategi bisnis cafe
3. Membandingkan hasil dan efisiensi kedua algoritma
4. Memberikan rekomendasi bisnis yang praktis berdasarkan hasil analisis

---

## 2. Tinjauan Pustaka

### 2.1 Association Rules

Association Rules adalah teknik data mining yang digunakan untuk menemukan hubungan antar item dalam dataset transaksi. Setiap rule dinyatakan dalam bentuk:

```
A → B
```

Dimana A disebut antecedent (kondisi) dan B disebut consequent (hasil). Tiga metrik utama yang digunakan untuk mengevaluasi kualitas association rules:

#### Support
Support mengukur seberapa sering rule muncul dalam dataset:
```
Support(A → B) = (Jumlah transaksi mengandung A dan B) / (Total transaksi)
```

#### Confidence
Confidence mengukur seberapa sering rule benar:
```
Confidence(A → B) = (Jumlah transaksi mengandung A dan B) / (Jumlah transaksi mengandung A)
```

#### Lift
Lift mengukur kekuatan korelasi antar item:
```
Lift(A → B) = Confidence(A → B) / Support(B)
```

**Interpretasi Lift:**
- Lift > 1: Item A dan B muncul bersama lebih sering dari yang diharapkan (positive correlation)
- Lift = 1: Item A dan B independen (no correlation)
- Lift < 1: Item A dan B muncul bersama lebih jarang dari yang diharapkan (negative correlation)

### 2.2 Algoritma Apriori

Algoritma Apriori adalah salah satu algoritma paling populer untuk menemukan frequent itemsets. Algoritma ini bekerja dengan prinsip "apriori property": jika sebuah itemset frequent, maka semua subset-nya juga harus frequent.

#### Langkah-langkah Apriori:

1. **Menentukan Minimum Support**: Menetapkan threshold support minimum untuk memfilter itemsets

2. **Membuat Frequent 1-Itemset (C1 → L1)**:
   - Hitung frekuensi kemunculan setiap item
   - Filter item yang memenuhi minimum support

3. **Membuat Frequent 2-Itemset (C2 → L2)**:
   - Buat kombinasi pasangan item dari L1
   - Hitung support untuk setiap pasangan
   - Filter pasangan yang memenuhi minimum support

4. **Membuat Frequent 3-Itemset (C3 → L3)**:
   - Buat kombinasi triplet item dari L2
   - Hitung support untuk setiap triplet
   - Filter triplet yang memenuhi minimum support

5. **Generate Association Rules**:
   - Dari frequent itemset, buat aturan asosiasi
   - Hitung confidence untuk setiap aturan
   - Filter aturan yang memenuhi minimum confidence
   - Hitung lift untuk menilai kekuatan asosiasi

#### Kelebihan Apriori:
- Sederhana dan mudah dipahami
- Cocok untuk dataset dengan jumlah item yang relatif sedikit

#### Kekurangan Apriori:
- Memerlukan banyak scan database
- Menghasilkan banyak candidate itemsets yang tidak perlu
- Tidak efisien untuk dataset besar

### 2.3 Algoritma FP-Growth

Algoritma FP-Growth (Frequent Pattern Growth) dikembangkan untuk mengatasi keterbatasan Apriori. Algoritma ini tidak menggunakan candidate generation, tetapi menggunakan struktur data FP-Tree untuk mengekstrak frequent patterns.

#### Langkah-langkah FP-Growth:

1. **Scan Database & Buat Frequent 1-Itemset**:
   - Hitung frekuensi setiap item
   - Urutkan item berdasarkan frekuensi (descending)
   - Filter item yang memenuhi minimum support

2. **Sort & Reorder Transactions**:
   - Urutkan item dalam setiap transaksi berdasarkan frekuensi
   - Buat ordered itemset untuk setiap transaksi

3. **Bangun FP-Tree**:
   - Buat root node (null)
   - Insert setiap transaksi yang sudah diurutkan ke FP-Tree
   - Increment count untuk node yang sudah ada
   - Buat node baru untuk item yang belum ada

4. **Mining Frequent Patterns**:
   - Untuk setiap item dalam frequent 1-itemset (dari bawah ke atas):
     - Cari conditional pattern base
     - Bangun conditional FP-Tree
     - Generate frequent patterns dari conditional FP-Tree

5. **Generate Association Rules**:
   - Dari frequent patterns yang ditemukan
   - Hitung confidence dan lift
   - Pilih rules yang relevan

#### Kelebihan FP-Growth:
- Hanya memerlukan 2 kali scan database
- Tidak menggunakan candidate generation
- Lebih efisien untuk dataset besar

#### Kekurangan FP-Growth:
- Lebih kompleks untuk diimplementasikan
- Memerlukan memori lebih besar untuk menyimpan FP-Tree

### 2.4 Perbandingan Apriori vs FP-Growth

| Aspek | Apriori | FP-Growth |
|-------|---------|-----------|
| Scan Database | Banyak | 2 kali |
| Candidate Generation | Ya | Tidak |
| Penggunaan Memori | Lebih sedikit | Lebih banyak |
| Efisiensi untuk Dataset Besar | Rendah | Tinggi |
| Kompleksitas Implementasi | Rendah | Tinggi |

---

## 3. Metodologi

### 3.1 Studi Kasus

Cafe "Kopi Senja" adalah kedai kopi lokal yang menyajikan berbagai menu minuman dan makanan. Responden dalam penelitian ini adalah 10 orang yang melakukan transaksi pembelian di cafe tersebut. Tujuannya adalah untuk memahami pola pemesanan pelanggan guna meningkatkan strategi bisnis cafe.

### 3.2 Dataset

#### Responden (10 Orang):
1. Dinunaya Syuja Aryoko
2. Ahmad Arya Dwi Febriansyah
3. Vito Agatha Satritama
4. Moh. Dani Wahyudi
5. Achmad Wildan Muzaky
6. Frenky
7. Andries Nauvalentin Roestam
8. Muhammad Alvin Firdaus
9. Husni Mubarok
10. Abiyyu Valin Zavero

#### Menu Cafe (Item Set):

**Minuman:**
- Kopi Hitam
- Kopi Susu
- Cappuccino
- Latte
- Teh Manis
- Jus Jeruk
- Air Mineral

**Makanan:**
- Roti Bakar
- Croissant
- Donut
- Kue Coklat
- Pisang Goreng
- Nasi Goreng
- Mie Goreng

**Snack:**
- Keripik Kentang
- Biskuit

#### Data Transaksi:

| Responden | Item yang Dipesan |
|-----------|-------------------|
| Dinunaya Syuja Aryoko | Kopi Susu, Roti Bakar, Keripik Kentang |
| Ahmad Arya Dwi Febriansyah | Cappuccino, Croissant, Air Mineral |
| Vito Agatha Satritama | Kopi Hitam, Roti Bakar, Keripik Kentang, Biskuit |
| Moh. Dani Wahyudi | Kopi Susu, Roti Bakar, Jus Jeruk |
| Achmad Wildan Muzaky | Teh Manis, Donut, Air Mineral |
| Frenky | Cappuccino, Croissant, Keripik Kentang |
| Andries Nauvalentin Roestam | Kopi Susu, Roti Bakar, Donut |
| Muhammad Alvin Firdaus | Latte, Nasi Goreng, Air Mineral |
| Husni Mubarok | Kopi Hitam, Roti Bakar, Keripik Kentang |
| Abiyyu Valin Zavero | Teh Manis, Donut, Kue Coklat |

### 3.3 Preprocessing

Dataset transaksi diubah menjadi format binary encoded matrix menggunakan TransactionEncoder dari library mlxtend. Setiap baris merepresentasikan satu transaksi dan setiap kolom merepresentasikan satu item. Nilai 1 menunjukkan item tersebut ada dalam transaksi, sedangkan 0 menunjukkan tidak ada.

### 3.4 Algoritma

#### Parameter yang Digunakan:
- **Minimum Support**: 30% (0.3)
- **Minimum Support Count**: 10 × 0.3 = 3 transaksi
- **Minimum Confidence**: 70% (0.7)

#### Implementasi:
1. **Apriori Manual**: Implementasi langkah demi langkah tanpa menggunakan library
2. **Apriori Python**: Menggunakan library mlxtend
3. **FP-Growth Manual**: Implementasi langkah demi langkah tanpa menggunakan library
4. **FP-Growth Python**: Menggunakan library mlxtend

---

## 4. Hasil dan Pembahasan

### 4.1 Exploratory Data Analysis

#### Code untuk Exploratory Data Analysis:

```python
# Menghitung frekuensi kemunculan item
item_counts = df.sum().sort_values(ascending=False)

# Menampilkan frekuensi item
print("Frekuensi Kemunculan Item:")
for item, count in item_counts.items():
    print(f"{item}: {count}")

# Menampilkan statistik deskriptif
print("\nStatistik Deskriptif:")
print(f"Total Item Terjual: {df.sum().sum()}")
print(f"Rata-rata Item per Transaksi: {df.sum().mean():.2f}")
print(f"Minimum Item per Transaksi: {df.sum(axis=1).min()}")
print(f"Maximum Item per Transaksi: {df.sum(axis=1).max()}")
```

**Penjelasan Code:**
- `df.sum()`: Menghitung jumlah True (1) untuk setiap kolom item
- `sort_values(ascending=False)`: Mengurutkan hasil dari yang tertinggi ke terendah
- `df.sum().sum()`: Menghitung total semua item yang terjual
- `df.sum().mean()`: Menghitung rata-rata item per transaksi
- `df.sum(axis=1).min()` dan `.max()`: Menghitung minimum dan maximum item per transaksi

#### Hasil EDA:

| Item | Frekuensi | Support (%) |
|------|-----------|-------------|
| Roti Bakar | 5 | 50% |
| Keripik Kentang | 4 | 40% |
| Air Mineral | 3 | 30% |
| Kopi Susu | 3 | 30% |
| Donut | 3 | 30% |
| Croissant | 2 | 20% |
| Teh Manis | 2 | 20% |
| Kopi Hitam | 2 | 20% |
| Cappuccino | 2 | 20% |
| Biskuit | 1 | 10% |
| Jus Jeruk | 1 | 10% |
| Kue Coklat | 1 | 10% |
| Nasi Goreng | 1 | 10% |
| Latte | 1 | 10% |

**Statistik Deskriptif:**
- Total Item Terjual: 31
- Rata-rata Item per Transaksi: 2.21
- Minimum Item per Transaksi: 3
- Maximum Item per Transaksi: 4

**Interpretasi:**
Item yang paling sering dibeli adalah Roti Bakar (50% dari semua transaksi), diikuti oleh Keripik Kentang (40%). Kopi Susu, Air Mineral, dan Donut masing-masing muncul dalam 30% transaksi. Hal ini menunjukkan bahwa Roti Bakar adalah menu yang paling populer di Cafe "Kopi Senja".

### 4.2 Implementasi Apriori

#### 4.2.1 Apriori - Perhitungan Manual

**Code untuk Perhitungan Manual Apriori:**

```python
def manual_apriori(transactions, min_support=0.3):
    n = len(transactions)
    min_support_count = int(n * min_support)

    print(f"Total Transaksi: {n}")
    print(f"Minimum Support: {min_support}")
    print(f"Minimum Support Count: {min_support_count}\n")

    # C1: Candidate 1-itemset
    c1 = {}
    for trans in transactions:
        for item in trans:
            if item not in c1:
                c1[item] = 1
            else:
                c1[item] += 1

    print("=== C1 (Candidate 1-itemset) ===")
    df_c1 = pd.DataFrame(list(c1.items()), columns=['Item', 'Count'])
    df_c1['Support'] = df_c1['Count'] / n
    print(df_c1.to_string(index=False))

    # L1: Frequent 1-itemset
    l1 = {k: v for k, v in c1.items() if v >= min_support_count}

    print(f"\n=== L1 (Frequent 1-itemset) - Support >= {min_support} ===")
    df_l1 = pd.DataFrame(list(l1.items()), columns=['Item', 'Count'])
    df_l1['Support'] = df_l1['Count'] / n
    print(df_l1.to_string(index=False))

    # C2: Candidate 2-itemset
    l1_items = list(l1.keys())
    c2 = {}

    for i in range(len(l1_items)):
        for j in range(i+1, len(l1_items)):
            itemset = frozenset([l1_items[i], l1_items[j]])
            count = 0
            for trans in transactions:
                if itemset.issubset(set(trans)):
                    count += 1
            c2[itemset] = count

    print(f"\n=== C2 (Candidate 2-itemset) ===")
    df_c2 = pd.DataFrame([(tuple(k), v) for k, v in c2.items()], columns=['Itemset', 'Count'])
    df_c2['Support'] = df_c2['Count'] / n
    print(df_c2.to_string(index=False))

    # L2: Frequent 2-itemset
    l2 = {k: v for k, v in c2.items() if v >= min_support_count}

    print(f"\n=== L2 (Frequent 2-itemset) - Support >= {min_support} ===")
    if l2:
        df_l2 = pd.DataFrame([(tuple(k), v) for k, v in l2.items()], columns=['Itemset', 'Count'])
        df_l2['Support'] = df_l2['Count'] / n
        print(df_l2.to_string(index=False))
    else:
        print("Tidak ada frequent 2-itemset yang ditemukan.")

    return c1, l1, c2, l2

c1, l1, c2, l2 = manual_apriori(transactions, min_support=0.3)
```

**Penjelasan Code:**
- `min_support_count = int(n * min_support)`: Menghitung minimum jumlah transaksi yang diperlukan (10 × 0.3 = 3)
- **C1 Generation**: Loop melalui setiap transaksi dan item untuk menghitung frekuensi kemunculan
- **L1 Filtering**: Hanya mempertahankan item dengan count >= 3
- **C2 Generation**: Membuat semua kombinasi pasangan dari L1 menggunakan `frozenset` untuk menghindari duplikasi
- **L2 Filtering**: Hanya mempertahankan pasangan dengan count >= 3

**Hasil Perhitungan Manual:**

**C1 (Candidate 1-itemset):**

| Item | Count | Support |
|------|-------|---------|
| Kopi Susu | 3 | 0.3 |
| Roti Bakar | 5 | 0.5 |
| Keripik Kentang | 4 | 0.4 |
| Cappuccino | 2 | 0.2 |
| Croissant | 2 | 0.2 |
| Air Mineral | 3 | 0.3 |
| Kopi Hitam | 2 | 0.2 |
| Biskuit | 1 | 0.1 |
| Jus Jeruk | 1 | 0.1 |
| Teh Manis | 2 | 0.2 |
| Donut | 3 | 0.3 |
| Latte | 1 | 0.1 |
| Nasi Goreng | 1 | 0.1 |
| Kue Coklat | 1 | 0.1 |

**L1 (Frequent 1-itemset) - Support >= 0.3:**

| Item | Count | Support |
|------|-------|---------|
| Kopi Susu | 3 | 0.3 |
| Roti Bakar | 5 | 0.5 |
| Keripik Kentang | 4 | 0.4 |
| Air Mineral | 3 | 0.3 |
| Donut | 3 | 0.3 |

**C2 (Candidate 2-itemset):**

| Itemset | Count | Support |
|---------|-------|---------|
| (Roti Bakar, Kopi Susu) | 3 | 0.3 |
| (Keripik Kentang, Kopi Susu) | 1 | 0.1 |
| (Air Mineral, Kopi Susu) | 0 | 0.0 |
| (Donut, Kopi Susu) | 1 | 0.1 |
| (Roti Bakar, Keripik Kentang) | 3 | 0.3 |
| (Air Mineral, Roti Bakar) | 0 | 0.0 |
| (Donut, Roti Bakar) | 1 | 0.1 |
| (Air Mineral, Keripik Kentang) | 0 | 0.0 |
| (Donut, Keripik Kentang) | 0 | 0.0 |
| (Donut, Air Mineral) | 1 | 0.1 |

**L2 (Frequent 2-itemset) - Support >= 0.3:**

| Itemset | Count | Support |
|---------|-------|---------|
| (Roti Bakar, Kopi Susu) | 3 | 0.3 |
| (Roti Bakar, Keripik Kentang) | 3 | 0.3 |

#### 4.2.2 Apriori - Implementasi Python (mlxtend)

**Code:**

```python
from mlxtend.frequent_patterns import apriori

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("=== Frequent Itemsets (Apriori) ===")
print(frequent_itemsets)

# Add length column
print("\n=== Frequent Itemsets by Length ===")
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets.sort_values(['length', 'support'], ascending=[True, False]))

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("\n=== Association Rules (Apriori) ===")
print("Metric: Confidence >= 70%")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

# Sort rules by lift and confidence
rules_sorted = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
print("\n=== Top 5 Rules berdasarkan Lift dan Confidence ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head().to_string(index=False))
```

**Penjelasan Code:**
- `apriori(df, min_support=0.3, use_colnames=True)`: Mencari frequent itemsets dengan minimum support 0.3
- `frequent_itemsets['itemsets'].apply(lambda x: len(x))`: Menambah kolom panjang itemset
- `association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)`: Membuat rules dengan minimum confidence 0.7
- `sort_values(['lift', 'confidence'], ascending=[False, False])`: Mengurutkan rules berdasarkan lift tertinggi, kemudian confidence

**Hasil Implementasi Python:**

**Frequent Itemsets (Apriori):**

| Support | Itemsets |
|---------|-----------|
| 0.3 | (Air Mineral) |
| 0.3 | (Donut) |
| 0.4 | (Keripik Kentang) |
| 0.3 | (Kopi Susu) |
| 0.5 | (Roti Bakar) |
| 0.3 | (Roti Bakar, Keripik Kentang) |
| 0.3 | (Roti Bakar, Kopi Susu) |

**Frequent Itemsets by Length:**

| Support | Itemsets | Length |
|---------|-----------|--------|
| 0.5 | (Roti Bakar) | 1 |
| 0.4 | (Keripik Kentang) | 1 |
| 0.3 | (Air Mineral) | 1 |
| 0.3 | (Donut) | 1 |
| 0.3 | (Kopi Susu) | 1 |
| 0.3 | (Roti Bakar, Keripik Kentang) | 2 |
| 0.3 | (Roti Bakar, Kopi Susu) | 2 |

**Association Rules (Apriori) - Metric: Confidence >= 70%:**

| Antecedents | Consequents | Support | Confidence | Lift |
|-------------|-------------|---------|------------|------|
| (Keripik Kentang) | (Roti Bakar) | 0.3 | 0.75 | 1.5 |
| (Kopi Susu) | (Roti Bakar) | 0.3 | 1.00 | 2.0 |

**Top 5 Rules berdasarkan Lift dan Confidence:**

| Antecedents | Consequents | Support | Confidence | Lift |
|-------------|-------------|---------|------------|------|
| (Kopi Susu) | (Roti Bakar) | 0.3 | 1.00 | 2.0 |
| (Keripik Kentang) | (Roti Bakar) | 0.3 | 0.75 | 1.5 |

### 4.3 Implementasi FP-Growth

#### 4.3.1 FP-Growth - Perhitungan Manual

**Code untuk Perhitungan Manual FP-Growth:**

```python
def manual_fpgrowth(transactions, min_support=0.3):
    n = len(transactions)
    min_support_count = int(n * min_support)

    print(f"=== FP-Growth Manual Calculation ===")
    print(f"Total Transaksi: {n}")
    print(f"Minimum Support: {min_support}")
    print(f"Minimum Support Count: {min_support_count}\n")

    # Step 1: Frequent 1-itemset (sorted by frequency)
    item_freq = {}
    for trans in transactions:
        for item in trans:
            item_freq[item] = item_freq.get(item, 0) + 1

    frequent_items = {k: v for k, v in item_freq.items() if v >= min_support_count}
    sorted_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)

    print("=== Frequent 1-itemset (Sorted) ===")
    for item, freq in sorted_items:
        print(f"{item}: {freq} ({freq/n:.2f})")

    item_rank = {item: rank for rank, (item, _) in enumerate(sorted_items)}

    # Step 2: Sort transactions
    ordered_transactions = []
    for trans in transactions:
        ordered = [item for item in trans if item in item_rank]
        ordered.sort(key=lambda x: item_rank[x])
        ordered_transactions.append(ordered)

    print("\n=== Ordered Transactions ===")
    for i, trans in enumerate(ordered_transactions, 1):
        print(f"T{i}: {trans}")

    # Step 3: Build FP-Tree (simplified representation)
    print("\n=== FP-Tree Structure ===")

    fp_tree = {}
    for trans in ordered_transactions:
        current_level = fp_tree
        for item in trans:
            if item not in current_level:
                current_level[item] = {'count': 1, 'children': {}}
            else:
                current_level[item]['count'] += 1
            current_level = current_level[item]['children']

    def print_tree(node, prefix=""):
        for item, data in node.items():
            print(f"{prefix}{item} (count: {data['count']})")
            if data['children']:
                print_tree(data['children'], prefix + "  ")

    print_tree(fp_tree)

    return frequent_items, ordered_transactions, fp_tree

frequent_items_fp, ordered_trans, fp_tree = manual_fpgrowth(transactions, min_support=0.3)
```

**Penjelasan Code:**
- **Step 1**: Menghitung frekuensi semua item dan mengurutkannya berdasarkan frekuensi (descending)
- **item_rank**: Memberikan ranking berdasarkan urutan frekuensi untuk sorting
- **Step 2**: Mengurutkan setiap transaksi berdasarkan ranking item
- **Step 3**: Membangun FP-Tree dengan struktur dictionary bersarang
  - Jika node sudah ada, increment count
  - Jika belum, buat node baru dengan count = 1

**Hasil Perhitungan Manual:**

**Frequent 1-itemset (Sorted by Frequency):**

| Rank | Item | Count | Support |
|------|------|-------|---------|
| 1 | Roti Bakar | 5 | 0.50 |
| 2 | Keripik Kentang | 4 | 0.40 |
| 3 | Kopi Susu | 3 | 0.30 |
| 4 | Air Mineral | 3 | 0.30 |
| 5 | Donut | 3 | 0.30 |

**Ordered Transactions (berdasarkan frekuensi):**

| T | Ordered Items |
|---|---------------|
| T1 | ['Roti Bakar', 'Keripik Kentang', 'Kopi Susu'] |
| T2 | ['Air Mineral'] |
| T3 | ['Roti Bakar', 'Keripik Kentang'] |
| T4 | ['Roti Bakar', 'Kopi Susu'] |
| T5 | ['Air Mineral', 'Donut'] |
| T6 | ['Keripik Kentang'] |
| T7 | ['Roti Bakar', 'Kopi Susu', 'Donut'] |
| T8 | ['Air Mineral'] |
| T9 | ['Roti Bakar', 'Keripik Kentang'] |
| T10 | ['Donut'] |

**FP-Tree Structure:**
```
Roti Bakar (count: 5)
  Keripik Kentang (count: 3)
    Kopi Susu (count: 1)
  Kopi Susu (count: 2)
    Donut (count: 1)
Air Mineral (count: 3)
  Donut (count: 1)
Keripik Kentang (count: 1)
Donut (count: 1)
```

#### 4.3.2 FP-Growth - Implementasi Python (mlxtend)

**Code:**

```python
from mlxtend.frequent_patterns import fpgrowth

# Generate frequent itemsets
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.3, use_colnames=True)

print("=== Frequent Itemsets (FP-Growth) ===")
print(frequent_itemsets_fpgrowth)

# Add length column
print("\n=== Frequent Itemsets by Length ===")
frequent_itemsets_fpgrowth['length'] = frequent_itemsets_fpgrowth['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets_fpgrowth.sort_values(['length', 'support'], ascending=[True, False]))

# Generate association rules
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.7)

print("\n=== Association Rules (FP-Growth) ===")
print("Metric: Confidence >= 70%")
print(rules_fpgrowth[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

# Sort rules by lift and confidence
rules_fpgrowth_sorted = rules_fpgrowth.sort_values(['lift', 'confidence'], ascending=[False, False])
print("\n=== Top 5 Rules berdasarkan Lift dan Confidence ===")
print(rules_fpgrowth_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head().to_string(index=False))
```

**Penjelasan Code:**
- `fpgrowth(df, min_support=0.3, use_colnames=True)`: Menggunakan algoritma FP-Growth untuk mencari frequent itemsets
- Parameter dan output yang dihasilkan sama seperti Apriori

**Hasil Implementasi Python:**

**Frequent Itemsets (FP-Growth):**

| Support | Itemsets |
|---------|-----------|
| 0.5 | (Roti Bakar) |
| 0.4 | (Keripik Kentang) |
| 0.3 | (Kopi Susu) |
| 0.3 | (Air Mineral) |
| 0.3 | (Donut) |
| 0.3 | (Roti Bakar, Keripik Kentang) |
| 0.3 | (Roti Bakar, Kopi Susu) |

**Frequent Itemsets by Length:**

| Support | Itemsets | Length |
|---------|-----------|--------|
| 0.5 | (Roti Bakar) | 1 |
| 0.4 | (Keripik Kentang) | 1 |
| 0.3 | (Kopi Susu) | 1 |
| 0.3 | (Air Mineral) | 1 |
| 0.3 | (Donut) | 1 |
| 0.3 | (Roti Bakar, Keripik Kentang) | 2 |
| 0.3 | (Roti Bakar, Kopi Susu) | 2 |

**Association Rules (FP-Growth) - Metric: Confidence >= 70%:**

| Antecedents | Consequents | Support | Confidence | Lift |
|-------------|-------------|---------|------------|------|
| (Keripik Kentang) | (Roti Bakar) | 0.3 | 0.75 | 1.5 |
| (Kopi Susu) | (Roti Bakar) | 0.3 | 1.00 | 2.0 |

**Top 5 Rules berdasarkan Lift dan Confidence:**

| Antecedents | Consequents | Support | Confidence | Lift |
|-------------|-------------|---------|------------|------|
| (Kopi Susu) | (Roti Bakar) | 0.3 | 1.00 | 2.0 |
| (Keripik Kentang) | (Roti Bakar) | 0.3 | 0.75 | 1.5 |

### 4.4 Perbandingan Hasil

**Code untuk Perbandingan:**

```python
# Perbandingan Frequent Itemsets
print("=== Perbandingan Frequent Itemsets: Apriori vs FP-Growth ===")
comparison = pd.DataFrame({
    'Apriori': frequent_itemsets.sort_values('support', ascending=False).reset_index(drop=True)['support'],
    'FP-Growth': frequent_itemsets_fpgrowth.sort_values('support', ascending=False).reset_index(drop=True)['support']
})
print(comparison)
print(f"\nApakah hasil sama? {frequent_itemsets.equals(frequent_itemsets_fpgrowth)}")

# Perbandingan Rules
print("\n=== Perbandingan Rules: Apriori vs FP-Growth ===")
rules_comparison = pd.DataFrame({
    'Apriori Count': len(rules),
    'FP-Growth Count': len(rules_fpgrowth)
}, index=['Jumlah Rules'])
print(rules_comparison)
```

**Penjelasan Code:**
- `frequent_itemsets.equals(frequent_itemsets_fpgrowth)`: Memeriksa apakah dua DataFrame identik (isi, urutan, dan tipe data sama)
- Perbandingan dibuat dengan mengurutkan kedua hasil berdasarkan support untuk memudahkan perbandingan

#### Frequent Itemsets:

| Apriori | FP-Growth |
|---------|-----------|
| 0.5 | 0.5 |
| 0.4 | 0.4 |
| 0.3 | 0.3 |
| 0.3 | 0.3 |
| 0.3 | 0.3 |
| 0.3 | 0.3 |
| 0.3 | 0.3 |

**Apakah hasil sama? False**

**Catatan:** Meskipun nilai support-nya identik, hasil perbandingan `False` karena urutan itemsets berbeda (Apriori dan FP-Growth mungkin menghasilkan urutan yang berbeda). Namun, dari perspektif konten, keduanya menghasilkan 7 frequent itemsets yang sama.

#### Association Rules:

| Apriori Count | FP-Growth Count |
|---------------|------------------|
| 2 | 2 |

**Kesimpulan:**

1. **Frequent Itemsets**: Kedua algoritma menghasilkan 7 frequent itemsets yang sama dari sisi konten dan nilai support.

2. **Association Rules**: Kedua algoritma menghasilkan 2 association rules yang identik, dengan nilai support, confidence, dan lift yang sama.

3. **Efisiensi**: FP-Growth lebih efisien dalam hal waktu eksekusi karena:
   - Hanya memerlukan 2 scan database
   - Tidak melakukan candidate generation yang tidak perlu
   - Menggunakan struktur FP-Tree yang lebih kompak

4. **Akurasi**: Hasil dari kedua algoritma identik, yang mengonfirmasi keakuratan implementasi masing-masing.

### 4.5 Top 4 Association Rules

**Code untuk Mengambil Top 4 Rules:**

```python
# Mengambil 4 rules terbaik berdasarkan lift dan confidence
top4_rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False]).head(4)

print("=== TOP 4 ASSOCIATION RULES ===")
for idx, row in top4_rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    print(f"\nRule {idx+1}:")
    print(f"  {antecedents} -> {consequents}")
    print(f"  Support: {row['support']:.4f} ({row['support']*100:.2f}%)")
    print(f"  Confidence: {row['confidence']:.4f} ({row['confidence']*100:.2f}%)")
    print(f"  Lift: {row['lift']:.4f}")

    if row['lift'] > 1:
        print(f"  Interpretasi: Positive correlation (lift > 1)")
    elif row['lift'] == 1:
        print(f"  Interpretasi: No correlation (lift = 1)")
    else:
        print(f"  Interpretasi: Negative correlation (lift < 1)")
```

**Penjelasan Code:**
- `sort_values(['lift', 'confidence'], ascending=[False, False])`: Mengurutkan rules berdasarkan lift (descending), kemudian confidence (descending)
- `head(4)`: Mengambil 4 rules teratas
- `list(row['antecedents'])`: Mengubah frozenset menjadi list untuk ditampilkan
- `row['support']*100`: Mengonversi nilai support desimal ke persentase
- Kondisi `if row['lift'] > 1`: Memberikan interpretasi berdasarkan nilai lift

**Hasil Top 4 Association Rules:**

#### Rule 2: Kopi Susu → Roti Bakar

| Metric | Nilai |
|--------|-------|
| Support | 0.3000 (30.00%) |
| Confidence | 1.0000 (100.00%) |
| Lift | 2.0000 |

**Interpretasi:**
- **Support 30%**: 30% dari semua transaksi mengandung Kopi Susu dan Roti Bakar bersamaan (3 dari 10 transaksi)
- **Confidence 100%**: 100% pelanggan yang membeli Kopi Susu juga membeli Roti Bakar (3 dari 3 transaksi yang mengandung Kopi Susu)
- **Lift 2.0**: Nilai lift > 1 menunjukkan **positive correlation**. Pelanggan 2 kali lebih mungkin membeli Roti Bakar jika mereka membeli Kopi Susu dibandingkan secara acak. Ini adalah korelasi positif yang sangat kuat.

**Implikasi Bisnis**: Kopi Susu dan Roti Bakar adalah kombinasi yang sangat kuat. Setiap pelanggan yang membeli Kopi Susu hampir pasti akan membeli Roti Bakar.

#### Rule 1: Keripik Kentang → Roti Bakar

| Metric | Nilai |
|--------|-------|
| Support | 0.3000 (30.00%) |
| Confidence | 0.7500 (75.00%) |
| Lift | 1.5000 |

**Interpretasi:**
- **Support 30%**: 30% dari semua transaksi mengandung Keripik Kentang dan Roti Bakar bersamaan (3 dari 10 transaksi)
- **Confidence 75%**: 75% pelanggan yang membeli Keripik Kentang juga membeli Roti Bakar (3 dari 4 transaksi yang mengandung Keripik Kentang)
- **Lift 1.5**: Nilai lift > 1 menunjukkan **positive correlation**. Pelanggan 1.5 kali lebih mungkin membeli Roti Bakar jika mereka membeli Keripik Kentang dibandingkan secara acak.

**Implikasi Bisnis**: Keripik Kentang dan Roti Bakar juga memiliki korelasi positif yang baik. Sebagian besar pelanggan yang membeli Keripik Kentang akan membeli Roti Bakar.

**Catatan**: Karena dataset ini hanya menghasilkan 2 rules yang memenuhi kriteria minimum confidence 70%, maka hanya 2 rules yang ditampilkan. Jika dataset lebih besar atau threshold lebih rendah, akan ada lebih banyak rules yang bisa ditampilkan.

### 4.6 Visualisasi Association Rules

**Code untuk Scatter Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules['support'], rules['confidence'],
                      c=rules['lift'], s=100, alpha=0.6, cmap='viridis')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Scatter Plot: Support vs Confidence (colored by Lift)')
plt.colorbar(scatter, label='Lift')
plt.axhline(y=0.7, color='r', linestyle='--', label='Min Confidence = 0.7')
plt.legend()
plt.show()
```

**Penjelasan Code:**
- `plt.scatter(rules['support'], rules['confidence'])`: Membuat scatter plot dengan support di sumbu x dan confidence di sumbu y
- `c=rules['lift']`: Menggunakan nilai lift sebagai warna (color mapping)
- `s=100`: Ukuran titik scatter
- `alpha=0.6`: Transparansi titik
- `cmap='viridis'`: Skema warna untuk lift
- `plt.colorbar()`: Menambahkan legend warna untuk nilai lift
- `plt.axhline(y=0.7)`: Menambahkan garis horizontal di y=0.7 untuk menunjukkan minimum confidence

**Interpretasi Visualisasi:**
Scatter plot menunjukkan hubungan antara support dan confidence dari association rules yang ditemukan. Setiap titik merepresentasikan satu rule:
- Sumbu X: Support (seberapa sering rule muncul)
- Sumbu Y: Confidence (seberapa akurat rule tersebut)
- Warna Titik: Nilai Lift (warna lebih cerah = lift lebih tinggi)
- Garis Merah: Batas minimum confidence (70%)

**Code untuk Bar Chart Top Rules:**

```python
# Siapkan data untuk plotting
top4_rules_plot = top4_rules.copy()
top4_rules_plot['rule'] = [f"{list(r['antecedents'])} -> {list(r['consequents'])}"
                            for _, r in top4_rules_plot.iterrows()]

plt.figure(figsize=(12, 6))
x_pos = np.arange(len(top4_rules_plot))
plt.bar(x_pos - 0.2, top4_rules_plot['support'], 0.2, label='Support')
plt.bar(x_pos, top4_rules_plot['confidence'], 0.2, label='Confidence')
plt.xlabel('Rules')
plt.ylabel('Value')
plt.title('Top 4 Rules: Support, Confidence, and Lift')
plt.xticks(x_pos, top4_rules_plot['rule'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
```

**Penjelasan Code:**
- `top4_rules_plot['rule']`: Membuat kolom baru dengan format rule string
- `np.arange(len(top4_rules_plot))`: Membuat posisi x untuk setiap bar
- `x_pos - 0.2`: Offset untuk bar support (sedikit ke kiri)
- `x_pos`: Posisi untuk bar confidence (di tengah)
- `0.2`: Lebar bar
- `rotation=45, ha='right'`: Rotasi label x-axis 45 derajat dengan alignment ke kanan
- `tight_layout()`: Mengatur layout agar tidak ada elemen yang terpotong

**Interpretasi Visualisasi:**
Bar chart membandingkan nilai support dan confidence untuk setiap rule:
- Bar Biru (Support): Seberapa sering rule muncul dalam dataset
- Bar Oranye (Confidence): Seberapa akurat rule tersebut
- Dari visualisasi, terlihat bahwa kedua rules memiliki support yang sama (0.3), namun rule "Kopi Susu → Roti Bakar" memiliki confidence yang lebih tinggi (1.0 vs 0.75)

---

## 5. Interpretasi dan Rekomendasi

### 5.1 Interpretasi Hasil

Berdasarkan hasil analisis menggunakan algoritma Apriori dan FP-Growth, dapat ditarik beberapa kesimpulan:

1. **Pola Pembelian Utama**: Roti Bakar adalah item yang paling populer, muncul dalam 50% transaksi (5 dari 10). Hal ini menunjukkan bahwa Roti Bakar adalah menu favorit pelanggan Cafe "Kopi Senja".

2. **Kombinasi yang Sering Terjadi**:
   - Roti Bakar + Keripik Kentang muncul dalam 30% transaksi
   - Roti Bakar + Kopi Susu juga muncul dalam 30% transaksi
   - Ini menunjukkan bahwa Roti Bakar sering dibeli sebagai pendamping minuman atau snack

3. **Korelasi Positif**: Kedua association rules yang ditemukan memiliki nilai lift > 1, yang menunjukkan bahwa item-item tersebut memiliki korelasi positif yang kuat.

4. **Confidence Sangat Tinggi**:
   - Rule {Kopi Susu} → {Roti Bakar} memiliki confidence 100%, yang berarti setiap pelanggan yang membeli Kopi Susu pasti juga membeli Roti Bakar
   - Rule {Keripik Kentang} → {Roti Bakar} memiliki confidence 75%, yang menunjukkan korelasi yang kuat

5. **Konsistensi Algoritma**: Apriori dan FP-Growth menghasilkan frequent itemsets dan association rules yang identik, mengonfirmasi keakuratan implementasi masing-masing.

### 5.2 Rekomendasi Bisnis

#### 1. Strategi Cross-Selling

Berdasarkan rules yang ditemukan, berikut adalah strategi cross-selling yang direkomendasikan:

| Ketika pelanggan membeli | Tawarkan tambahan | Alasan |
|-------------------------|-------------------|---------|
| Kopi Susu | Roti Bakar | Confidence 100%, sangat kuat |
| Keripik Kentang | Roti Bakar | Confidence 75%, cukup kuat |

**Implementasi:**
- Train kasir untuk menanyakan apakah pelanggan ingin menambah Roti Bakar ketika mereka membeli Kopi Susu atau Keripik Kentang
- Tambahkan popup atau banner di layar POS (Point of Sale) yang menyarankan Roti Bakar sebagai item tambahan

#### 2. Penempatan Menu

- Letakkan Roti Bakar di lokasi yang sangat visible dan mudah diakses
- Tempatkan Kopi Susu dan Keripik Kentang di dekat area Roti Bakar
- Buat display khusus yang menampilkan ketiga item ini bersamaan dengan gambar menarik

#### 3. Paket Promosi

**Paket "Kopi & Roti" (Confidence 100%):**
- Kopi Susu + Roti Bakar = Diskon 10%
- Berdasarkan rule dengan confidence tertinggi, kombinasi ini hampir pasti diminati

**Paket "Sore Santai":**
- Kopi Susu + Roti Bakar + Keripik Kentang = Diskon 15%
- Memaksimalkan nilai transaksi dengan menggabungkan item yang sering dibeli bersama

#### 4. Manajemen Stok

- Prioritaskan ketersediaan stok Roti Bakar (item paling populer, 50% transaksi)
- Pastikan stok Kopi Susu dan Keripik Kentang selalu tersedia dalam jumlah yang cukup
- Hindari kekosongan stok pada Roti Bakar untuk mencegah kehilangan penjualan yang besar

#### 5. Promosi Digital

- Kirimkan notifikasi atau email marketing: "Kopi Susu enak diminum dengan Roti Bakar segar! Pesan sekarang dan dapatkan diskon 10%."
- Tampilkan banner promosi di aplikasi mobile atau website yang menampilkan kombinasi Kopi Susu + Roti Bakar

---

## 6. Kesimpulan

Penelitian ini telah berhasil menerapkan algoritma Association Rules (Apriori dan FP-Growth) untuk menganalisis pola pembelian pelanggan Cafe "Kopi Senja". Berikut adalah kesimpulan utama:

1. **Efektivitas Algoritma**: Kedua algoritma (Apriori dan FP-Growth) menghasilkan hasil yang identik dalam hal frequent itemsets dan association rules. Namun, FP-Growth lebih efisien secara komputasional karena hanya memerlukan 2 scan database dan tidak melakukan candidate generation yang tidak perlu.

2. **Pola Pembelian**: Roti Bakar adalah item yang paling populer (50% transaksi), sering dibeli bersama dengan Kopi Susu dan Keripik Kentang. Keripik Kentang menempati posisi kedua (40% transaksi).

3. **Association Rules**: Ditemukan 2 association rules yang memenuhi kriteria minimum confidence 70%:
   - {Kopi Susu} → {Roti Bakar} dengan confidence 100% dan lift 2.0
   - {Keripik Kentang} → {Roti Bakar} dengan confidence 75% dan lift 1.5
   Kedua rules memiliki nilai lift > 1, yang menunjukkan korelasi positif yang kuat antar item.

4. **Implementasi Bisnis**: Rekomendasi bisnis yang dibuat berdasarkan hasil analisis dapat digunakan untuk meningkatkan strategi cross-selling, penempatan menu, dan manajemen stok. Rule dengan confidence 100% sangat penting karena hampir selalu terjadi.

5. **Keterbatasan**: Penelitian ini menggunakan dataset yang relatif kecil (10 transaksi dengan 14 item unik). Untuk hasil yang lebih umum dan representatif, disarankan untuk menggunakan dataset yang lebih besar (ratusan atau ribuan transaksi).

6. **Total Item Terjual**: Dari 10 transaksi, terdapat 31 item terjual dengan rata-rata 3.1 item per transaksi. Hal ini menunjukkan bahwa pelanggan cenderung membeli beberapa item dalam satu kali transaksi, yang mendukung strategi cross-selling.

---

## 7. Saran

### 7.1 Pengembangan Lanjut

1. **Dataset yang Lebih Besar**: Lakukan analisis dengan dataset yang lebih besar (ratusan atau ribuan transaksi) untuk:
   - Mendapatkan hasil yang lebih representatif
   - Menemukan lebih banyak association rules
   - Mendapatkan kombinasi item yang lebih kompleks (3-itemset atau lebih)

2. **Analisis Time-Series**: Tambahkan dimensi waktu untuk menganalisis pola pembelian:
   - Waktu tertentu (pagi, siang, sore, malam)
   - Hari tertentu (weekday vs weekend)
   - Musim atau bulan tertentu

3. **Segmentasi Pelanggan**: Lakukan segmentasi pelanggan berdasarkan pola pembelian:
   - Pelanggan yang sering membeli kombinasi Kopi Susu + Roti Bakar
   - Pelanggan yang lebih menyukai minuman saja
   - Pelanggan yang sering membeli paket lengkap

4. **Algoritma Lain**: Coba algoritma lain untuk membandingkan hasil:
   - ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)
   - CARMA (Continuous Association Rule Mining Algorithm)
   - Penggunaan berbagai threshold untuk support dan confidence

### 7.2 Penerapan Algoritma Lain

1. **K-Means Clustering**: Untuk mengelompokkan pelanggan berdasarkan pola pembelian mereka, yang dapat digunakan untuk:
   - Targeted marketing
   - Personalisasi penawaran
   - Segmentasi pelanggan berdasarkan preferensi

2. **Sequential Pattern Mining**: Untuk menganalisis urutan pembelian pelanggan dari waktu ke waktu, yang dapat membantu:
   - Memahami alur perjalanan pelanggan
   - Memprediksi item apa yang mungkin dibeli selanjutnya
   - Mengoptimalkan penempatan menu berdasarkan urutan pembelian

3. **Collaborative Filtering**: Untuk membuat sistem rekomendasi yang lebih kompleks:
   - Rekomendasi berdasarkan pelanggan dengan preferensi serupa
   - Item-based collaborative filtering
   - Hybrid approach dengan content-based filtering

### 7.3 Implementasi Bisnis

1. **Integrasi dengan POS**: Integrasi hasil analisis dengan sistem Point of Sale untuk:
   - Rekomendasi otomatis saat kasir memasukkan item tertentu
   - Pop-up saran tambahan item
   - Pencatatan efektivitas promosi cross-selling

2. **Dashboard Monitoring**: Buat dashboard untuk memantau:
   - Performa association rules dari waktu ke waktu
   - Efektivitas promosi cross-selling
   - Perubahan pola pembelian pelanggan

3. **A/B Testing**: Lakukan A/B testing untuk:
   - Menguji efektivitas penempatan menu yang berbeda
   - Membandingkan performa paket promosi yang berbeda
   - Mengukur dampak rekomendasi digital terhadap penjualan

---

## 8. Referensi

1. Han, J., Pei, J., & Kamber, M. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.

2. Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules. *Proceedings of the 20th VLDB Conference*, 487-499.

3. Han, J., Pei, J., & Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation. *SIGMOD Record*, 29(2), 1-12.

4. Raschka, S. (2024). *MLxtend: Machine Learning Extensions*. GitHub Repository. https://github.com/rasbt/mlxtend

5. Tan, P.-N., Steinbach, M., & Kumar, V. (2005). *Introduction to Data Mining*. Pearson Addison Wesley.

---

## 9. Lampiran

### Lampiran 1: Kode Lengkap Python

Kode lengkap Python untuk implementasi dapat dilihat pada file `tugas4.ipynb`.

### Lampiran 2: Data Mentah

**Dataset Transaksi (10 Responden):**

```python
responden = [
    'Dinunaya Syuja Aryoko',
    'Ahmad Arya Dwi Febriansyah',
    'Vito Agatha Satritama',
    'Moh. Dani Wahyudi',
    'Achmad Wildan Muzaky',
    'Frenky',
    'Andries Nauvalentin Roestam',
    'Muhammad Alvin Firdaus',
    'Husni Mubarok',
    'Abiyyu Valin Zavero'
]

transactions = [
    ['Kopi Susu', 'Roti Bakar', 'Keripik Kentang'],
    ['Cappuccino', 'Croissant', 'Air Mineral'],
    ['Kopi Hitam', 'Roti Bakar', 'Keripik Kentang', 'Biskuit'],
    ['Kopi Susu', 'Roti Bakar', 'Jus Jeruk'],
    ['Teh Manis', 'Donut', 'Air Mineral'],
    ['Cappuccino', 'Croissant', 'Keripik Kentang'],
    ['Kopi Susu', 'Roti Bakar', 'Donut'],
    ['Latte', 'Nasi Goreng', 'Air Mineral'],
    ['Kopi Hitam', 'Roti Bakar', 'Keripik Kentang'],
    ['Teh Manis', 'Donut', 'Kue Coklat']
]
```

**Detail Per Transaksi:**

| Transaksi | Responden | Item yang Dipesan | Jumlah Item |
|-----------|-----------|-------------------|-------------|
| T1 | Dinunaya Syuja Aryoko | Kopi Susu, Roti Bakar, Keripik Kentang | 3 |
| T2 | Ahmad Arya Dwi Febriansyah | Cappuccino, Croissant, Air Mineral | 3 |
| T3 | Vito Agatha Satritama | Kopi Hitam, Roti Bakar, Keripik Kentang, Biskuit | 4 |
| T4 | Moh. Dani Wahyudi | Kopi Susu, Roti Bakar, Jus Jeruk | 3 |
| T5 | Achmad Wildan Muzaky | Teh Manis, Donut, Air Mineral | 3 |
| T6 | Frenky | Cappuccino, Croissant, Keripik Kentang | 3 |
| T7 | Andries Nauvalentin Roestam | Kopi Susu, Roti Bakar, Donut | 3 |
| T8 | Muhammad Alvin Firdaus | Latte, Nasi Goreng, Air Mineral | 3 |
| T9 | Husni Mubarok | Kopi Hitam, Roti Bakar, Keripik Kentang | 3 |
| T10 | Abiyyu Valin Zavero | Teh Manis, Donut, Kue Coklat | 3 |

### Lampiran 3: Rumus-rumus

**Support:**
```
Support(A → B) = (Jumlah transaksi mengandung A dan B) / (Total transaksi)
```

**Confidence:**
```
Confidence(A → B) = (Jumlah transaksi mengandung A dan B) / (Jumlah transaksi mengandung A)
```

**Lift:**
```
Lift(A → B) = Confidence(A → B) / Support(B)
```

### Lampiran 4: Ringkasan Hasil Lengkap

**Parameter yang Digunakan:**
- Minimum Support: 30% (0.3)
- Minimum Support Count: 3 transaksi
- Minimum Confidence: 70% (0.7)

**Frequent 1-Itemset (L1):**

| Item | Count | Support (%) |
|------|-------|-------------|
| Roti Bakar | 5 | 50% |
| Keripik Kentang | 4 | 40% |
| Kopi Susu | 3 | 30% |
| Air Mineral | 3 | 30% |
| Donut | 3 | 30% |

**Frequent 2-Itemset (L2):**

| Itemset | Count | Support (%) |
|---------|-------|-------------|
| (Roti Bakar, Kopi Susu) | 3 | 30% |
| (Roti Bakar, Keripik Kentang) | 3 | 30% |

**Association Rules yang Ditemukan:**

| Rule | Support | Confidence | Lift | Interpretasi |
|------|---------|------------|------|-------------|
| {Kopi Susu} → {Roti Bakar} | 30% | 100% | 2.0 | Positive correlation (sangat kuat) |
| {Keripik Kentang} → {Roti Bakar} | 30% | 75% | 1.5 | Positive correlation (kuat) |

**Perbandingan Algoritma:**

| Aspek | Apriori | FP-Growth |
|-------|---------|-----------|
| Jumlah Frequent Itemsets | 7 | 7 |
| Jumlah Association Rules | 2 | 2 |
| Scan Database | Banyak | 2 kali |
| Candidate Generation | Ya | Tidak |
| Hasil | Identik | Identik |

**Statistik Dataset:**
- Total Transaksi: 10
- Total Item Unik: 14
- Total Item Terjual: 31
- Rata-rata Item per Transaksi: 2.21
- Minimum Item per Transaksi: 3
- Maximum Item per Transaksi: 4

---

**End of Report**
