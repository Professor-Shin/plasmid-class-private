# สรุปสถานะโปรเจกต์: RefSeq Plasmid vs Chromosome Dataset Pipeline

เอกสารนี้สรุปว่า **ตอนนี้มีข้อมูลอะไรอยู่ในมือ** (จำนวนรวม/แยกตามคลาส/ที่มา) และ
**ได้ทำอะไรกับ pipeline ไปแล้วบ้าง**

---

## 1. ข้อมูลที่มีอยู่ในมือตอนนี้

### 1.1 แหล่งที่มา (source)

| รายการ | ค่า |
|---|---|
| ไฟล์ metadata ต้นทาง | `assembly_summary.txt` (235 MB, 530,916 แถว) |
| แหล่งที่มาอย่างเป็นทางการ | `https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt` |
| วันที่ดาวน์โหลด snapshot | 2026-09-15 |
| ขอบเขต | RefSeq bacteria เท่านั้น (`GCF_*`) — **ตรวจสอบแล้วว่าเป็นแบคทีเรีย 100%** ไม่มี Archaea ปน (ดูหัวข้อ 1.4) |

### 1.2 กรองแล้วเหลือเท่าไหร่ (ก่อนสุ่มตัวอย่าง)

จาก 530,916 แถว → กรองด้วยเงื่อนไข `assembly_level == "Complete Genome"`,
`version_status == "latest"`, ต้องเป็น RefSeq accession, มี `ftp_path` ใช้ได้,
และไม่ถูก NCBI แจ้ง `excluded_from_refseq` → เหลือ **66,230 assemblies**
(15,927 สปีชีส์ที่แตกต่างกัน)

### 1.3 ตัวอย่างที่ใช้สร้าง dataset จริง (สุ่มแบบ species-balanced)

จาก 66,230 ตัว สุ่มมา **1 assembly ต่อ 1 species_taxid** (seed=42) ได้ **500 assemblies**
- ดาวน์โหลด FASTA สำเร็จ 500/500 (0 ล้มเหลว, ตรวจ md5 ครบ), รวม ~0.70 GB

### 1.4 องค์ประกอบทางอนุกรมวิธาน (verified ด้วย NCBI Taxonomy)

| Domain | assemblies (จาก 66,230 ที่กรองแล้ว) | สปีชีส์ |
|---|---|---|
| **Bacteria** | **66,230 (100.00%)** | 15,927 |
| Archaea | 0 | 0 |

> คอลัมน์ `group` ใน assembly_summary.txt เขียนว่า "bacteria" ทุกแถวเพราะมันสะท้อน
> โฟลเดอร์ FTP ไม่ใช่ taxonomy จริง จึงตรวจสอบซ้ำด้วย NCBI Taxonomy lineage ของทุก
> species_taxid (15,927 ตัว) — ยืนยันว่าเป็นแบคทีเรียแท้ 100%

### 1.5 Replicon ที่แยกได้ (chromosome / plasmid) — จาก 500 assemblies ตัวอย่าง

| ประเภท | จำนวน replicon | หมายเหตุ |
|---|---|---|
| **Chromosome** | **527** | label = 0 |
| **Plasmid** | **381** | label = 1 |
| Unknown | 1 | ไม่ถูกบังคับเข้าคลาสใดคลาสหนึ่ง — excluded จาก dataset |
| **รวม** | **909 FASTA records** | = 909 replicons (1 record = 1 replicon เสมอ) |

การจำแนกประเภทมาจาก `*_assembly_report.txt` ของ NCBI เป็นหลัก (904/909 records)
ที่เหลือ 5 records ใช้ keyword จาก defline — **ตรวจสอบไขว้กับ NCBI bulk report
(prokaryotes.txt) แล้ว: 280/500 assemblies ที่ match กันได้ จำนวน
chromosome/plasmid ตรงกัน 100% (0 ความขัดแย้ง)**

การกระจายความยาว (bp):

| | n | min | median | mean | max | รวม bp |
|---|---|---|---|---|---|---|
| Chromosome | 527 | 105,350 | 4,005,916 | 4,469,783 | 13,696,185 | 2,355,575,937 |
| Plasmid | 381 | 1,119 | 63,562 | 150,908 | 1,900,099 | 57,496,151 |

สถิติอื่น: 166/500 assemblies มีทั้ง chromosome และ plasmid, 92 assemblies มี
plasmid มากกว่า 1 ตัว (มากสุด 16 ตัวในหนึ่ง assembly), ไม่พบ duplicate sequence,
พบ non-ACGT เพียง 13 base จาก 5 replicons (สัดส่วนน้อยมาก)

### 1.6 ขนาดประชากรทั้งหมด (ประมาณการ จาก 66,230 assemblies ทั้งหมด ไม่ใช่แค่ 500 ตัวอย่าง)

ใช้ NCBI bulk report `GENOME_REPORTS/prokaryotes.txt` ตรวจสอบได้ตรง 39,569/66,230
(59.7% — ส่วนที่เหลือเป็น assembly ใหม่กว่าที่ snapshot ยังไม่ทัน) ส่วนที่เหลือ
ประมาณการจากคอลัมน์ `replicon_count` ซึ่งตรงกับ bulk report ถึง 98.74%:

| | วัดตรง (matched) | ประมาณการ (unmatched) | รวมทั้งประชากร |
|---|---|---|---|
| assemblies | 39,569 | 26,661 | 66,230 |
| replicons รวม | 89,035 | 62,884 | ~151,919 |
| chromosomes | 40,903 | ~27,560 | ~68,463 |
| plasmids | 47,227 | ~34,714 | ~81,941 |

44.67% ของ assemblies ที่วัดตรงมี plasmid อย่างน้อย 1 ตัว — **สูงกว่า** สัดส่วน
33.2% ที่พบในตัวอย่าง 500 ตัว เพราะ species-balanced sampling ตัด strain ที่ซ้ำ
กันออก (เช่น E. coli, Klebsiella ที่มักมี plasmid เยอะและถูกแยกไปหลาย assembly)

### 1.7 Fragment (dataset สุดท้ายที่ใช้เทรน ML) — สร้างแล้วครบ

fragment ขนาด 5,000 bp แบบ non-overlapping จาก parent replicon ที่ยาว ≥ 5,000 bp
รวม **482,161 fragments** แบ่งตาม parent replicon cluster (ไม่ใช่สุ่มแบบ fragment):

| split | fragments | chromosome | plasmid | plasmid % | replicons | assemblies |
|---|---|---|---|---|---|---|
| train | 337,512 | 329,597 | 7,915 | 2.345% | 612 | 394 |
| validation | 72,317 | 70,621 | 1,696 | 2.345% | 131 | 105 |
| test | 72,332 | 70,636 | 1,696 | 2.345% | 126 | 112 |
| **รวม** | **482,161** | **470,854** | **11,307** | **2.345%** | 869 | 500 |

สัดส่วน plasmid เท่ากันทุก split (spread < 0.00001) — class balance ไม่เพี้ยนระหว่าง
split ตามที่ตั้งเป้าไว้ อัตราส่วน train/val/test ตรง 70/15/15 พอดี

---

## 2. สิ่งที่ทำไปแล้วกับ pipeline

Pipeline สร้างเป็น 7 ขั้นตอน (`pipeline/step01_*.py` ถึง `step07_*.py`)
สั่งรันทั้งหมดผ่าน `run_pipeline.py`:

| ขั้น | สคริปต์ | ทำอะไร | ผลลัพธ์ |
|---|---|---|---|
| 1 | `step01_filter_assemblies.py` | หา header row ของ assembly_summary.txt แบบ robust (ไม่ใช้ `comment="#"` เพราะจะลบ header ทิ้งด้วย), กรอง Complete Genome + latest + RefSeq, สุ่มแบบ species-balanced | `data/metadata/assemblies.csv` (500 แถว) |
| 2 | `step02_download_fasta.py` | ดาวน์โหลด `*_genomic.fna.gz` และ `*_assembly_report.txt` ต่อ assembly, resumable + retry + ตรวจ md5 | `data/fasta/GCF_.../` (500 โฟลเดอร์, 0.70 GB) |
| 3 | `step03_parse_replicons.py` | แยก FASTA record เป็น replicon จริง จำแนก chromosome/plasmid/unknown จาก assembly_report ก่อน แล้วค่อย fallback ไป defline keyword — ไม่เดาจากลำดับ record | `data/metadata/replicons.csv` (909 replicons) |
| 4 | `step04_qc.py` | ตรวจคุณภาพข้อมูล 14 รายการตามสเปก (จำนวน assembly/replicon, length distribution, duplicate, non-ACGT ฯลฯ) ไม่ทิ้งข้อมูลใดๆ | `reports/data_qc.csv`, `data_qc.txt` |
| 5 | `step05_cluster.py` | ลด redundancy ก่อน split: ทำ MinHash sketch (k=21) ประเมิน ANI + containment เพื่อจับ replicon ที่เหมือน/ซ้ำกันมาก รวมเป็น cluster เดียวกัน (single-linkage) | `data/metadata/clusters.csv` (869→843 clusters) |
| 6 | `step06_split.py` | แบ่ง train/val/test **ที่ระดับ cluster** ไม่ใช่ fragment เพื่อกัน data leakage, รักษาสัดส่วน 70/15/15 และ class balance | `data/split/{train,val,test}_replicons.csv` |
| 7 | `step07_fragments.py` | ตัด fragment ยาว 5,000 bp แบบ non-overlapping จาก parent replicon ตาม split ที่กำหนดไว้แล้ว พร้อมตรวจสอบความถูกต้อง (fragment_id ตรงกันระหว่าง FASTA/CSV, ไม่มี replicon ข้าม split) | `data/dataset/{train,val,test}.{fasta,csv}` (482,161 fragments) |

### สิ่งที่แก้ไขเพิ่มเติมระหว่างทาง

1. **MinHash sketcher ใช้ memory เกินและช้า** — เขียนใหม่ให้ประมวลผลเป็น chunk
   ด้วย uint64 doubling แทนการสร้าง float64 array ของทั้ง genome ทำให้เร็วขึ้น 6
   เท่า (0.62 วินาที/5 Mbp) และ memory คงที่ไม่ว่า replicon จะยาวแค่ไหน
2. **Length ratio อย่างเดียวจับ containment ไม่ได้** — พบ plasmid 39 kb ที่มี ANI
   99.4% กับ plasmid 72 kb แต่ length ratio แค่ 0.54 (หลุดเกณฑ์เดิม) จึงเพิ่มเกณฑ์
   containment (สัดส่วน k-mer ของตัวสั้นที่พบในตัวยาว ≥ 90%) และป้องกัน false
   positive ด้วยการกำหนด sample size ขั้นต่ำ (≥50 hash ที่ overlap กัน)

### สคริปต์ตรวจสอบเพิ่มเติม (นอกเหนือจาก pipeline หลัก)

- **`verify_dataset.py`** — ตรวจสอบ dataset สุดท้ายแบบอิสระ (ไม่พึ่ง logic เดิมของ
  pipeline) 20 รายการ: fragment_id ไม่ซ้ำ, FASTA/CSV ตรงกัน, ไม่มี replicon ข้าม
  split, label ตรงกับ replicon_type, **สุ่มตรวจ fragment กับลำดับ genome ต้นฉบับ
  แบบ byte-to-byte** ฯลฯ — **ผ่านทั้งหมด**
- **`analyze_population.py`** — ตอบคำถามว่าทั้งประชากร 66,230 assemblies (ไม่ใช่
  แค่ 500 ตัวอย่าง) เป็นแบคทีเรียจริงหรือไม่ (ตรวจกับ NCBI Taxonomy) และมี
  chromosome/plasmid รวมเท่าไหร่ (ตรวจกับ NCBI bulk report) → ผลลัพธ์อยู่ในหัวข้อ
  1.4 และ 1.6 ข้างบน

### ความสามารถด้าน reproducibility

ทุกพารามิเตอร์ (assembly_level, sampling strategy, similarity threshold, split
ratio, fragment length, random seed) เก็บไว้ใน `config.yaml` และมีบันทึกครบใน
`reports/run_manifest.json` (source URL, วันที่ดาวน์โหลด, software version,
จำนวนในแต่ละขั้นตอน) — รันด้วย config + snapshot เดียวกันจะได้ dataset เดิมทุก
ครั้ง

---

## 3. งานที่ยังไม่ได้ทำ / ข้อควรระวัง

- Species-balanced sampling (1 assembly/species) ทำให้สัดส่วน plasmid ใน 500
  ตัวอย่าง (33.2%) ต่ำกว่าประชากรจริง (44.67%) — ถ้าต้องการ plasmid มากขึ้นให้
  ปรับ `max_per_species` เป็น 2-3 ใน `config.yaml`
- ตัวเลข chromosome/plasmid ของ assemblies ที่ยังไม่ match กับ bulk report
  (26,661 ตัว) เป็นค่าประมาณการ ไม่ใช่ค่าที่วัดตรง — ถ้าต้องการค่าที่แม่นยำ 100%
  ต้องเรียก NCBI Datasets API ต่อ assembly เพิ่มเติม (ใช้เวลาหลายชั่วโมง)
- Class imbalance ระหว่าง chromosome/plasmid (2.345% plasmid) ถูกเก็บไว้ตามจริง
  ไม่ได้ปรับสมดุลให้ — ต้องจัดการที่ขั้นตอนเทรนโมเดล (เช่น class weight)
