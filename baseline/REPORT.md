# รายงาน Baseline: PlasFlow, Zou et al. และ PlasClass บน RefSeq Plasmid vs Chromosome Dataset

เอกสารนี้อธิบายการ **นำงานตีพิมพ์ 3 ชิ้นมาสร้างใหม่ (re-implement) และเทรนใหม่ทั้งหมดบนชุดข้อมูลของเรา**
เพื่อใช้เป็นเส้นฐานเปรียบเทียบ (baseline) สำหรับโมเดลจำแนก plasmid/chromosome ที่จะพัฒนาต่อ

| Baseline | บทความ | โค้ดต้นฉบับที่ใช้อ้างอิง |
|---|---|---|
| **PlasFlow** | Krawczyk PS, Lipinski L, Dziembowski A. *PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures.* Nucleic Acids Res. 2018;46(6):e35 | <https://github.com/smaegol/PlasFlow> |
| **Zou et al.** | Zou X, Nguyen M, Overbeek J, Cao B, Davis JJ. *Classification of bacterial plasmid and chromosome derived sequences using machine learning.* PLoS ONE. 2022;17(12):e0279280 | <https://github.com/BV-BRC-dependencies/zou-plasmid-prediction> |
| **PlasClass** | Pellow D, Zorea A, Probst M, et al. *Plasmid classification in metagenomic datasets using an improved machine learning approach.* PLoS Comput Biol. 2020;16(10):e1007781 | <https://github.com/Shamir-Lab/PlasClass> |

---

## 1. เหตุผลในการเลือกสามงานนี้

จากการพิจารณาบทความก่อนหน้า มีเพียงกลุ่มนี้ที่เป็น **sequence-signature classifier ล้วน**
คือใช้เฉพาะความถี่ของ k-mer จากลำดับดีเอ็นเอ ไม่ต้องพึ่ง gene calling, โปรตีน, หรือฐานข้อมูลอ้างอิงภายนอก
จึงเทียบได้ตรงกับ dataset ของเราที่เป็น fragment ดีเอ็นเอ 5,000 bp พร้อม label เท่านั้น

งานอีกสามชิ้น (PLASMe, PlasmidHunter, Deeplasmid) ใช้ข้อมูลระดับยีน/โปรตีนหรือ alignment กับฐานข้อมูล
plasmid ที่รู้จักแล้ว ซึ่ง (ก) ต้องสร้าง infrastructure เพิ่มทั้งชุด และ (ข) เสี่ยงต่อการที่ฐานข้อมูลอ้างอิง
ของเครื่องมือเหล่านั้นมี replicon ชุดเดียวกับ test set ของเราอยู่แล้ว จึงไม่นำมาทำ baseline ในรอบนี้

> **หมายเหตุเรื่องแหล่งโค้ดของ PlasClass:** ลิงก์ที่ได้รับมาตอนแรกคือ
> <https://gitlab.com/sirarredondo/Plasmid_Assembly> ซึ่งตรวจสอบแล้วพบว่าเป็น pipeline สำหรับรัน
> **PlasmidSPAdes / Recycler / cBAR / PlasmidFinder** (งาน *ประกอบ* พลาสมิดจาก short reads, สร้างปี 2016)
> **ไม่มีโค้ดของ PlasClass** และเก่ากว่า PlasClass (2020) อยู่ 4 ปี จึงถอดสถาปัตยกรรมจาก repo ทางการ
> `Shamir-Lab/PlasClass` แทน อนึ่ง เครื่องมือใน repo นั้นส่วนใหญ่ต้องใช้ **reads หรือ assembly graph**
> ซึ่ง dataset ของเราไม่มี (เรามีเฉพาะลำดับ fragment) จึงรันไม่ได้ในทุกกรณี

### 1.1 ทำไมต้อง "เทรนใหม่" ไม่ใช้โมเดลสำเร็จรูป

ทั้งสามงานปล่อยน้ำหนักโมเดลที่เทรนแล้วมาด้วย (PlasFlow มี TensorFlow checkpoint, Zou มี
`5k6mer_model.h5`, PlasClass มีไฟล์ `m1000`–`m500000` ที่ pickle ไว้)
แต่เราเลือก **เทรนใหม่จากศูนย์บน train split ของเราเอง** ด้วยเหตุผลสำคัญคือ

1. **การปนเปื้อนของชุดทดสอบ** — PlasFlow เทรนจาก RefSeq, Zou เทรนจาก BV-BRC/PATRIC และ
   PlasClass เทรนจาก NCBI/PLSDB ซึ่งทั้งหมด
   ครอบคลุม genome ชุดเดียวกับที่ dataset ของเราสุ่มมา ถ้ารันโมเดลสำเร็จรูปบน test set ของเรา
   โมเดลเหล่านั้นมีโอกาสสูงที่จะ "เคยเห็น" replicon เดียวกันมาแล้วตอนเทรน ผลที่ได้จะสูงเกินจริง
   และไม่สามารถนำไปเทียบกับโมเดลของเราที่ถูกกันข้อมูลอย่างเข้มงวดด้วย cluster-level split
2. **ความเท่าเทียมของเงื่อนไข** — baseline ที่ดีต้องเห็นข้อมูลเทรนชุดเดียวกัน ถูกวัดบน test ชุดเดียวกัน
   และอยู่ภายใต้ข้อจำกัดเรื่อง data leakage เดียวกันกับโมเดลที่เราจะพัฒนา
3. ความแตกต่างที่วัดได้จึงมาจาก **สถาปัตยกรรมและ feature representation** ไม่ใช่จากความต่างของชุดข้อมูล

> ผลที่รายงานในเอกสารนี้จึงเป็น *"สถาปัตยกรรมของ PlasFlow/Zou/PlasClass เมื่อเทรนบนข้อมูลของเรา"*
> ไม่ใช่ *"ประสิทธิภาพของเครื่องมือเหล่านั้นตามที่ผู้เขียนเผยแพร่"* — ห้ามนำตัวเลขไปอ้างสลับกัน

---

## 2. ข้อมูลที่ใช้

ใช้ dataset จาก `../dataset pipeline1` โดยตรงและ **ไม่แตะการแบ่ง split เดิม** (แบ่งที่ระดับ replicon cluster
ก่อนตัด fragment เพื่อกัน leakage)

| split | fragments | chromosome | plasmid | plasmid % |
|---|---|---|---|---|
| train | 337,512 | 329,597 | 7,915 | 2.345% |
| validation | 72,317 | 70,621 | 1,696 | 2.345% |
| test | 72,332 | 70,636 | 1,696 | 2.345% |

fragment ทุกชิ้นยาว 5,000 bp พอดี แบบ non-overlapping มาจาก 500 assemblies / 869 replicons / 843 clusters

### 2.1 สองระบอบการเทรน (training regimes)

ประเด็นสำคัญที่สุดของการทำ baseline ครั้งนี้คือ **ทุกเปเปอร์เทรนบนข้อมูลที่สมดุลเกือบ 1:1**
(PlasFlow: 1,961 chromosome / 7,604 plasmid replicons; Zou: 10,584 / 10,654 sequences;
PlasClass: สุ่ม fragment จำนวนเท่ากันทั้งสองคลาสเสมอในโค้ดเทรนของมันเอง)
ขณะที่ dataset ของเราเก็บสัดส่วนตามธรรมชาติไว้ที่ plasmid เพียง 2.345% (อัตราส่วนราว 43:1)

ถ้าเทรนตามสถาปัตยกรรมเดิมบนข้อมูลที่ไม่สมดุลตรง ๆ แล้วได้ผลแย่ จะสรุปไม่ได้ว่าเป็นเพราะสถาปัตยกรรม
หรือเพราะ class imbalance เราจึงเทรน **ทั้งสองแบบ** และวัดผลบน test set ชุดเดิมที่ไม่สมดุล (ของจริง) เสมอ

| ระบอบ | ข้อมูลเทรน | validation | เหตุผล |
|---|---|---|---|
| **A: balanced** | สุ่มลด chromosome ให้เหลือ 1:1 → 15,830 fragments (7,915 ต่อคลาส) | 3,392 (1:1) | ทำซ้ำเงื่อนไขเดิมของทั้งสองเปเปอร์ |
| **B: full** | train split เต็ม 337,512 fragments (2.345% plasmid) | 72,317 (เต็ม) | เงื่อนไขจริงที่โมเดลบน dataset นี้ต้องเจอ |

test set ที่ใช้วัดผล = 72,332 fragments เต็ม (1,696 plasmid) ทั้งสองระบอบ

**PlasClass มีเฉพาะระบอบ A** เพราะ (ก) `plasclass/train.py` สุ่ม fragment เท่ากันทั้งสองคลาสเสมอ
ระบอบ B จึงไม่ใช่วิธีของมัน และ (ข) เมทริกซ์ฟีเจอร์ของ train split เต็มมีขนาด
337,512 × 10,952 = 14.8 GB ซึ่งเกินหน่วยความจำของเครื่อง (8 GB) และ logistic regression
ต้องใช้ข้อมูลทั้งก้อนพร้อมกัน ต่างจากโครงข่ายประสาทที่เทรนแบบ minibatch ได้

---

## 3. สภาพแวดล้อมการทดลอง (experimental setup)

ตัวเลขทุกตัวในหัวข้อ 5 ผลิตจากสภาพแวดล้อมที่ระบุไว้ด้านล่างนี้ **ครั้งเดียว ในวันที่ 2026-09-23**
ข้อมูลทั้งหมดในหัวข้อนี้ถูกบันทึกอัตโนมัติไว้ที่ `results/environment.json` ด้วย
`capture_environment.py` (รันด้วยอินเทอร์พรีเตอร์ตัวเดียวกับที่ใช้ทดลอง) จึงตรวจสอบย้อนหลังได้

### 3.1 ฮาร์ดแวร์

| รายการ | ค่า |
|---|---|
| CPU | Intel Core i5-8300H @ 2.30 GHz (Coffee Lake, 4 คอร์จริง / 8 เธรด, max 2,304 MHz) |
| RAM | 7.88 GiB (ระหว่างการทดลองมีหน่วยความจำว่างจริงประมาณ 1.4 GB) |
| Storage | ADATA LEGEND 710 SSD 477 GB (ข้อมูลและ cache ทั้งหมดอยู่บนไดรฟ์ D:) |
| GPU | **ไม่มี / ไม่ได้ใช้** — `torch.cuda.is_available()` = `False` ทุกการรัน |

ข้อจำกัดเรื่อง RAM 8 GB เป็นตัวกำหนดสถาปัตยกรรมของโค้ดโดยตรง (ดูหัวข้อ 4.6) เพราะ feature matrix
ของ 7-mer เพียงชุดเดียวมีขนาด 10.3 GiB ซึ่งใหญ่กว่าหน่วยความจำทั้งเครื่อง

### 3.2 ระบบปฏิบัติการและอินเทอร์พรีเตอร์

| รายการ | ค่า |
|---|---|
| OS | Microsoft Windows 11 Pro, build 10.0.22631, 64-bit |
| Shell | PowerShell 5.1 |
| Python | CPython 3.13.5 (AMD64) |
| ที่อยู่ | `C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe` |
| วิธีเรียก | `py -3.13` (Python ไม่ได้อยู่ใน `PATH` ของเชลล์นี้ การเรียก `python` เฉย ๆ จะไปโดน stub ของ Microsoft Store) |

> เครื่องนี้มี Python 3.11 ติดตั้งอยู่ด้วย แต่ **การทดลองทั้งหมดใช้ 3.13.5 เท่านั้น**
> ถ้ารันซ้ำด้วย `python` แทน `py -3.13` อาจได้อินเทอร์พรีเตอร์คนละตัวและเวอร์ชันไลบรารีคนละชุด

### 3.3 ไลบรารีและเวอร์ชัน

| แพ็กเกจ | เวอร์ชัน | ใช้ทำอะไร |
|---|---|---|
| `torch` | **2.11.0+cpu** | โมเดลทั้งสองสถาปัตยกรรม (CPU build, ไม่มี CUDA) |
| `numpy` | 2.4.4 | การนับ k-mer, memmap, transform ทั้งหมด |
| `pandas` | 3.0.2 | อ่าน metadata CSV ของ dataset |
| `scikit-learn` | 1.8.0 | metrics (AUROC, AUPRC, MCC, confusion matrix) และใช้เป็นตัวอ้างอิงตอน verify |
| `scipy` | 1.17.1 | dependency ของ scikit-learn |
| `PyYAML` | 6.0.3 | อ่าน `config.yaml` |
| `joblib` | 1.5.3 | dependency ของ scikit-learn |
| `threadpoolctl` | 3.6.0 | dependency ของ scikit-learn |

คำสั่งติดตั้งให้ได้สภาพแวดล้อมเดียวกัน:

```bash
py -3.13 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
py -3.13 -m pip install numpy==2.4.4 pandas==3.0.2 scikit-learn==1.8.0 scipy==1.17.1 PyYAML==6.0.3
```

**backend การคำนวณของ PyTorch** (จาก `torch.__config__.parallel_info()`):

```
ATen parallel backend : OpenMP
Intel oneAPI MKL      : 2025.3-Product Build 20251007
Intel MKL-DNN         : v3.10.2 (git f1d471933dc852f956fd05389f9313c7148783d5)
at::get_num_threads() : 4   (ค่าเริ่มต้น = จำนวนคอร์จริง)
OMP_NUM_THREADS       : ไม่ได้ตั้ง
MKL_NUM_THREADS       : ไม่ได้ตั้ง
```

ไม่มีการติดตั้ง TensorFlow, Biostrings/rpy2 หรือ KMC3 — ทั้งสามอย่างที่โค้ดต้นฉบับต้องใช้ถูกแทนที่ด้วย
การ implement ใหม่ที่ตรวจสอบความถูกต้องแล้ว (หัวข้อ 4.4)

### 3.4 ที่มาของโค้ดต้นฉบับที่ใช้ถอดสถาปัตยกรรม

โคลนทั้งสาม repository เมื่อ **2026-09-23** และอ่านโค้ดจาก commit ที่ระบุด้านล่าง
(บันทึกไว้ในส่วน `upstream:` ของ `config.yaml` ด้วย)

| repo | commit | วันที่ของ commit | ไฟล์ที่ใช้ถอดสถาปัตยกรรม |
|---|---|---|---|
| `github.com/smaegol/PlasFlow` | `45604ff637c65f709dabff63e665fa60896be241` | 2021-06-28 | `PlasFlow.py`, `scripts/PlasFlow_train.py` |
| `github.com/BV-BRC-dependencies/zou-plasmid-prediction` | `2519c21ddfbd3900d272c5a40b59e1501278ecd0` | 2022-11-15 | `model_train.py`, `model_predict.py`, `README.md`, `kmer_order.csv` |
| `github.com/Shamir-Lab/PlasClass` | `bcf19c7de9022344afc075edbfa2facf8bf38c57` | 2023-07-25 | `plasclass/train.py`, `plasclass/plasclass.py`, `plasclass/plasclass_utils.py` |

ไฟล์ `kmer_order.csv` (2,080 บรรทัด) ถูกคัดลอกเข้ามาเก็บไว้ที่ `reference/kmc_kmer_order.csv`
เพื่อให้การตรวจสอบลำดับคอลัมน์ canonical 6-mer ทำซ้ำได้โดยไม่ต้องโคลน repo ใหม่
**ไม่ได้ใช้น้ำหนักโมเดลที่เทรนแล้วจากทั้งสาม repo** (เหตุผลในหัวข้อ 1.1)

### 3.5 ชุดข้อมูลที่ป้อนเข้าและ checksum

ไฟล์ต้นทางทั้ง 6 ไฟล์จาก `../dataset pipeline1/data/dataset/` (สร้างเมื่อ 2026-09-15) —
ถ้า checksum ไม่ตรงกับด้านล่าง แปลว่าเป็นคนละชุดข้อมูลและ **ตัวเลขผลลัพธ์จะเทียบกันไม่ได้**

| ไฟล์ | ขนาด (ไบต์) | SHA-256 |
|---|---|---|
| `train.fasta` | 1,718,273,592 | `dc501eb6880875244a07c358784da39be61f16c47a3872a3f40256cb2635e20f` |
| `train.csv` | 59,128,170 | `4537babcbf959badb775abce1e8960e6b0da7bb970ac7bf338184b80b5fe9c77` |
| `validation.fasta` | 368,165,847 | `4d2c8018d48cdf6b3952f9e9a55e529665115566f7f30a88aabb0dd40bb0c68b` |
| `validation.csv` | 13,001,838 | `90982ec33e40149cc11c831a1221454aa85d30000a4170f997398d4a1ae6c0e6` |
| `test.fasta` | 368,242,212 | `b08c1a24028591ca4ec76005c2caaaa006fca2559e26512a87f579c981f4d501` |
| `test.csv` | 12,630,995 | `960e57b6b2596ddbfb2bf3c690915abb19fe731c7fb7fb7e5c730f2b4b2166e5` |

ไฟล์กำหนดค่าที่ใช้: `config.yaml`
SHA-256 = `607385350a68ebc9e465487ab441757677ad45e635d62bea85d584569f11d90f`
(เนื้อหาเต็มถูกคัดลอกลงใน `results/environment.json` ภายใต้คีย์ `config.values` ด้วย)

`capture_environment.py` ยังบันทึก SHA-256 ของไฟล์ `.py` ทุกไฟล์ใน `baseline/` ไว้ด้วย
จึงยืนยันได้ว่าผลลัพธ์มาจากโค้ดเวอร์ชันใด

### 3.6 การตั้งค่าจำนวนเธรด (มีผลต่อเวลา ไม่มีผลต่อตัวเลขความแม่นยำ)

CPU นี้มี 4 คอร์จริง และ PyTorch ตั้งค่าเริ่มต้นเป็น 4 เธรด แต่ batch ของงานนี้เล็กมาก
(PlasFlow ใช้ batch 32 ตามค่า default ของ `DNNClassifier`) จนค่าใช้จ่ายในการซิงก์เธรดของ OpenMP
มากกว่าประโยชน์ที่ได้จากการขนาน จึง **เบนช์มาร์กก่อนตั้งค่า** (วัดเวลาต่อ optimization step จริง 300 step)

| การตั้งค่า | 1 เธรด | 2 เธรด | 4 เธรด | 8 เธรด |
|---|---|---|---|---|
| PlasFlow k=5 (batch 32, 1,024 features) | 3.04 ms | **0.99 ms** | 1.21 ms | 1.64 ms |
| PlasFlow k=7 (batch 32, 16,384 features) | 24.36 ms | **7.63 ms** | 8.33 ms | 9.76 ms |
| Zou (batch 256, 2,080 features) | — | 25.79 ms | **19.04 ms** | 22.33 ms |

จึงตั้ง `torch_threads: 2` สำหรับ PlasFlow และ `torch_threads: 4` สำหรับ Zou ใน `config.yaml`
(การตั้ง 8 เธรดบนซีพียู 4 คอร์จริงช้ากว่าทุกกรณี เพราะ hyper-threading ไม่ช่วยงาน matmul ที่ใช้ SIMD เต็มแล้ว)
ถ้ารันบนเครื่องอื่นควรเบนช์มาร์กใหม่ ค่าเหล่านี้ไม่ใช่ค่าสากล

### 3.7 ลำดับคำสั่งที่รันจริงและเวลาที่ใช้

รันตามลำดับนี้ทั้งหมดในเซสชันเดียว log ดิบเก็บไว้ที่ `results/*.log`

| ลำดับ | คำสั่ง | เวลาที่ใช้ (นาฬิกาจริง) | ผลลัพธ์ |
|---|---|---|---|
| 1 | `py -3.13 verify_features.py` | ~5 วินาที | ผ่านทุกข้อ (ไม่มีไฟล์ output) |
| 2 | `py -3.13 prepare_features.py` | **15 นาที 52 วินาที** (04:45:20 → 05:01:12) | `cache/` 23.4 GiB, `results/prepare_features.log` |
| 3 | `py -3.13 run_plasflow.py` | **~26 นาที** (→ 05:32:47) | `results/plasflow_{balanced,full}.json`, `run_plasflow.log` |
| 4 | `py -3.13 run_zou.py` | **27 นาที 10 วินาที** (05:32:47 → 05:59:57) | `results/zou_{balanced,full}.json`, `run_zou.log` |
| 5 | `py -3.13 prepare_features.py` (รอบสองสำหรับ PlasClass) | **~9 นาที** (2026-09-24) | canonical k=3,4,5,7 เพิ่ม 8.0 GiB, `prepare_features_plasclass.log` |
| 6 | `py -3.13 run_plasclass.py` | **~2 นาที** | `results/plasclass_balanced.json`, `run_plasclass.log` |
| 7 | `py -3.13 summarize.py` | 45 วินาที | `results/summary.md` |
| 8 | `py -3.13 capture_environment.py` | ~40 วินาที (ส่วนใหญ่คือ checksum ไฟล์ 2.4 GB) | `results/environment.json` |

รวมทั้งหมดประมาณ **80 นาที** บนเครื่องที่ระบุไว้ข้างต้น โดยไม่ใช้ GPU
(PlasFlow + Zou รันวันที่ 2026-09-23, ส่วน PlasClass เพิ่มเข้ามาวันที่ 2026-09-24
โดยใช้ feature cache เดิมซ้ำและสร้างเพิ่มเฉพาะ canonical k=3,4,5,7)
(แยกตามระบอบ: PlasFlow A จบ 05:14:32, PlasFlow B จบ 05:32:47, Zou A จบ 05:44:48, Zou B จบ 05:59:57)
เวลาที่มากกว่าผลรวมในตาราง "เวลาเทรน" ด้านล่างคือเวลาที่ใช้ทำนายผลบน validation และ test เต็มทั้งสองชุด
ซึ่งสำหรับ k=7 ต้องอ่าน feature 2.2 GiB × 2 ชุด ต่อหนึ่งโมเดล

รายละเอียดเวลาของขั้นที่ 2 (สร้าง feature):

| ขั้นตอน | train | validation | test |
|---|---|---|---|
| encode FASTA → รหัสเบส uint8 | 20.3 s | 4.3 s | 4.4 s |
| นับ 5-mer (1,024 features) | 123.7 s | 15.5 s | 14.9 s |
| นับ 6-mer (4,096 features) | 146.8 s | 18.8 s | 19.3 s |
| นับ 7-mer (16,384 features) | 231.0 s | 82.4 s | 30.4 s |
| นับ canonical 6-mer (2,080 features) | 157.5 s | 18.7 s | 18.8 s |
| นับ canonical 3-mer (32 features, PlasClass) | 56.6 s | 11.7 s | 11.4 s |
| นับ canonical 4-mer (136 features, PlasClass) | 58.5 s | 13.9 s | 13.3 s |
| นับ canonical 5-mer (512 features, PlasClass) | 68.0 s | 14.7 s | 14.8 s |
| นับ canonical 7-mer (8,192 features, PlasClass) | 188.5 s | 25.8 s | 25.3 s |

รายละเอียดเวลาของขั้นที่ 3–4 (เทรน, วินาทีต่อโมเดล):

| การรัน | สมาชิก 1 | สมาชิก 2 | สมาชิก 3 | รวม |
|---|---|---|---|---|
| PlasFlow ระบอบ A | k5: 61 s | k6: 84 s | k7: 411 s | 556 s |
| PlasFlow ระบอบ B | k5: 67 s | k6: 117 s | k7: 910 s | 1,094 s |
| Zou ระบอบ A | 156 s | 387 s | 162 s | 705 s |
| Zou ระบอบ B | 250 s | 424 s | 234 s | 908 s |
| PlasClass ระบอบ A | logistic regression: 58 s (290 iterations) | — | — | 58 s |

(เวลาของ Zou แต่ละตัวต่างกันเพราะ early stopping หยุดคนละ epoch ส่วน PlasFlow k=7 ช้าที่สุด
เพราะต้องอ่านข้อมูล 10.3 GiB จากดิสก์วนซ้ำ ~4.7 รอบตลอด 50,000 steps)

### 3.8 พื้นที่ดิสก์ที่ใช้

| สิ่งที่สร้าง | ขนาด |
|---|---|
| `cache/` (62 ไฟล์: รหัสเบส + feature matrix ทุกชุด + สถิติ transform) | **31.4 GiB** |
| — `seq_*.u8` (รหัสเบส 3 splits) | 2.25 GiB |
| — `feat_*_k5f.u16` | 0.92 GiB |
| — `feat_*_k6f.u16` | 3.68 GiB |
| — `feat_*_k7f.u16` | 14.72 GiB |
| — `feat_*_k6c.u16` (canonical, สำหรับ Zou และ PlasClass) | 1.87 GiB |
| — `feat_*_k3c/k4c/k5c/k7c.u16` (canonical, สำหรับ PlasClass) | 8.00 GiB |
| `results/probs/` (ความน่าจะเป็นราย fragment 32 ไฟล์) | ~18 MB |
| `results/*.json` + `*.log` + `summary.md` | ~0.2 MB |

`cache/` และ `results/probs/` ถูกใส่ไว้ใน `.gitignore` เพราะสร้างใหม่ได้เสมอจากชุดข้อมูลเดิม
ควรมีพื้นที่ว่างอย่างน้อย **40 GB** ก่อนรัน

### 3.9 ขอบเขตของการทำซ้ำได้ (determinism)

**สิ่งที่กำหนดด้วย seed** — `random_seed: 42` ใน `config.yaml` ถูกใช้สร้าง

- ลำดับสุ่มของแถว train ที่เขียนลง memmap (`cache/train_perm.npy`)
- การสุ่มลดคลาส chromosome ให้สมดุล 1:1 ในระบอบ A (ทั้ง train และ validation)
- การสุ่มเลือก 6,000 แถวของ validation ที่ใช้เฝ้าดูเส้นการเรียนรู้ของระบอบ B
- การสุ่มเลือก test subsample แบบ 1:1 ในหัวข้อ 5.3
- น้ำหนักเริ่มต้นและ dropout mask ของทุกโมเดล ผ่าน `torch.manual_seed` โดยใช้ seed
  **42 + k** สำหรับ PlasFlow (ได้ 47, 48, 49) และ **42 / 142 / 242** สำหรับสมาชิกทั้งสามของ Zou

**รับประกันได้:** รันซ้ำบน *เครื่องเดียวกัน* ด้วย `config.yaml` และชุดข้อมูล checksum เดิม
จะได้ตัวเลขเดิมทุกหลัก (ไม่มี CUDA จึงไม่มีความไม่แน่นอนจาก cuDNN และ op ที่ใช้บน CPU เป็น deterministic ทั้งหมด)

**ไม่รับประกัน:** ถ้าเปลี่ยนเครื่อง เปลี่ยนจำนวนเธรด หรือเปลี่ยนเวอร์ชัน MKL/PyTorch
ลำดับการบวกใน matmul อาจต่างกันเล็กน้อย ทำให้ผลต่างกันในทศนิยมท้าย ๆ (ไม่กระทบข้อสรุป)
นอกจากนี้ **ถ้าเปลี่ยนค่า `block_bytes`** ขนาด block ของ loader จะเปลี่ยน ลำดับตัวอย่างที่โมเดลเห็น
จึงเปลี่ยนตาม และผลจะไม่เท่าเดิมทุกหลัก — ค่านี้จึงถูกบันทึกไว้ใน `config.yaml` และ `environment.json`

### 3.10 ไฟล์บันทึกสภาพแวดล้อม

```bash
py -3.13 capture_environment.py     # -> results/environment.json
```

`results/environment.json` บันทึก: เวลาที่บันทึก, สเปกฮาร์ดแวร์, OS, เวอร์ชัน Python และทุกแพ็กเกจ,
ข้อมูล backend ของ PyTorch, การตั้งค่าเธรด, commit ของ repo ต้นทาง, เนื้อหาและ checksum ของ
`config.yaml`, ขนาดและ SHA-256 ของไฟล์ชุดข้อมูลทั้ง 6 ไฟล์, SHA-256 ของซอร์สโค้ดทุกไฟล์ใน
`baseline/`, และขนาดรวมของ `cache/` — ให้แนบไฟล์นี้ไปกับผลการทดลองเสมอ

---

## 4. วิธีการ

### 4.1 PlasFlow — สถาปัตยกรรมที่ถอดจากโค้ดต้นฉบับ

อ่านจาก `PlasFlow.py` และ `scripts/PlasFlow_train.py` ได้โครงสร้างดังนี้

```
FASTA → oligonucleotideFrequency (Biostrings) → นับ k-mer ดิบ 4^k คอลัมน์ (สายเดียว ไม่ทำ canonical)
      → TF-IDF (TfidfTransformer(smooth_idf=False) + L2 norm)
      → DNNClassifier 3 ตัว:  k=5 hidden [20,20] | k=6 hidden [20,20] | k=7 hidden [30]  (ReLU)
      → เฉลี่ยความน่าจะเป็นจากทั้ง 3 ตัว (voting)
      → ถ้าความน่าจะเป็นสูงสุด < 0.7 → "unclassified"
```

| พารามิเตอร์ | ค่าที่ใช้ | ที่มา |
|---|---|---|
| optimizer | Adagrad, lr = 0.05, initial_accumulator = 0.1 | ค่า default ของ `tf.contrib.learn.DNNClassifier` |
| batch size | 32 | ค่า default ของ DNNClassifier |
| training steps | 50,000 | ชื่อ checkpoint ที่ปล่อยมา (`model.ckpt-50000`) |
| activation | ReLU | ชื่อโฟลเดอร์โมเดล `*_neurons_relu` |
| initialization | glorot uniform + bias 0 | ค่า default ของ TF/Keras |
| การเลือกโมเดล | ไม่มี ใช้โมเดลที่ step สุดท้าย | ต้นฉบับเทรนจบที่ 50,000 steps ตรง ๆ |

### 4.2 Zou et al. — สถาปัตยกรรมที่ถอดจากโค้ดต้นฉบับ

อ่านจาก `model_train.py`, `model_predict.py` และ `README.md` ของ repo

```
FASTA (5,000 bp) → KMC3 -k6 -ci1 -cs5000 → ความถี่ canonical 6-mer 2,080 คอลัมน์
                 → StandardScaler
                 → MLP 7 ชั้น: 256 → 256 → 128 → 128 → 32 → 10 → 1 (sigmoid)
                   dropout 0.4 / 0.4 / 0.5 / 0.2 หลังสี่ชั้นแรก, L2 = 1e-4 ที่สี่ชั้นแรก
                 → three-vote ensemble จากโมเดล 3 ตัว
```

| พารามิเตอร์ | ค่าที่ใช้ | ที่มา |
|---|---|---|
| optimizer | Adam, lr = 0.001 | `optimizers.Adam(lr=0.001)` |
| loss | binary cross-entropy | `BinaryCrossentropy()` |
| batch size | 256 | `batchsz = 256` |
| early stopping | `min_delta` = 0.001 | `EarlyStopping(...)` |
| model selection | เก็บ checkpoint ที่ val ดีที่สุด | `ModelCheckpoint(save_best_only=True)` |
| ensemble | 3 โมเดล เสียงข้างมาก | three-vote ในบทความ |

**หมายเหตุเรื่อง 2,080 features:** KMC นับ k-mer แบบ canonical (รวม k-mer กับ reverse complement
เป็นตัวเดียวกัน) จำนวน canonical 6-mer = (4⁶ + 4³)/2 = 2,080 พอดี เราเขียนตัวนับ k-mer เองด้วย NumPy
และ **ตรวจสอบแล้วว่าลำดับคอลัมน์ตรงกับไฟล์ `kmer_order.csv` ของ repo ต้นฉบับทั้ง 2,080 คอลัมน์**

### 4.3 PlasClass — สถาปัตยกรรมที่ถอดจากโค้ดต้นฉบับ

อ่านจาก `plasclass/train.py`, `plasclass/plasclass.py` และ `plasclass/plasclass_utils.py`

```
FASTA → นับ canonical k-mer ของ k = 3, 4, 5, 6, 7 แยกกัน
      → นอร์มัลไลซ์ "แยกในแต่ละ k" ให้ผลรวมของบล็อกนั้นเป็น 1 แล้วต่อกัน
        (32 + 136 + 512 + 2,080 + 8,192 = 10,952 features)
      → StandardScaler
      → LogisticRegression (L2, C = 1.0)
      → เลือกโมเดลตามช่วงความยาวของลำดับ
```

| พารามิเตอร์ | ค่าที่ใช้ | ที่มา |
|---|---|---|
| k ที่ใช้ | 3, 4, 5, 6, 7 | `--kmers` default `'3,4,5,6,7'` |
| การนับ | canonical (รวม k-mer กับ reverse complement) ข้าม k-mer ที่มีเบสนอก ACGT | `compute_kmer_inds` + `count_kmers` |
| การนอร์มัลไลซ์ | หารด้วยผลรวมของ **แต่ละ k แยกกัน** | `kmer_counts[k] / counts_sum` |
| scaler | `StandardScaler` fit บนข้อมูลเทรน | `StandardScaler().fit(data)` |
| classifier | `LogisticRegression` L2, C = 1.0 | `LogisticRegression(solver='liblinear')` |
| ช่วงความยาว | 1000 / 10000 / 100000 / 500000 | `--lengths` default |
| ข้อมูลเทรน | สุ่ม fragment จำนวนเท่ากันทั้งสองคลาส (coverage 5×, สูงสุด 90,000 ต่อคลาส) | `get_num_frags`, `get_start_inds` |

**ประเด็นสำคัญเรื่องช่วงความยาว:** PlasClass เทรนโมเดลแยกกันสี่ตัวและเลือกใช้ตามความยาวของลำดับ
ด้วยฟังก์ชัน `_get_scale` ซึ่งเทียบกับ *จุดกึ่งกลาง* ระหว่างช่วง สำหรับ fragment 5,000 bp ของเรา

```
5000 > 1000  →  ตรวจต่อ
5000 ≤ (1000 + 10000)/2 = 5500  →  ใช้โมเดลช่วง "1000"
```

ดังนั้นในทางปฏิบัติมีเพียงโมเดลเดียวที่ถูกใช้กับชุดข้อมูลนี้ เราจึงเทรนโมเดลเดียว
(และเทรนมันด้วย fragment 5,000 bp จริง ซึ่งตรงกับข้อมูลมากกว่าโมเดลสำเร็จรูปของ PlasClass
ที่เทรนช่วงนี้ด้วย fragment 1,000 bp)

### 4.4 การตรวจสอบความถูกต้องของ feature pipeline

`verify_features.py` ตรวจก่อนเทรนทุกครั้ง และผ่านทั้งหมด:

- การนับ k-mer (k = 3, 5, 6, 7 และ canonical k = 4, 6) ตรงกับการนับแบบ brute force ทุกช่อง
- หน้าต่างที่มีเบสไม่ใช่ ACGT ถูกข้าม เหมือนพฤติกรรมของทั้ง Biostrings และ KMC
- ลำดับคอลัมน์ canonical 6-mer ตรงกับ `kmer_order.csv` ของ KMC ทุกคอลัมน์
- TF-IDF ให้ผลเท่ากับ `sklearn.feature_extraction.text.TfidfTransformer(smooth_idf=False)`
- การ standardize ให้ผลเท่ากับ `sklearn.preprocessing.StandardScaler`

### 4.5 รายการความแตกต่างจากต้นฉบับ (deviations) — ต้องระบุในรายงานวิจัย

**PlasFlow**

| หัวข้อ | ต้นฉบับ | ของเรา | เหตุผล |
|---|---|---|---|
| จำนวนคลาสผลลัพธ์ | 26 คลาส (taxon × {plasmid, chromosome}) | 2 คลาส | dataset ของเราไม่มี label ระดับ taxon งานที่วัดคือ binary |
| ความยาว fragment ตอนเทรน | 10 kb | 5 kb | เป็นหน่วยข้อมูลของ dataset เรา |
| การ fit TF-IDF | `fit_transform` ใหม่ทุกชุดข้อมูล รวมตอน inference | fit บน train แล้ว transform val/test | กันไม่ให้สถิติของ test รั่วเข้าสู่ feature |
| framework | TensorFlow 1.x (`tf.contrib.learn`) | PyTorch 2.11 | `tf.contrib` ถูกถอดออกตั้งแต่ TF 2 — จำลอง default เดิม (glorot init, Adagrad accumulator 0.1) |
| ข้อมูลเทรน | RefSeq 9,565 replicons | train split ของเรา | จุดประสงค์คือเทียบบนข้อมูลเดียวกัน |

**Zou et al.**

| หัวข้อ | ต้นฉบับ | ของเรา | เหตุผล |
|---|---|---|---|
| การแบ่ง train/val/test | สุ่ม 7:1:2 (repo) / 10-fold CV (บทความ) | cluster-level split 70/15/15 ของ dataset เรา | การสุ่มแบ่งระดับ fragment ทำให้ fragment จาก replicon เดียวกันข้าม split = leakage |
| ตัวนับ k-mer | KMC3 | NumPy counter (ตรวจแล้วว่าลำดับคอลัมน์เหมือนกัน) | ไม่ต้องติดตั้ง binary เพิ่ม และทำซ้ำได้ทุกแพลตฟอร์ม |
| fit StandardScaler | fit บนข้อมูลทั้งหมด (train+val+test) | fit บน train เท่านั้น | ต้นฉบับมี leakage เล็กน้อยในขั้นนี้ |
| จำนวน epoch สูงสุด | 400 (patience 80) | ระบอบ A: 400 / 80 ตามเดิม, ระบอบ B: 15 / 4 | ปรับให้จำนวน optimization step ใกล้เคียงกัน (ดูด้านล่าง) |
| ตัวชี้วัดสำหรับ early stopping | `val_accuracy` | ระบอบ A: `val_accuracy` ตามเดิม, ระบอบ B: val AUPRC | ที่ prevalence 2.3% การทายว่า chromosome ทั้งหมดได้ accuracy 97.66% ทำให้ val_accuracy ไร้ความหมาย |
| framework | TensorFlow/Keras 2.0 | PyTorch 2.11 | ใช้ glorot init และตั้ง `weight_decay = 2λ` เพื่อให้ gradient ของ L2 ตรงกับ `kernel_regularizer=l2(λ)` ของ Keras |

**PlasClass**

| หัวข้อ | ต้นฉบับ | ของเรา | เหตุผล |
|---|---|---|---|
| solver ของ logistic regression | `liblinear` | `lbfgs` (บน float32) | `liblinear` ของ scikit-learn บังคับแปลงข้อมูลเป็น float64 → เมทริกซ์ 15,830 × 10,952 กลายเป็น 1.39 GB ซึ่งเกิน RAM ว่างของเครื่อง ทั้งสอง solver ลดค่า objective เดียวกัน (L2 logistic, C = 1.0) — **ตรวจสอบแล้ว** (ดูด้านล่าง) |
| `max_iter` | 100 (ค่า default) | 1,000 | ให้ lbfgs ลู่เข้าจริง ๆ ผลคือหยุดเองที่ 290 iterations |
| การสร้าง fragment สำหรับเทรน | สุ่มตำแหน่งเริ่มจาก replicon อ้างอิง coverage 5× (ทับซ้อนกันได้) | ใช้ fragment 5,000 bp แบบ non-overlapping ที่มีอยู่แล้วใน train split | dataset ของเราถูกออกแบบให้ไม่ทับซ้อนโดยตั้งใจ การสุ่มซ้ำตำแหน่งจะสร้างข้อมูลที่สหสัมพันธ์กันเอง |
| จำนวน fragment ต่อคลาส | สูงสุด 90,000 | 7,915 (เท่าที่มีจริงในคลาส plasmid) | จำนวน plasmid fragment ที่ไม่ทับซ้อนมีอยู่เท่านี้ |
| โมเดลตามช่วงความยาว | 4 โมเดล | 1 โมเดล (ช่วง "1000") | fragment ทุกชิ้นยาว 5,000 bp จึงถูก route ไปโมเดลเดียวกันหมด |
| ระบอบการเทรน | สมดุลเสมอ | ระบอบ A เท่านั้น | เป็นวิธีของมันเอง + ข้อจำกัดหน่วยความจำ (ดูหัวข้อ 2.1) |
| ตัวนับ k-mer | state machine แบบเลื่อนทีละเบส | ตัวนับ vectorised เดียวกับที่ใช้กับ Zou | ตรวจสอบแล้วว่าตรงกับการนับแบบ brute force (หัวข้อ 4.4) ส่วนพฤติกรรมตอนเจอเบสนอก ACGT ต่างกันเล็กน้อย แต่ทั้ง dataset มีเบสแบบนั้นเพียง 13 ตำแหน่ง |

**การตรวจสอบว่า `lbfgs` แทน `liblinear` ได้จริง:** `run_plasclass.py` เทรนทั้งสอง solver
บนชุดย่อย 3,000 แถวด้วยข้อมูลเดียวกันแล้วเทียบผล

| ตัวชี้วัด | ค่า |
|---|---|
| ผลต่างความน่าจะเป็นสูงสุด | 0.0111 |
| ผลต่างความน่าจะเป็นเฉลี่ย | 0.00059 |
| การตัดสินใจที่ threshold 0.5 ตรงกัน | **100.00%** |
| สหสัมพันธ์ของสัมประสิทธิ์ (Pearson r) | 0.966 |

การตัดสินใจตรงกันทุกตัวอย่าง จึงสรุปได้ว่าการเปลี่ยน solver ไม่กระทบข้อสรุป
(ค่าสัมประสิทธิ์ต่างกันบ้างเป็นเรื่องปกติเมื่อ feature 10,952 ตัวมีสหสัมพันธ์กันสูง
ทิศทางของ decision boundary ที่ได้จึงเหมือนกันแม้ค่าน้ำหนักรายตัวจะกระจายต่างกัน)

**เรื่องงบ optimization step ของระบอบ B:** ต้นฉบับเทรนข้อมูลราว 15,000 ตัวอย่าง batch 256 → ~62 step/epoch
× 400 epoch ≈ 24,800 steps ส่วนระบอบ B ของเรามี 337,512 ตัวอย่าง → 1,319 step/epoch หากเทรน 400 epoch
จะเท่ากับ 527,600 steps (มากกว่าต้นฉบับ 20 เท่า) จึงจำกัดที่ 15 epoch ≈ 19,800 steps ซึ่งเทียบเคียงกับ
งบเดิม และยังมี early stopping ทำงานอยู่

### 4.6 ข้อจำกัดด้านทรัพยากรและวิธีจัดการ

เครื่องที่รันมี RAM 8 GB และไม่มี GPU แต่ feature matrix ของ 7-mer มีขนาด 337,512 × 16,384 (11 GB)
จึงออกแบบให้ไม่มีการโหลดข้อมูลทั้งก้อนเข้าหน่วยความจำเลย:

- เก็บ k-mer counts เป็น `uint16` memmap บนดิสก์ (รวมทุกชุดฟีเจอร์ ~25 GB, สร้างใหม่ได้เสมอ)
- **เขียนแถวของ train ในลำดับที่สุ่มไว้ล่วงหน้า** แล้วตอนเทรนอ่านทีละ block 64 MB ตามลำดับที่สับ block
  และสับแถวภายใน block อีกชั้น — ได้ผลใกล้เคียงการสุ่มเต็มรูปแบบโดยที่การอ่านดิสก์ยังเป็นแบบ sequential
- TF-IDF / StandardScaler คำนวณสถิติแบบ streaming แล้ว transform ทีละ minibatch

---

## 5. ผลการทดลอง

ตารางเต็มทุกชุดอยู่ใน `results/summary.md` ส่วนด้านล่างคือตารางหลักและการตีความ
**ทุกตัวเลขวัดบน test split เต็ม 72,332 fragments (plasmid 1,696 = 2.345%) ที่ไม่เคยถูกใช้เทรน**

### 5.1 ภาพรวม (ค่า threshold มาตรฐาน 0.5)

| โมเดล | accuracy | balanced acc | precision | recall | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| ทายว่า chromosome ทั้งหมด | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0234 |
| PlasFlow — ระบอบ A (balanced) | 0.8105 | 0.7910 | 0.0894 | 0.7706 | 0.1602 | 0.2193 | 0.8710 | 0.1565 |
| PlasFlow — ระบอบ B (full) | 0.9766 | 0.5026 | 0.7500 | 0.0053 | 0.0105 | 0.0618 | 0.8881 | 0.2110 |
| Zou et al. — ระบอบ A (balanced) | 0.8164 | 0.8090 | 0.0950 | 0.8013 | 0.1699 | 0.2348 | 0.8830 | 0.1889 |
| Zou et al. — ระบอบ B (full) | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.9092** | **0.2733** |
| PlasClass — ระบอบ A (balanced) | 0.7075 | 0.6995 | 0.0537 | 0.6910 | 0.0997 | 0.1315 | 0.7594 | 0.0672 |

อ่านตารางนี้ต้องระวังสองอย่าง:

1. **accuracy ใช้ไม่ได้กับงานนี้** — การทายว่าเป็น chromosome ทั้งหมดได้ accuracy 97.66% ดังนั้น
   accuracy 97.66% ของระบอบ B ไม่ได้แปลว่าโมเดลเก่ง แต่แปลว่ามันไม่เคยทายว่า plasmid เลย
   (confusion matrix ของ Zou ระบอบ B คือ TP = 0, FP = 0 พอดี) ให้ใช้ **MCC, F1, AUPRC** แทน
2. **ระบอบ B ไม่ได้แย่กว่าระบอบ A** — ถึง F1 ที่ threshold 0.5 จะเป็น 0 แต่ AUROC 0.89–0.91 และ
   AUPRC 0.21–0.27 (สูงกว่าค่าสุ่ม 0.0234 ประมาณ 9–12 เท่า) แปลว่าโมเดล **จัดอันดับ** ได้ดีกว่าระบอบ A
   ด้วยซ้ำ ปัญหาอยู่ที่ **จุดตัดการตัดสินใจ** ล้วน ๆ ไม่ใช่ที่ความสามารถในการแยก

### 5.2 ผลของการปรับ threshold — ประเด็นที่สำคัญที่สุด

เลือก threshold ที่ให้ F1 สูงสุด **บน validation split** แล้วนำไปใช้กับ test (ไม่ได้จูนบน test)

| โมเดล | threshold | accuracy | precision | recall | F1 | MCC |
|---|---|---|---|---|---|---|
| PlasFlow A | 0.5 → **0.966** | 0.9433 | 0.1812 | 0.4033 | 0.2501 | 0.2447 |
| PlasFlow B | 0.5 → **0.092** | 0.9456 | 0.2020 | 0.4481 | 0.2785 | 0.2764 |
| Zou A | 0.5 → **0.991** | 0.9336 | 0.1773 | 0.5035 | 0.2622 | 0.2716 |
| **Zou B** | 0.5 → **0.161** | 0.9672 | 0.3272 | 0.3768 | **0.3502** | **0.3344** |
| PlasClass A | 0.5 → 1.000 | 0.8881 | 0.0776 | 0.3467 | 0.1269 | 0.1225 |

การปรับ threshold อย่างเดียวทำให้ Zou ระบอบ B กระโดดจาก F1 = 0.000 เป็น **0.350** และ MCC = 0.334
ซึ่งเป็นผลที่ดีที่สุดในการทดลองทั้งหมด → **ค่าที่ควรใช้เป็นเส้นฐานให้โมเดลใหม่เอาชนะคือ
AUPRC 0.273 และ F1 0.350 / MCC 0.334 ที่ threshold ปรับจาก validation**

PlasClass เป็นข้อยกเว้นที่น่าสนใจ: threshold ที่ให้ F1 สูงสุดบน validation ถูกดันไปถึง 1.000
(ปัดจาก 0.99997) และถึงอย่างนั้น F1 ก็ยังขึ้นได้แค่ 0.127 สะท้อนว่า logistic regression
ที่มี 10,952 features บนข้อมูลเทรน 15,830 แถวให้ความน่าจะเป็นที่ "มั่นใจเกินจริง"
(อิ่มตัวใกล้ 0 หรือ 1) จนการเลื่อนจุดตัดช่วยได้น้อย

### 5.3 เทียบกับตัวเลขที่เปเปอร์รายงาน

เปเปอร์ทั้งสองรายงาน accuracy บนชุดทดสอบที่สมดุล จึงต้องเทียบบน **test subsample แบบ 1:1**
(1,696 plasmid + 1,696 chromosome) เท่านั้นถึงจะเทียบได้อย่างยุติธรรม

| โมเดล | accuracy (test สมดุล) | MCC | AUROC | เปเปอร์รายงานไว้ |
|---|---|---|---|---|
| PlasFlow — ระบอบ A | 0.7883 | 0.5770 | 0.8741 | ~96% |
| Zou et al. — ระบอบ A | 0.8072 | 0.6144 | 0.8874 | 89% เดี่ยว / 92% ensemble |
| PlasClass — ระบอบ A | 0.6940 | 0.3880 | 0.7543 | บทความรายงานว่าดีกว่า PlasFlow และ cBar (ไม่ได้เทียบเป็นตัวเลข accuracy ตรง ๆ) |
| PlasFlow — ระบอบ B | 0.5027 | 0.0516 | 0.8908 | — |
| Zou et al. — ระบอบ B | 0.5000 | 0.0000 | **0.9136** | — |

ผลของเราต่ำกว่าที่เปเปอร์รายงานราว **10–20 จุด** ซึ่งอธิบายได้ด้วยความต่างของการออกแบบการทดลอง
ไม่ใช่ข้อบกพร่องของการ implement (ดูหัวข้อ 6.1)

### 5.4 ผลของขนาด k และ ensemble

AUPRC ของสมาชิกแต่ละตัวบน test เต็ม:

| สมาชิก | ระบอบ A | ระบอบ B |
|---|---|---|
| PlasFlow k=5 (1,024 features) | 0.1236 | 0.1911 |
| PlasFlow k=6 (4,096 features) | 0.1334 | 0.2098 |
| PlasFlow k=7 (16,384 features) | 0.1058 | 0.1737 |
| **PlasFlow ensemble (เฉลี่ยความน่าจะเป็น)** | **0.1565** | **0.2110** |
| Zou โมเดล 1 / 2 / 3 | 0.1720 / 0.1703 / 0.1730 | 0.2194 / 0.2565 / 0.2898 |
| **Zou ensemble (เฉลี่ยความน่าจะเป็น)** | **0.1889** | 0.2733 |

- **k = 7 แย่ที่สุดเสมอ** เหตุผลเชิงปริมาณ: fragment 5,000 bp มีหน้าต่าง 7-mer เพียง 4,994 หน้าต่าง
  แต่มี feature 16,384 ช่อง → เฉลี่ยแล้วแต่ละช่องมีค่าเพียง 0.30 สเปกตรัม 7-mer ของ fragment ขนาดนี้
  จึงเป็นศูนย์เกือบทั้งหมดและมี sampling noise สูงมาก (PlasFlow ต้นฉบับเทรนบน fragment 10 kb
  ซึ่งมีหน้าต่างมากกว่าเท่าตัว) → **ถ้าจะใช้ k-mer กับ fragment 5 kb ควรอยู่ที่ k = 5–6**
- **ensemble ช่วยจริงใน PlasFlow** (0.1565 สูงกว่าสมาชิกที่ดีที่สุด 0.1334) เพราะสมาชิกใช้ k ต่างกัน
  จึงมีความผิดพลาดที่ไม่สัมพันธ์กัน ส่วน Zou ที่ต่างกันแค่ random seed ได้ผลน้อยกว่า
  (0.2733 ยังต่ำกว่าสมาชิกที่ดีที่สุด 0.2898) และ **ความแปรปรวนระหว่าง seed สูงมาก (0.219–0.290)**
  ซึ่งเป็นข้อควรระวังเวลารายงานผลจากการรันครั้งเดียว
- **PlasClass ใช้ k ทั้งห้าค่าพร้อมกันแต่กลับได้ผลต่ำที่สุด** (AUPRC 0.0672) ทั้งที่มี feature มากที่สุด
  (10,952 ช่อง) ความต่างสำคัญคือมันเป็นโมเดล **เชิงเส้น** ขณะที่อีกสองตัวเป็นโครงข่ายประสาทที่มีชั้นซ่อน
  → บนงานข้ามสปีชีส์แบบนี้ ขอบเขตการตัดสินใจเชิงเส้นบนความถี่ k-mer ดูจะไม่เพียงพอ
  และการเพิ่มจำนวน feature ไม่ได้ชดเชยข้อจำกัดนั้น

### 5.5 กฎ threshold 0.7 ของ PlasFlow (ตอบว่า "unclassified")

| ระบอบ | จัดประเภทได้ | unclassified | unclassified เฉพาะ plasmid | accuracy เฉพาะที่จัดได้ | recall เฉพาะที่จัดได้ |
|---|---|---|---|---|---|
| A (balanced) | 56,518 | 21.86% | 26.65% | 0.8663 | 0.8449 |
| B (full) | 72,068 | 0.36% | 6.49% | 0.9780 | 0.0000 |

กฎนี้ช่วยได้เฉพาะระบอบ A (ตัด fragment ที่ไม่มั่นใจออก 21.9% แล้ว accuracy ขึ้นจาก 0.81 เป็น 0.87)
แต่ในระบอบ B แทบไม่ทำงาน เพราะโมเดล "มั่นใจว่าเป็น chromosome" เกือบทุกชิ้นอยู่แล้ว

### 5.6 ผลระดับ replicon (เฉลี่ยความน่าจะเป็นของทุก fragment ในหนึ่ง replicon)

test split มี 126 replicons (chromosome 77 / plasmid 49)

| โมเดล | accuracy | precision | recall | F1 | MCC | AUROC |
|---|---|---|---|---|---|---|
| PlasFlow — A | 0.8651 | 0.8333 | 0.8163 | 0.8247 | 0.7152 | 0.9247 |
| Zou et al. — A | **0.8889** | 0.8723 | 0.8367 | **0.8542** | **0.7649** | 0.9231 |
| PlasClass — A | 0.8016 | 0.7400 | 0.7551 | 0.7475 | 0.5842 | 0.8712 |
| PlasFlow — B | 0.6111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9009 |
| Zou et al. — B | 0.6111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8728 |

**ข้อค้นพบสำคัญ:** เมื่อรวมผลทั้ง replicon แล้ว MCC พุ่งจาก 0.23 (ระดับ fragment) เป็น 0.76
สัญญาณระดับ k-mer ของ fragment เดี่ยว 5 kb นั้นอ่อน แต่พอเฉลี่ยข้ามหลายสิบ–หลายร้อย fragment
ของ replicon เดียวกัน noise หักล้างกันจนแยกได้ดี → ถ้าเป้าหมายปลายทางคือจำแนก contig/replicon
ตัวเลขระดับ fragment จะประเมินความมีประโยชน์ของโมเดลต่ำกว่าความเป็นจริง

### 5.7 พฤติกรรมการเทรน

- **PlasFlow ระบอบ A overfit ชัดเจน** — val AUPRC สูงสุดที่ราว step 5,000–15,000 แล้วค่อย ๆ ลดลง
  ขณะที่ training loss ลงไปถึง 0.0015 ที่ step 50,000 เนื่องจากต้นฉบับไม่มี early stopping
  (เทรนครบ 50,000 steps แล้วใช้โมเดลสุดท้าย) เราจึงทำตามอย่างซื่อตรง แต่หมายความว่า
  **ตัวเลขของ PlasFlow ในรายงานนี้มาจาก checkpoint ที่ overfit แล้ว** การเพิ่ม early stopping
  น่าจะดันผลขึ้นได้อีกเล็กน้อย (เป็นการแก้ที่บรรทัดเดียวใน `run_plasflow.py`)
- **Zou ระบอบ B early stop ที่ epoch 7–9 (ดีที่สุดที่ epoch 3–5)** ซึ่งต่ำกว่าเพดาน 15 epoch ที่เราตั้งไว้มาก
  → การลดงบ epoch ตามหัวข้อ 4.5 **ไม่ใช่ปัจจัยจำกัดผล** ข้อกังวลเรื่อง deviation ข้อนี้จึงตกไป
- **PlasClass ลู่เข้าเองที่ 290 iterations** (ต่ำกว่าเพดาน 1,000 ที่ตั้งไว้) ใช้เวลาเทรนเพียง 58 วินาที
  ถูกที่สุดในบรรดา baseline ทั้งหมด แต่ก็ให้ผลต่ำที่สุดเช่นกัน
- ต้นทุนรวมทั้งหมด (12 โมเดล) ประมาณ **55 นาที** บน CPU 8 คอร์ ไม่ต้องใช้ GPU

> หมายเหตุการอ่าน log: ค่า `val_auprc` ระหว่างเทรนของระบอบ A วัดบน validation แบบสมดุล
> (prevalence 50% จึงอยู่ราว 0.78) ส่วนระบอบ B วัดบน validation จริง (prevalence 2.3% จึงอยู่ราว 0.10)
> ตัวเลขสองชุดนี้ใช้เฝ้าดูการเทรนเท่านั้น **เทียบกันข้ามระบอบไม่ได้** ค่าที่ใช้สรุปผลคือค่าบน test เต็ม
> ในหัวข้อ 5.1–5.6

| การรัน | fragments ที่เทรน | สมาชิก | เวลารวม (วินาที) |
|---|---|---|---|
| PlasFlow — A | 15,830 | 3 | 556 |
| PlasFlow — B | 337,512 | 3 | 1,094 |
| Zou — A | 15,830 | 3 | 705 |
| Zou — B | 337,512 | 3 | 908 |
| PlasClass — A | 15,830 | 1 | 58 |

---

## 6. อภิปราย

### 6.1 ทำไมผลถึงต่ำกว่าที่เปเปอร์รายงาน

ไม่ใช่เพราะ implement ผิด (feature pipeline ผ่านการตรวจเทียบ brute force และ sklearn ทุกข้อ)
แต่เพราะ **ชุดทดสอบของเรายากกว่าอย่างมีนัยสำคัญ** ด้วยเหตุผลที่ออกแบบไว้ตั้งแต่ต้น:

1. **การแบ่ง split ระดับ cluster** — dataset ของเราจับ replicon ที่คล้ายกัน (ANI ≥ 0.95 หรือ containment
   ≥ 0.90) เข้า cluster เดียวกันก่อนแบ่ง ทำให้ **ไม่มี** replicon ใน test ที่มีญาติเกือบเหมือนกันอยู่ใน train
   ขณะที่โปรโตคอลของทั้งสองเปเปอร์สุ่มแบ่งที่ระดับลำดับ/fragment ซึ่ง plasmid ที่แทบเหมือนกัน
   (เช่น plasmid ดื้อยาตัวเดียวกันที่พบในหลายสายพันธุ์) สามารถอยู่ทั้งใน train และ test ได้
2. **ความหลากหลายทางอนุกรมวิธานสูงกว่ามาก** — dataset ของเราสุ่มแบบ 1 assembly ต่อ 1 สปีชีส์
   จาก 500 สปีชีส์ที่ต่างกัน ส่วนชุดข้อมูลของทั้งสองเปเปอร์กระจุกอยู่ที่สปีชีส์ทางคลินิกไม่กี่ตัว
   งานของเราจึงเป็น cross-species generalization ที่แท้จริง
3. **fragment 5 kb แทน 10 kb** (PlasFlow) ทำให้สเปกตรัม k-mer มี sampling noise สูงขึ้น โดยเฉพาะ k = 7
4. **PlasFlow ต้นฉบับมี 26 คลาสที่ผูกกับ taxon** ซึ่งให้สัญญาณเสริมเรื่ององค์ประกอบเบสของแต่ละกลุ่ม
   ส่วนเวอร์ชันไบนารีของเราไม่มีข้อมูลส่วนนั้น
5. **เฉพาะ PlasClass:** โมเดลสำเร็จรูปของมันเทรนด้วย fragment สูงสุด 90,000 ชิ้นต่อคลาส
   ขณะที่เรามี plasmid fragment แบบไม่ทับซ้อนเพียง 7,915 ชิ้น — น้อยกว่าราว 11 เท่า
   สำหรับโมเดลที่มี 10,952 พารามิเตอร์ ข้อจำกัดนี้น่าจะมีผลมากเป็นพิเศษ

ข้อ 1 และ 2 คือสิ่งที่ตั้งใจให้เกิดขึ้นใน dataset — **ตัวเลขที่ต่ำลงคือราคาของการวัดผลที่ซื่อสัตย์ขึ้น**
ไม่ใช่ข้อเสีย และเป็นเหตุผลว่าทำไมจึงห้ามนำตัวเลข 96% จากบทความ PlasFlow มาเทียบกับโมเดลของเราตรง ๆ

### 6.2 ข้อสรุปที่นำไปใช้ออกแบบโมเดลของเราได้

1. **ต้องรายงาน MCC / F1 / AUPRC ไม่ใช่ accuracy** และต้องแสดงเส้นฐาน "ทายว่า chromosome ทั้งหมด"
   (accuracy 97.66%, AUPRC 0.0234) กำกับไว้เสมอ
2. **class imbalance คือปัญหาเรื่องจุดตัด ไม่ใช่เรื่องการเรียนรู้** — โมเดลที่เทรนบนข้อมูลไม่สมดุล
   เรียนรู้ได้ดีกว่า (AUPRC 0.27 เทียบกับ 0.19) แต่ threshold 0.5 ใช้ไม่ได้ จึงควรเทรนบนข้อมูลเต็ม
   แล้วจัดการที่ threshold / class weight / focal loss แทนการทิ้งข้อมูล chromosome ไป 95%
3. **การเลือก threshold ต้องทำบน validation เท่านั้น** และต้องระบุในรายงาน เพราะมันเปลี่ยนผลจาก
   F1 = 0.000 เป็น 0.350 ได้ ถ้าไม่ระบุโปรโตคอลนี้ ตัวเลขจะเทียบกันไม่ได้เลย
4. **k = 5–6 เหมาะกับ fragment 5 kb มากกว่า k = 7** เพราะจำนวนหน้าต่างต่อ fragment เป็นข้อจำกัดจริง
5. **ตั้งเป้าที่ตัวเลขนี้:** AUPRC > 0.273, MCC > 0.334 ที่ระดับ fragment และ MCC > 0.765 ที่ระดับ replicon
6. **รันหลาย seed** เพราะ AUPRC ของ Zou แกว่ง 0.219–0.290 ระหว่าง seed การรายงานจากรันเดียวเสี่ยงเข้าใจผิด
7. **ความไม่เป็นเชิงเส้นสำคัญกว่าจำนวน feature** — PlasClass ใช้ feature มากที่สุด (10,952 ช่อง
   ครอบคลุม k = 3–7) แต่เป็นโมเดลเชิงเส้น จึงได้ AUPRC ต่ำที่สุด (0.067) ขณะที่ Zou ใช้ feature
   เพียง 2,080 ช่องแต่มี MLP 7 ชั้น ได้ AUPRC สูงสุด (0.273) → การลงทุนกับสถาปัตยกรรมให้ผลตอบแทน
   มากกว่าการเพิ่มมิติของ feature บนงานนี้

### 6.3 ข้อจำกัดของ baseline ชุดนี้

- ตัวเลขทั้งหมดคือ *สถาปัตยกรรมของเปเปอร์เมื่อเทรนบนข้อมูลของเรา* ไม่ใช่ประสิทธิภาพของเครื่องมือ
  ตามที่ผู้เขียนเผยแพร่ ห้ามนำไปอ้างสลับกัน
- **PlasClass ไม่มีผลในระบอบ B** ด้วยเหตุผลทางวิธีการและข้อจำกัดหน่วยความจำ (หัวข้อ 2.1)
  จึงยังตอบไม่ได้ว่าการเทรนบนข้อมูลไม่สมดุลจะช่วย PlasClass ได้เหมือนที่ช่วย Zou หรือไม่
  ถ้าต้องการคำตอบต้องใช้เครื่องที่มี RAM ≥ 32 GB หรือเปลี่ยนไปใช้ `SGDClassifier` แบบ streaming
  ซึ่งจะกลายเป็นคนละ estimator กับต้นฉบับ
- dataset ปัจจุบันมาจาก 500 assemblies (species-balanced) ถ้าเพิ่มจำนวน assembly ผลของ baseline
  จะเปลี่ยน ต้องรันใหม่ทั้งหมดก่อนเปรียบเทียบ
- fragment เป็นชิ้นที่ตัดจาก genome สมบูรณ์ ไม่ใช่ contig จริงจากการประกอบ จึงไม่มี assembly error
  ทั้งสองเปเปอร์ประเมินบน contig จริง/จำลอง ผลจึงไม่เทียบกันตรง ๆ ในมิตินี้เช่นกัน
- ยังไม่ได้รัน Deeplasmid / PLASMe / PlasmidHunter (ต้องมี Pfam, BLAST หรือ DIAMOND + ฐานข้อมูลโปรตีน
  และมีความเสี่ยงเรื่องการปนเปื้อนของชุดทดสอบตามที่อภิปรายในหัวข้อ 1)
- ไม่ได้รันโมเดลสำเร็จรูป `5k6mer_model.h5` ของ Zou เทียบด้วย เพราะต้องใช้ TensorFlow และผลจะ
  ตีความไม่ได้เนื่องจากปัญหาการปนเปื้อน — ถ้าต้องการทำจริงต้องพิสูจน์ก่อนว่า replicon ใน test ของเรา
  ไม่อยู่ในชุดเทรนของเขา (ไฟล์ `plasmid_id_5k` / `chromosome_id_5k` ใน repo มีรายการ accession ให้ตรวจได้)

---

## 7. วิธีทำซ้ำ

```bash
cd baseline
py -3.13 verify_features.py     # ตรวจความถูกต้องของตัวนับ k-mer และ transform
py -3.13 prepare_features.py    # encode FASTA + สร้าง k-mer matrix ทั้งหมด
py -3.13 run_plasflow.py        # 3 โมเดล × 2 ระบอบ
py -3.13 run_zou.py             # 3 โมเดล × 2 ระบอบ
py -3.13 run_plasclass.py       # logistic regression 1 โมเดล (ระบอบ A เท่านั้น)
py -3.13 summarize.py           # สร้าง results/summary.md
py -3.13 capture_environment.py # บันทึก results/environment.json
```

ทุกพารามิเตอร์อยู่ใน `config.yaml`, seed = 42, ผลลัพธ์ดิบอยู่ใน `results/*.json`
และความน่าจะเป็นราย fragment อยู่ใน `results/probs/*.npy`

**ก่อนรันซ้ำ ให้ตรวจสอบหัวข้อ 3 ทั้งหมด** โดยเฉพาะ checksum ของชุดข้อมูล (3.5)
และขอบเขตของการทำซ้ำได้ (3.9) — ถ้าสภาพแวดล้อมต่างจากที่ระบุไว้ ต้องรายงานว่าต่างตรงไหน
พร้อมแนบ `results/environment.json` ของการรันใหม่ไปกับผลลัพธ์เสมอ
ต้องมีพื้นที่ว่างบนดิสก์อย่างน้อย 30 GB และใช้เวลาราว 70 นาทีบนเครื่องระดับเดียวกัน

---

## 8. อ้างอิง

1. Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic
   data using genome signatures. *Nucleic Acids Research*. 2018;46(6):e35.
   โค้ด: <https://github.com/smaegol/PlasFlow>
2. Zou X, Nguyen M, Overbeek J, Cao B, Davis JJ. Classification of bacterial plasmid and chromosome
   derived sequences using machine learning. *PLoS ONE*. 2022;17(12):e0279280.
   โค้ด: <https://github.com/BV-BRC-dependencies/zou-plasmid-prediction>
3. Pellow D, Zorea A, Probst M, Furman O, Segal A, Mizrahi I, Shamir R. Plasmid classification
   in metagenomic datasets using an improved machine learning approach. *PLoS Computational
   Biology*. 2020;16(10):e1007781. โค้ด: <https://github.com/Shamir-Lab/PlasClass>
4. Kokot M, Dlugosz M, Deorowicz S. KMC 3: counting and manipulating k-mer statistics.
   *Bioinformatics*. 2017;33(17):2759–2761. (ตัวนับ k-mer ที่ Zou et al. ใช้ และเป็นที่มาของ
   ลำดับคอลัมน์ canonical 6-mer จำนวน 2,080 ช่อง)
5. เอกสารประกอบชุดข้อมูล: `../dataset pipeline1/README.md` และ `../dataset pipeline1/SUMMARY.md`
6. รายงานของโมเดลที่เสนอเอง (ESM-2 + attention): `../esm2-attention/REPORT.md`
