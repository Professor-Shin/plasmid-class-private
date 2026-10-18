# Platon และ MOB-suite เป็น baseline

เพิ่มเครื่องมือที่ตีพิมพ์แล้ว 2 ตัวเข้าไปในตารางเปรียบเทียบ โดย **ไม่เทรนใหม่** เพราะทั้งคู่เป็น
database/rule-based ไม่ใช่ตัวจำแนกที่เรียนรู้ได้ โค้ดในโฟลเดอร์นี้จึงเป็น "โครงรอบ" เครื่องมือ:
แบ่งข้อมูลให้เป็นรูปแบบที่เครื่องมือออกแบบมารับ, อ่าน output โดยไม่เดาตำแหน่งคอลัมน์,
และคำนวณกฎการตัดสินที่เอกสารระบุไว้

| เครื่องมือ | อ้างอิง | ติดตั้ง |
|---|---|---|
| Platon | Schwengers et al. 2020, *Microb Genom*, doi:10.1099/mgen.0.000398 | `conda install -c conda-forge -c bioconda platon` + DB ~2.8 GB |
| MOB-suite (`mob_recon`) | Robertson & Nash 2018, *Microb Genom*, doi:10.1099/mgen.0.000206 | `conda install -c conda-forge -c bioconda mob_suite` + `mob_init` |

## สองการตัดสินใจที่สำคัญ

**1. 1 shard = 1 assembly ไม่ใช่ 1 fragment** — ทั้งสองเครื่องมือออกแบบมารับ "contigs ของ
draft assembly หนึ่งตัว" ไม่ใช่ sequence เดี่ยวที่ไม่เกี่ยวกัน dataset นี้แบ่ง split ที่ระดับ
cluster/replicon และ fragment ของ assembly หนึ่งก็ tile เต็ม replicon แบบไม่ซ้อนกัน →
fragment ทั้งหมดของ assembly หนึ่ง**คือ** draft assembly ที่แตกเป็น contig จริง ๆ
(test split: 16,944 fragments → **116 shards**, เฉลี่ย ~3.1 Mbp ≈ 1 จีโนม/shard)

**2. Platon รัน `--characterize` รอบเดียว** — `.tsv` ปกติเขียนแค่ contig ที่ตัดสินว่าเป็น
plasmid แต่ `--characterize` ปิด filter และรายงาน RDS + hallmark **ทุกคอลัมน์ของทุก contig**
จึงคำนวณกฎทั้ง 3 mode ย้อนหลังได้จากการรันครั้งเดียว และได้ RDS เป็น **continuous score**
ไปคำนวณ AUROC/AUPRC ด้วย `server/13_smoke_tools.sh` ตรวจว่ากฎที่คำนวณย้อนหลังตรงกับ
`platon --mode accuracy` ของจริง 100% ก่อนจะเชื่อ

## ข้อจำกัดเชิงวิธีวิทยาที่ต้องเขียนในรายงาน

fragment เป็น **linear sub-sequence** ของ replicon จึงไม่มี circularity และมักขาด oriT/rep
hallmark ซึ่ง Platon mode `accuracy` พึ่งพา และ mob_recon ก็ตัด contig ที่สั้นกว่า
`--min_length` (1000 bp) ทิ้ง → ทั้งคู่จะเอียงไปทาง chromosome/unclassified

**นี่เป็นผลที่ถูกต้องและน่าสนใจ ไม่ใช่บั๊ก**: เครื่องมือที่สมมติว่า input คือ assembled contig
ทำงานแย่ลงกับ fragment และนั่นคือเหตุผลที่ต้องรายงาน**ระดับ replicon**ควบคู่กับระดับ fragment

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.yaml` | path, split, กฎ mode, นโยบาย unclassified |
| `shard_by_assembly.py` | test split → `shards/<assembly>.fasta` + `manifest.csv` + `fragment_index_test.csv` |
| `make_smoke_fasta.py` | FASTA 20 fragment (plasmid 10 / chromosome 10) สำหรับ smoke test |
| `collect_platon.py` | รวม TSV ทุก shard → `predictions_platon_test.csv` + label ทั้ง 3 mode |
| `collect_mobsuite.py` | รวม `contig_report.txt` → `predictions_mobsuite_test.csv` |
| `score_tools.py` | ให้คะแนนด้วย `baseline/baselines/metrics.py` ตัวเดิม → `tool_baselines.{json,csv}` |
| `verify_platon_modes.py` | พิสูจน์ว่ากฎที่คำนวณย้อนหลัง == `--mode accuracy` ของจริง |
| `toolbase/parse.py` | อ่าน output **ตามชื่อคอลัมน์** + assert คอลัมน์ที่ต้องใช้ |
| `toolbase/rules.py` | กฎ 3 mode ของ Platon + label/score ของ mob_recon |
| `toolbase/cfg.py` | โหลด config + import metric ของ baseline เดิม |

## ลำดับการรัน

บนเซิร์ฟเวอร์ใช้ `server/submit_all.sh tools` (จัดการ job array + dependency ให้) หรือรันมือ:

```bash
python shard_by_assembly.py                    # 116 shards
# ต่อ shard (job array ทำให้): platon --characterize --db $PLATON_DB -o out/platon/<s> -p <s> shards/<s>.fasta
# ต่อ shard (job array ทำให้): mob_recon --infile shards/<s>.fasta --outdir out/mobsuite/<s> --force
python collect_platon.py
python collect_mobsuite.py
python score_tools.py
```

## การให้คะแนน — 3 นโยบายต่อเครื่องมือ

ทั้งคู่อาจไม่ตัดสิน fragment บางตัว จึงรายงาน 3 แบบ (เลียนแบบ
`metrics.plasflow_threshold_rule` ที่จัดการ unclassified band ของ PlasFlow อยู่แล้ว):

| policy | ความหมาย |
|---|---|
| `chromosome` | unclassified → chromosome (conservative, เป็นตัวเลข headline) |
| `plasmid` | unclassified → plasmid (upper bound ของ recall) |
| `exclude` | ตัด unclassified ออก + รายงาน coverage แยก |

สองคอลัมน์ที่ต้องอ่านอย่างระวัง:
- `hard_label_score=True` → AUROC/AUPRC ของแถวนั้นมาจาก**การตัดสิน** ไม่ใช่ความน่าจะเป็น
  (mob_recon ไม่คืนค่าความน่าจะเป็นเลย) ตัวเลขจึงถูกครอบงำโดย label
- `mcc_oracle` → threshold จูนบน **test split เอง** เพราะเราไม่ได้รันเครื่องมือบน validation
  (ไม่ต้องเทรนจึงไม่จำเป็น) ใช้เป็น **upper bound** ไม่ใช่จุดทำงานที่เลือกได้ล่วงหน้า
