# เครื่องมือที่ตีพิมพ์แล้ว 4 ตัวเป็น baseline

เพิ่มเครื่องมือที่ตีพิมพ์แล้ว 4 ตัวเข้าไปในตารางเปรียบเทียบ โดย **ไม่เทรนใหม่** เพราะทั้งหมดเป็น
database/rule-based หรือโมเดลที่เทรนมาแล้ว ไม่ใช่ตัวจำแนกที่เราต้องเรียนรู้เอง โค้ดในโฟลเดอร์นี้
จึงเป็น "โครงรอบ" เครื่องมือ: แบ่งข้อมูลให้เป็นรูปแบบที่เครื่องมือออกแบบมารับ, อ่าน output
โดยไม่เดาตำแหน่งคอลัมน์, และคำนวณกฎการตัดสินที่เอกสาร/ซอร์สระบุไว้

> baseline ตัวที่ 5 ในตารางคือ **Zou et al.** ซึ่งเป็น**สถาปัตยกรรม ไม่ใช่เครื่องมือ** จึงต้อง
> เทรนใหม่บน dataset นี้ และอยู่ที่ `baseline/run_zou_mixed.py` ไม่ใช่ในโฟลเดอร์นี้

| เครื่องมือ | อ้างอิง | ติดตั้ง |
|---|---|---|
| Platon | Schwengers et al. 2020, *Microb Genom*, doi:10.1099/mgen.0.000398 | `conda install -c conda-forge -c bioconda platon` + DB ~2.8 GB |
| MOB-suite (`mob_recon`) | Robertson & Nash 2018, *Microb Genom*, doi:10.1099/mgen.0.000206 | `conda install -c conda-forge -c bioconda mob_suite` + `mob_init` |
| PLASMe | Tang et al. 2023, *Nucleic Acids Res*, doi:10.1093/nar/gkad578 | git clone + conda env + DB 12.4 GB (ขยายแล้ว ~16 GB) |
| PlasmidHunter | Tian et al. 2024, *Brief Bioinform*, doi:10.1093/bib/bbae322 | `pip install plasmidhunter` + `diamond=2.1.8` + DB ~0.4 GB |

## สองการตัดสินใจที่สำคัญ

**1. 1 shard = 1 assembly ไม่ใช่ 1 fragment** — ทั้งสองเครื่องมือออกแบบมารับ "contigs ของ
draft assembly หนึ่งตัว" ไม่ใช่ sequence เดี่ยวที่ไม่เกี่ยวกัน dataset นี้ fragment tile เต็ม
replicon แบบไม่ซ้อนกัน การรวม fragment ของ assembly เดียวกันเข้าด้วยกันจึงใกล้เคียง
"ชุด contig ของไอโซเลตหนึ่ง" มากที่สุดเท่าที่ข้อมูลนี้ให้ได้
(test split: 16,944 fragments → **116 shards**, เฉลี่ย ~3.1 Mbp)

> **แต่ shard ≠ draft assembly ที่สมบูรณ์** — split แบ่งที่ระดับ cluster/replicon
> **ไม่ใช่** ระดับ assembly ดังนั้นโครโมโซมของไอโซเลตหนึ่งอาจอยู่ใน train ขณะที่พลาสมิด
> ของมันอยู่ใน test (112 จาก 500 assembly ข้าม split จริง) ใน test split จึงมี
> **70/116 shard ที่มีแต่ fragment โครโมโซม, 42 shard ที่มีแต่พลาสมิด, และแค่ 4 shard
> ที่มีทั้งสองอย่าง** มัธยฐานคือมี replicon ของไอโซเลตนั้นอยู่เพียงครึ่งเดียว
>
> shard ที่มีแต่พลาสมิดเป็น input ที่ **ง่ายกว่า** draft assembly จริง ส่วน shard ที่มีแต่
> โครโมโซมก็ **ยากกว่า** และไม่มีอันไหนเป็นรูปแบบที่เครื่องมือออกแบบมารับ —
> `analysis/dataset_stats.py` วัดตัวเลขนี้ไว้ (`dataset_shard_composition.csv`)
> และรายงานพิมพ์ตารางนี้ไว้ข้าง ๆ ผลของเครื่องมือเสมอ

**2. รันเครื่องมือรอบเดียว แล้วคำนวณทุก mode ย้อนหลัง** — ทำได้กับทั้ง Platon และ PLASMe
ด้วยเหตุผลเดียวกัน คือทั้งคู่มีไฟล์ที่เก็บ "ตัวเลขดิบของทุก contig" ไว้อยู่แล้ว

- **Platon `--characterize`** — `.tsv` ปกติเขียนแค่ contig ที่ตัดสินว่าเป็น plasmid แต่
  `--characterize` ปิด filter และรายงาน RDS + hallmark **ทุกคอลัมน์ของทุก contig** จึงคำนวณ
  กฎทั้ง 3 mode ย้อนหลังได้ และได้ RDS เป็น **continuous score** ไปคำนวณ AUROC/AUPRC
- **PLASMe `<temp>/PLASMe_candidate.csv`** — preset ทั้ง 3 เป็นแค่ threshold สามตัวบน
  identity, coverage และ probability ของ Transformer ซึ่งไฟล์นี้เขียนไว้ครบทุก contig ที่
  align ได้ จึงได้ครบ 3 mode + ได้ **Transformer probability เป็น continuous score**
  (ไฟล์ที่ PLASMe โชว์ออกมาเอง `<output>_report.csv` มีแต่ contig ที่มันตัดสินว่าเป็นพลาสมิด
  ใช้เป็นตัวส่วนไม่ได้ — `16_plasme.sh` จึงเก็บ temp dir ไว้ไม่ลบ)

ทั้งสองกฎถูกพิสูจน์ก่อนเชื่อ: `server/13_smoke_tools.sh` ตรวจว่ากฎที่คำนวณย้อนหลังตรงกับ
`platon --mode accuracy` ของจริงและตรงกับ `_report.csv` ของ PLASMe เอง 100%
และ `collect_plasme.py` ตรวจกฎของ PLASMe ซ้ำ**ทุก shard** (ฟรี เพราะทุก shard มี report
ของตัวเองอยู่แล้ว) ถ้าไม่ตรง collector จะหยุดและไม่ให้คะแนนเลย

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
| `collect_plasme.py` | รวม `PLASMe_candidate.csv` → `predictions_plasme_test.csv` + ตรวจกฎทุก shard |
| `collect_plasmidhunter.py` | รวม `predictions.tsv` → `predictions_plasmidhunter_test.csv` |
| `collect_runtime.py` | รวม `runtime.json` (หรือ timestamp ใน log) → `tool_runtime.{json,csv}` |
| `score_tools.py` | ให้คะแนนทั้ง 4 ด้วย `baseline/baselines/metrics.py` ตัวเดิม → `tool_baselines.{json,csv}` |
| `verify_platon_modes.py` | พิสูจน์ว่ากฎที่คำนวณย้อนหลัง == `--mode accuracy` ของจริง |
| `verify_plasme_modes.py` | พิสูจน์ว่ากฎที่คำนวณย้อนหลัง == `_report.csv` ของ PLASMe เอง |
| `toolbase/parse.py` | อ่าน output **ตามชื่อคอลัมน์** + assert คอลัมน์ที่ต้องใช้ (ทั้ง 4 เครื่องมือ) |
| `toolbase/rules.py` | กฎ 3 mode ของ Platon, label/score ของ mob_recon, 3 preset ของ PLASMe, label ของ PlasmidHunter |
| `toolbase/cfg.py` | โหลด config + import metric ของ baseline เดิม |

## ลำดับการรัน

บนเซิร์ฟเวอร์ใช้ `server/submit_all.sh tools` (จัดการ job array + dependency ให้) หรือรันมือ:

```bash
python shard_by_assembly.py                    # 116 shards
# ต่อ shard (job array ทำให้):
#   platon --characterize --db $PLATON_DB -o out/platon/<s> -p <s> shards/<s>.fasta
#   mob_recon --infile shards/<s>.fasta --outdir out/mobsuite/<s> --force
#   python $PLASME_DIR/PLASMe.py shards/<s>.fasta out/plasme/<s>/plasmid.fna \
#          -d $PLASME_DB -t 8 --temp out/plasme/<s>/temp
#   plasmidhunter -i shards/<s>.fasta -c 8 -o out/plasmidhunter/<s>
python collect_platon.py
python collect_mobsuite.py
python collect_plasme.py
python collect_plasmidhunter.py
python score_tools.py
python collect_runtime.py
```

## การให้คะแนน — 3 นโยบายต่อเครื่องมือ

ทุกตัวอาจไม่ตัดสิน fragment บางตัว — Platon เมื่อไม่เจอ marker protein, mob_recon เมื่อ
contig สั้นกว่า cutoff, PLASMe เมื่อ contig align กับ reference ไม่ติดเลย, PlasmidHunter
เมื่อโปรตีนของ contig ไม่มี DIAMOND hit — จึงรายงาน 3 แบบ (เลียนแบบ
`metrics.plasflow_threshold_rule` ที่จัดการ unclassified band ของ PlasFlow อยู่แล้ว):

| policy | ความหมาย |
|---|---|
| `chromosome` | unclassified → chromosome (conservative, เป็นตัวเลข headline) |
| `plasmid` | unclassified → plasmid (upper bound ของ recall) |
| `exclude` | ตัด unclassified ออก + รายงาน coverage แยก |

สองคอลัมน์ที่ต้องอ่านอย่างระวัง:
- `hard_label_score=True` → AUROC/AUPRC ของแถวนั้นมาจาก**การตัดสิน** ไม่ใช่ความน่าจะเป็น
  (mob_recon ไม่คืนค่าความน่าจะเป็นเลย) ตัวเลขจึงถูกครอบงำโดย label
  แถวที่เทียบ AUROC/AUPRC ได้อย่างเป็นธรรมมี 3 แถว: `platon_rds@*` (RDS),
  `plasme_transformer` (posterior ของ Transformer) และ `plasmidhunter`
  (posterior ของ Gaussian naive Bayes — ตัวเดียวที่ครอบคลุมทุก contig ที่มันให้คะแนน)
- `mcc_oracle` → threshold จูนบน **test split เอง** เพราะเราไม่ได้รันเครื่องมือบน validation
  (ไม่ต้องเทรนจึงไม่จำเป็น) ใช้เป็น **upper bound** ไม่ใช่จุดทำงานที่เลือกได้ล่วงหน้า
