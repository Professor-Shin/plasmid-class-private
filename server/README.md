# การรันงานบน CU-AIM

คู่มือลำดับคำสั่งสำหรับรันงานนี้บนคลัสเตอร์ CU-AIM — เพิ่ม baseline **Platon** และ
**MOB-suite (mob_recon)** และรัน ESM-2 แบบไม่มีคอขวดของโน้ตบุ๊ก

อ้างอิงข้อกำหนดของเครื่องจาก `survival-manual-cuaim.pdf` (CU-AIM User Manual 2026)

---

## ก่อนเริ่ม — 3 เรื่องที่ทำให้ติดบ่อย

1. **ต้องต่อ Chula VPN (Cisco AnyConnect) ทุกครั้ง** ก่อน `ssh` หรือ `rsync` ไม่งั้นต่อไม่ได้เลย
   (คู่มือ §1.1)
2. **VS Code ต้องเป็นรุ่น 1.85.2 เท่านั้น** รุ่นใหม่ใช้กับ CU-AIM ไม่ได้เพราะชนกับ system library
   ของเครื่อง ให้ติดตั้งแยกไว้อีกตัวและปิด auto-update (คู่มือ §2.2) — ไม่ต้องถอนรุ่นที่ใช้อยู่
3. **`/data/home` โควต้าเล็ก ห้ามเก็บงานไว้ที่นั่น** (คู่มือ §3.1 ย้ำ 2 ครั้ง) `env.sh` จึง export
   `CONDA_ENVS_DIRS`, `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR`,
   `SINGULARITY_CACHEDIR` ไปที่ `/data/project/...` ทั้งหมด **ถ้าลืม source `env.sh`
   `conda create` จะลง `~/.conda/envs` และ home เต็มทันที** (conda env 3 ตัว ~8 GB + DB ~7 GB)

---

## ลำดับคำสั่ง

### 0. สำรวจเครื่อง (ทำครั้งแรกครั้งเดียว)

```bash
ssh <user>@161.200.107.77 -i ~/.ssh/id_rsa_cuaim
cd <ที่ที่ clone โค้ดไว้>
bash server/probe_server.sh 2>&1 | tee server/logs/probe.txt
```

สคริปต์นี้ **ไม่เขียนอะไรเลย** แค่รายงาน: `check_qos` / `check_group` (QoS และ account ที่บัญชีได้จริง),
`sinfo` (ชื่อ partition และ GRES), `MaxArraySize`, module ที่มี, พื้นที่ `/data/project`, และ
login node ออกเน็ตได้ไหม

**แล้วแก้ `server/env.sh`** ตามหัวข้อ SUMMARY ที่สคริปต์พิมพ์ไว้ท้ายสุด — อย่างน้อยต้องแก้
`PROJECT`, `CUAIM_ACCOUNT`, `PART_CPU`, `PART_GPU`, `GRES_GPU`

> พื้นที่ที่ต้องใช้ **~24 GB**: dataset 2.5 + Platon DB 2.8 + MOB DB ~4 +
> embeddings **5.2** + conda envs ~8 + log/output ~1
>
> ตัวเลข embeddings **วัดจากข้อมูลจริง** ไม่ใช่ประมาณ: ปลด `max_orfs` แล้วได้ **8.79 M ORFs**
> จาก 112,929 fragments (2.13 G กรดอะมิโน) → float16 = **5.2 GiB**, float32 = **10.5 GiB**,
> ถ้าใช้ ESM-2 t12_35M = 7.9 GiB ดังนั้น `store_float16: true` ไม่ใช่ของแถม ถ้าปิดต้องเผื่อ
> เพิ่ม ~5 GB **ถ้าพื้นที่ไม่พอ** ดูหัวข้อ "พื้นที่ไม่พอ" ด้านล่าง

### 1. ติดตั้ง (login node, ~30-60 นาที)

```bash
bash server/00_setup_envs.sh 2>&1 | tee server/logs/setup.txt
```

สร้าง conda env 3 ตัวใน `$PROJECT/envs/` (แยกกันเพราะ MOB-suite บังคับ `numpy<1.23.5`,
`pandas<=1.5.3` ซึ่งอยู่ร่วมกับ `numpy 2.4.4` ที่โค้ดเราใช้ไม่ได้), ดาวน์โหลด Platon DB +
MOB-suite DB, และ **ดึง ESM-2 weights ล่วงหน้า** ลง `$HF_HOME` เผื่อ compute node ไม่มีเน็ต

รันซ้ำได้ — ทุกขั้นข้ามถ้ามีของอยู่แล้ว ดาวน์โหลดก็ resume ได้

`torch.cuda.is_available() -> False` ตอนนี้ **ถูกต้อง** เพราะ login node ไม่มี GPU
การตรวจจริงอยู่ใน `20_esm_embed.sh` ซึ่งรันใน allocation ที่มี GPU

### 2. ส่งข้อมูลขึ้น (รันบนโน้ตบุ๊ก)

```bash
CUAIM_USER=<user> CUAIM_KEY=~/.ssh/id_rsa_cuaim \
PROJECT=/data/project/<group>/<dir> bash server/01_sync_from_laptop.sh
```

ส่ง `dataset_mixed` (FASTA + CSV ~2.5 GB), metadata, และผล baseline เดิม พร้อมสร้าง
`server/dataset_md5.txt` จากเครื่องที่ผลิตข้อมูล

**ไม่ rebuild dataset บนเซิร์ฟเวอร์** เพราะ step01/02 ดึงจาก NCBI ซึ่ง `assembly_summary.txt`
เปลี่ยนทุกสัปดาห์ → เสี่ยงได้ dataset ไม่เหมือนเดิมและตัวเลขในรายงานทั้งหมดเทียบไม่ได้

จากนั้น clone โค้ดบนเซิร์ฟเวอร์:
```bash
cd "$PROJECT" && git clone <repo> . || git pull
```

### 3. ตรวจว่าข้อมูลมาครบ

```bash
sbatch server/02_verify_data.sh
```

เทียบ md5 ทุกไฟล์กับโน้ตบุ๊ก + รัน `verify_dataset.py` + ยืนยันว่าจำนวน fragment/assembly
ตรงกับที่ sbatch ทุกตัวถูกคำนวณมาจาก (test = 16,944 fragments จาก **116 assemblies**)

### 4. รันทั้ง pipeline

```bash
bash server/submit_all.sh                 # ทั้งหมด
bash server/submit_all.sh tools           # แค่ Platon + MOB-suite
bash server/submit_all.sh esm             # แค่ ESM-2
DRY_RUN=1 bash server/submit_all.sh       # ดูคำสั่ง sbatch ก่อนไม่ submit จริง
```

```
10_shard ──┬─ 11_platon[1-116%5]   (cpu40h) ──┬─ 14_collect_score ─┐
           ├─ 12_mobsuite[1-116%5] (cpu40h) ──┤                    ├─ 30_report
           └─ 13_smoke_tools (gate) ──────────┘                    │
20_esm_embed (gpu40g) ── 21_esm_merge ── 22_esm_train (gpu40g) ────┘
```

ติดตามงาน:
```bash
squeue -u "$USER" -o '%.18i %.9P %.12j %.8T %.10M %.9l %R %b'
sacct -j <id> --format=JobID,JobName%18,State,ExitCode,Elapsed
scancel -f <id>            # ยกเลิก (รวม job step ลูกทั้งหมด)
```

---

## ทำไมรูปร่างงานเป็นแบบนี้

**Platon/MOB-suite แบ่งเป็น 116 งาน = 1 งานต่อ 1 assembly** ไม่ใช่ต่อ fragment และไม่ใช่
ก้อนเดียว เพราะทั้งคู่ออกแบบมารับ "contigs ของ draft assembly หนึ่งตัว" dataset เราแบ่ง split
ที่ระดับ replicon และ fragment ของ assembly หนึ่งก็ tile เต็ม replicon แบบไม่ซ้อนกัน →
fragment ของ assembly หนึ่ง **คือ draft assembly ที่แตกเป็น contig จริง ๆ** นี่เป็นการใช้
เครื่องมืออย่างซื่อตรงที่สุด และได้ขนาด array ที่พอดีไปด้วย (เฉลี่ย ~3.1 Mbp/งาน ≈ 1 จีโนม)

**Throttle `%5`** เพราะ QoS `cpu40h` ให้ 40 CPU และคู่มือ §5.1.2 เตือนว่าลิมิตอาจนับ**รวม**
ทุก job ของผู้ใช้ → 5 งาน × 8 CPU = 40 พอดีทั้งสองกรณี

**Platon รัน `--characterize` รอบเดียว** เพราะ `.tsv` ปกติเขียนแค่ contig ที่เป็น plasmid
แต่ `--characterize` ปิด filter และรายงาน RDS + hallmark ทุกคอลัมน์ของ**ทุก** contig →
คำนวณกฎทั้ง 3 mode ย้อนหลังได้ (`tools-baseline/toolbase/rules.py`) และได้ **RDS เป็น
continuous score** สำหรับ AUROC/AUPRC `13_smoke_tools.sh` พิสูจน์ว่ากฎที่คำนวณย้อนหลัง
ตรงกับ `platon --mode accuracy` ของจริง 100% **ถ้าไม่ตรงมันจะ fail และ 14 จะไม่รัน**

**ESM-2 เป็น job เดียวไม่ใช่ array** เพราะ QoS ให้ GPU 1 ตัวต่อ job (ยกเว้น `gpu40g24m` ได้ 2)
shard จึงมีไว้เพื่อ **resume** ไม่ใช่ขนาน — แต่ละ shard เป็น `srun` แยกมี log ของตัวเอง และ
shard ที่ cache แล้วจะถูกข้าม **ถ้า job ชน time limit ให้ `sbatch` สคริปต์เดิมซ้ำ มันเดินต่อเอง**
(โน้ตบุ๊กไม่มีคุณสมบัตินี้ — split ที่ถูก interrupt เริ่มใหม่จาก 0 และ test split เคยใช้ 233 นาที)

**ขอ 16 CPU พร้อม GPU** เพื่อให้ ORF worker 14 ตัวป้อนงานให้ GPU ไม่ขาด — **แต่คอขวดคือ
transformer ไม่ใช่ ORF calling** (วัดแล้ว: six-frame translation ทั้ง dataset ใช้แค่ ~1.3
CPU-hours เพราะ `translate` ของ Biopython เป็น C ส่วนงานที่หนักจริงคือ 2.13 G กรดอะมิโน
ที่ต้องผ่าน ESM-2) ตอนวางแผนผมเข้าใจผิดว่า ORF calling จะเป็นคอขวด

---

## การตรวจสอบที่ฝังอยู่ในสคริปต์ (ไม่ต้องสั่งเอง)

| ตรวจอะไร | อยู่ที่ไหน | ถ้า fail หมายความว่า |
|---|---|---|
| ข้อมูลเหมือนโน้ตบุ๊ก bit-for-bit | `02_verify_data.sh` | rsync ไม่ครบ หรือ dataset ถูก rebuild |
| จำนวน fragment/assembly ตรงกับที่ sbatch คำนวณไว้ | `_dataset_facts.py` | ต้องแก้ array size ก่อน submit |
| สองเครื่องมือรันได้จริง + schema ของ output ถูก | `13_smoke_tools.sh` | เวอร์ชันเครื่องมือเปลี่ยน column → ต่อ alias ใน `toolbase/parse.py` |
| กฎ Platon ที่คำนวณย้อนหลังตรงกับของจริง | `13_smoke_tools.sh` | ต้องถอยไปรัน `--mode accuracy` ทั้ง 116 shard |
| ไม่มี fragment หายระหว่างทาง | `collect_*.py` | shard บางตัวล้ม ให้ resubmit array |
| batched == per-fragment, GPU == CPU | `verify_components.py` | การ refactor ทำโมเดลเปลี่ยนความหมาย |
| ตัวเลขเดิมยัง reproduce ได้ | `_regression_check.py` | **หยุด** อย่าตีความผลเซิร์ฟเวอร์เป็นผลการทดลอง |

`_regression_check.py` รัน `config.yaml` **ตัวเดิม** (batch_size 1, max_orfs 10, subsample เล็ก)
บน cache ของโน้ตบุ๊ก แล้วเทียบ MCC กับ `results_mixed/esm2_variants.csv` ที่รายงานอ้างอิง
โดยใช้ tolerance = seed spread ของรอบเดิมเอง ข้ามได้ด้วย `SKIP_REGRESSION=1` หลังผ่านแล้ว

---

## พื้นที่ไม่พอ — ตัดตามลำดับนี้

1. `esm.store_float16: true` ใน `config_server.yaml` (ค่าตั้งต้นเปิดอยู่แล้ว) — cache 3.1 GB
   เหลือ ~1.6 GB `verify_components.py` วัดให้ว่าเสียความแม่นยำเท่าไหร่
2. ลบ shard หลัง merge — `21_esm_merge.sh` ทำให้อยู่แล้ว (ตั้ง `KEEP_SHARDS=1` ถ้าไม่อยากลบ)
3. `subsample:` ใน `config_server.yaml` ใส่ค่า train ที่ใหญ่ขึ้น (เช่น 20,000) แทน `null` —
   ยังแก้คอขวด `max_orfs` และ class-ratio ได้ครบ เสียแค่ขนาด train set
4. ใช้ Singularity แทน conda (`TOOL_RUNTIME=singularity` ใน `env.sh`) ประหยัด ~6 GB
5. ลบ Platon DB หลัง array เสร็จ (เก็บแค่ TSV)

---

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | หน้าที่ |
|---|---|
| `env.sh` | **ปุ่มปรับทุกอย่างของคลัสเตอร์อยู่ที่นี่ไฟล์เดียว** |
| `lib.sh` | โหลด module, activate env, สลับ conda/singularity, map array task → shard |
| `probe_server.sh` | สำรวจเครื่อง (read-only) ทำก่อนอื่น |
| `00_setup_envs.sh` | conda env + DB + ESM weights (login node) |
| `01_sync_from_laptop.sh` | rsync dataset ขึ้นเซิร์ฟเวอร์ (รันบนโน้ตบุ๊ก) |
| `02_verify_data.sh` | md5 + `verify_dataset.py` + ตรวจจำนวน |
| `10_shard.sh` | test split → 1 FASTA ต่อ assembly |
| `11_platon.sh` / `12_mobsuite.sh` | job array 116 งาน resume ได้ |
| `13_smoke_tools.sh` | **gate**: เครื่องมือรันได้ + กฎ Platon ถูก |
| `14_collect_score.sh` | รวมผล → ให้คะแนนด้วย metric ของ baseline เดิม |
| `20_esm_embed.sh` | GPU: ORF + ESM-2 embedding วน shard resume ได้ |
| `21_esm_merge.sh` | รวม shard เป็น cache เดียวต่อ split |
| `22_esm_train.sh` | GPU: ตรวจ equivalence + regression แล้วเทรน ladder |
| `30_report.sh` | ตารางรวมทุก baseline เป็น markdown |
| `submit_all.sh` | ร้อย dependency ทั้งหมด |
