# บันทึกการติดตั้ง environment บน CU-AIM (`00_setup_envs.sh` 6 ขั้น)

> **สำหรับใคร:** คนที่ต้องรัน setup นี้ซ้ำบนเครื่องอื่น หรือตรวจว่าตัวเลขในรายงานมาจาก
> environment แบบไหน ถ้าต้องการแค่*ขั้นตอนที่ต้องทำ* ให้อ่าน [MANUAL.md](MANUAL.md)
> ไฟล์นี้ตอบว่า**เจออะไรระหว่างทางและแก้ยังไง**
>
> วันที่ติดตั้งสำเร็จ: 2026-10-26 07:57 (+07) · เครื่อง: CU-AIM (`cuaim-frontend1` / `cuaim-fat-01`)
> · commit ที่ใช้: `b72cd7a`

---

## 1. สภาพแวดล้อมที่เจอจริง

ค่าทั้งหมดมาจาก `bash server/probe_server.sh` ไม่ใช่จากคู่มือ

| | ค่า |
|---|---|
| account / user | `g_naruemon` / `naruemonp` |
| partition | `cpu`, `gpu`, `interactive-cpu`, `interactive-gpu` |
| QoS ที่บัญชีมี | `cpu40h`, `gpu10gh`, `gpu20gh`, `gpu40g`, `gpu40g24m`, `interactive-gpu` |
| node | cpu: `cuaim-fat-[01-02]` (72+ CPU, 690 GB) · gpu: `cuaim-dgx-01` (A100 MIG) |
| glibc | **2.17** (manylinux2014) |
| module ที่มี | `Anaconda3/2021.11` (conda 4.x), `CUDA/11.8.0`, `Apptainer/1.3.2` |
| `mamba` | **ไม่มีในระบบ** |
| driver CUDA | 12.4 → wheel `cu124` |
| `MaxArraySize` | 1001 (งานนี้ต้องการ ≥116) |
| พื้นที่ | `/data` ว่าง 914 TB · `/data/home` limit 200 GB |
| เน็ตจาก login node | zenodo / huggingface / anaconda / ftp.ncbi เข้าถึงได้ |

---

## 2. คำสั่งที่ใช้จริง (ลำดับสุดท้ายที่สำเร็จ)

```bash
# --- ครั้งเดียว: ค่าเฉพาะเครื่อง อยู่ในไฟล์ที่ git ไม่แตะ
cd /data/project/naruemon/siwakorn-s_plasmid_class/plasmid-class-private
cat > server/env.local.sh <<'EOF'
export PROJECT="/data/project/naruemon/siwakorn-s_plasmid_class/plasmid-class-private"
export CUAIM_ACCOUNT="g_naruemon"
EOF
source server/env.sh

# --- solver ที่กิน RAM น้อย ติดตั้งใน $PROJECT ไม่แตะ base ของคลัสเตอร์
mkdir -p "$PROJECT/bin" && cd "$PROJECT/bin"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj bin/micromamba --strip-components=1
ln -sf "$PROJECT/bin/micromamba" "$PROJECT/bin/mamba"
export PATH="$PROJECT/bin:$PATH"

# --- ติดตั้งจริง: ต้องอยู่ใน allocation ไม่ใช่ login node เปล่า ๆ
srun -p "$PART_CPU" --qos="$QOS_CPU" --account="$CUAIM_ACCOUNT" \
     -c 4 --mem=24G -t 3:00:00 --pty bash -l
cd "$PROJECT" && export PATH="$PROJECT/bin:$PATH"
bash server/00_setup_envs.sh 2>&1 | tee -a server/logs/setup.txt
```

รันซ้ำได้เสมอ ทุกขั้นข้ามเองถ้าผ่านแล้ว (เขียน `.setup_ok` ไว้) ขั้นที่ค้างครึ่งทางจะถูกสร้างใหม่

---

## 3. ทั้ง 6 ขั้นทำอะไร และได้อะไร

| ขั้น | ทำอะไร | ผลลัพธ์ |
|---|---|---|
| **1/6** | conda env `plasmid-ml` — python 3.11 + torch จาก CUDA wheel + pin ใน `requirements-plasmid-ml.txt` | `envs/plasmid-ml` **5.8 GB** + `plasmid-ml-server.lock.txt` |
| **2/6** | conda env `platon` — ลาก `prodigal`, `diamond`, `hmmer`, `blast`, `mummer` มาด้วย | `envs/platon` **1.1 GB** |
| **3/6** | conda env `mobsuite` — ลาก `blast+`, `mash` | `envs/mobsuite` **5.0 GB** |
| **4/6** | ดาวน์โหลด + แตก Platon DB (1.6 GB บีบอัด) แล้วเขียน `PROVENANCE.txt` | `db/platon` **2.8 GB** |
| **5/6** | `mob_init` ดาวน์โหลด 450 MB แล้วสร้าง BLAST/mash index เอง | `db/mobsuite` **2.9 GB** |
| **6/6** | ดึงน้ำหนัก ESM-2 ไว้ล่วงหน้า (compute node อาจไม่มีเน็ต) | `.hf` **160 MB** |

**ทำไมต้องแยก env สามตัว** — MOB-suite pin `numpy<1.23.5` / `pandas<=1.5.3` ซึ่งอยู่ร่วมกับ
numpy/pandas ที่โค้ดโปรเจกต์ต้องใช้ไม่ได้ และ Platon ก็ pin biopython/blast ของตัวเองทับอีกชั้น

---

## 4. ปัญหาที่เจอและวิธีแก้ (เรียงตามที่เจอ)

### 4.1 conda solve ถูก OOM-kill บน login node

```
Solving environment: ...working... /…/server/lib.sh: line 21:  8542 Killed   "$CONDA_EXE" …
```

ล้มที่ขั้น 3/6 ตอน solve `mob_suite` ข้อความไม่ได้บอกว่าหน่วยความจำหมด แต่ `Killed` + exit 137
คือ kernel ฆ่า process

**สาเหตุ** conda classic solver กิน RAM หลาย GB บนกราฟ dependency ของ bioconda และ login node
จำกัดหน่วยความจำต่อ process — ซึ่งตรงกับที่คู่มือ CU-AIM §2.1 บอกว่า login node มีไว้จัดการไฟล์
กับ submit งานเท่านั้น (สคริปต์ `10_shard.sh` ในโปรเจกต์นี้ก็อ้างข้อเดียวกันอยู่แล้ว แต่ตัว
`00_setup_envs.sh` กลับเขียนว่าให้รันบน login node — ขัดกันเอง)

**แก้ 2 ชั้น**
1. ขอ allocation ก่อน (`srun … --mem=24G --pty bash -l`) — MANUAL ขั้น 2 เขียนใหม่แล้ว
2. ใช้ solver ที่เบากว่า: `conda install -n base conda-libmamba-solver` **ใช้ไม่ได้** เพราะ
   CU-AIM มีแค่ `Anaconda3/2021.11` (conda 4.x ไม่มี plugin นี้) และ base เป็น module ที่
   ทั้งคลัสเตอร์ใช้ร่วมกัน **ห้ามแตะ** → ใช้ `micromamba` ไบนารีเดี่ยวใน `$PROJECT/bin` แทน
   (`00_setup_envs.sh` มองหาชื่อ `mamba` ใน PATH อยู่แล้ว)

สคริปต์ตอนนี้จับ exit 137/139 แล้วพิมพ์ทางแก้ทั้งสองทางออกมาเอง — commit `8ed6630`

### 4.2 glibc 2.17 กับ torch (เจอก่อนหน้านี้ ยืนยันซ้ำ)

PyTorch ≥ 2.6 ย้าย wheel ไป manylinux_2_28 (ต้องการ glibc ≥ 2.28) **ลงได้แต่ import ไม่ขึ้น**
สคริปต์อ่าน glibc จาก `ldd --version` แล้ว cap ที่ `torch==2.5.1` อัตโนมัติ พร้อมเลือก `cuXXX`
ตาม driver จาก `nvidia-smi` — ผลที่เลือกจริงบันทึกไว้ที่ `$PROJECT/.cuda_tag`

ได้ `torch 2.5.1+cu124` บน python 3.11 (เหตุผลเดียวกับที่ VS Code ต้อง ≤ 1.85.2)

### 4.3 pin ของ numpy/pandas/sklearn/scipy ลงไม่ได้

pin ที่ล็อกไว้จากโน้ตบุ๊กมีเฉพาะ wheel แบบ manylinux_2_28 จึงต้องผ่อนลงมาเป็นรุ่นล่าสุดที่
ลงบน glibc 2.17 ได้ — commit `04e4fde`

| | โน้ตบุ๊ก | เซิร์ฟเวอร์ |
|---|---|---|
| torch | 2.11.0+cpu | **2.5.1+cu124** |
| numpy | 2.4.4 | **2.2.6** |
| pandas | 3.0.2 | **2.3.2** (ข้าม major) |
| scikit-learn | 1.8.0 | **1.7.2** |
| scipy | 1.17.1 | **1.16.3** |
| transformers / tokenizers / safetensors / biopython | 5.7.0 / 0.22.2 / 0.7.0 / 1.87 | เท่าเดิม |

ตรวจแล้วว่าโค้ดไม่ได้ใช้ API ที่มีเฉพาะ pandas 3 หรือที่ pandas 2 ถอดไปแล้ว และไม่มีจุดไหน
assert เวอร์ชัน — ด่านที่จะจับถ้าผลเปลี่ยนคือ `02_verify_data.sh` ข้อ 2 (pipeline คำนวณ
dataset ใหม่เอง) กับ component checks ใน `22_esm_train.sh`

> **ผลต่อรายงาน:** environment นี้ไม่ reproduce โน้ตบุ๊กแบบ bit ต่อ bit ต้องอ่าน
> regression check เทียบกับ tolerance ไม่ใช่คาดหวัง `diff 0.0000` และต้อง commit
> `plasmid-ml-server.lock.txt` (ผลของ `pip freeze`) ไว้เป็นหลักฐานของ Methods

### 4.4 `mob_init` ล้มทั้งที่โหลดสำเร็จ

```
INFO : Downloading databases successful, now building databases at …
ERROR: Downloading databases failed, please check your internet connection and retry
ERROR: Process failed with error [Errno 13] Permission denied: 'makeblastdb'
```

**ข้อความแรกหลอก** — ดาวน์โหลด 450 MB สำเร็จแล้ว ที่ล้มคือตอนสร้าง index

**สาเหตุ** สคริปต์เรียก `"$ENV_MOB/bin/mob_init"` ด้วย absolute path โดยไม่ activate env แต่
`mob_init` ไปเรียก `makeblastdb` / `blastn` / `mash` ด้วย**ชื่อเปล่า** ซึ่งไม่อยู่ใน `PATH`

**แก้** `PATH="$ENV_MOB/bin:$PATH" "$ENV_MOB/bin/mob_init" …` — commit `b72cd7a`

### 4.5 ขั้น DB เคย `SKIP` ทั้งที่ยังไม่ครบ

เงื่อนไขเดิมคือ "โฟลเดอร์ไม่ว่าง" ซึ่งการดาวน์โหลดที่ตายกลางทางก็ทิ้งไฟล์ไว้เต็มไปหมด
เปลี่ยนเป็นเขียน `.setup_ok` เมื่อสำเร็จจริง แล้ว skip เฉพาะเมื่อเจอ sentinel นั้น
(เป็นบั๊กแบบเดียวกับที่เคยแก้ให้ conda env ไปแล้ว) — commit `b72cd7a`

> **ถ้า migrate จากสถานะเก่า** ที่ DB ครบแล้วแต่ยังไม่มี sentinel ให้ `touch "$PLATON_DB/.setup_ok"`
> เอง ไม่งั้นมันจะโหลด 1.6 GB ใหม่

### 4.6 job script ไม่มี `-l`

ทุกตัวอย่างในคู่มือ CU-AIM ใช้ `#!/bin/bash -l` เพราะ `module` เป็น shell function จาก
`/etc/profile.d` ซึ่ง non-login shell ไม่ source แต่สคริปต์ในโปรเจกต์เป็น `#!/bin/bash` เปล่า
ทั้ง 13 ไฟล์ → เสี่ยง `module: command not found` ตอน `sbatch`

**แก้** เติม `-l` ทุกไฟล์ + เพิ่ม `cuaim_init_modules` ใน `lib.sh` ที่กู้ `module` จาก
`$MODULESHOME` ให้อีกชั้น — commit `8ed6630`

### 4.7 `gpu40g` ให้ 8 ชั่วโมง ไม่ใช่ 8 วัน

`check_qos` บนเครื่องจริงรายงาน `MaxWall 08:00:00` ขณะที่ Table 3.1 ของคู่มือเขียน 8 days
`env.sh` เคยตั้ง `TIME_EMBED=2-00:00:00` ซึ่ง Slurm จะปฏิเสธตอน submit

**แก้** ตั้งทั้ง `TIME_EMBED` / `TIME_TRAIN` = `08:00:00` และบันทึกทางเลือก `gpu10gh`
(4 วัน แต่ MIG `2g.10gb` เล็กกว่า) ไว้ — commit `6a4fe03`

### 4.8 `env.sh` ชนทุกครั้งที่ `git pull`

ค่าเฉพาะเครื่อง (`PROJECT`, `CUAIM_ACCOUNT`) เคยอยู่ในไฟล์ที่ tracked ทำให้ pull ชนสองรอบ
ย้ายไป `server/env.local.sh` (gitignore, มี `.example` ให้) ซึ่ง `env.sh` source ก่อน
แล้วค่อยใส่ default ของตัวเอง — commit `33fe23d`

**กับดักที่ตามมา:** `env.sh` ใช้รูป `: "${VAR:=default}"` = ใช้ค่านี้*ถ้ายังไม่มีค่า* ฉะนั้น
ตัวแปรที่ export ค้างใน shell จะ**ไม่ถูกทับ**เมื่อ source ซ้ำหลังแก้ไฟล์ → เปิด shell ใหม่
หรือ `unset` ก่อน

### 4.9 อื่น ๆ

- **push จากเซิร์ฟเวอร์ไม่ได้** — credential ที่ cache ในบัญชีร่วมเป็นของ GitHub user อื่น
  (อ่านได้ เขียนไม่ได้) และไม่ควรเอา token ตัวเองไปวางในบัญชีที่ใช้ร่วมกัน →
  commit บนเซิร์ฟเวอร์ แล้วดึงจากเครื่องตัวเองด้วย
  `git fetch ssh://<user>@<host>/<PROJECT path> main:<branch>` แล้ว push จากเครื่องนั้น
- **`CUAIM_ACCOUNT` ว่าง ≠ ไม่ใส่** — `--account=` ที่ว่างเปล่าทำให้ Slurm ตอบ
  `Invalid account or account/partition combination specified` ถ้าไม่แน่ใจชื่อ account
  ให้ปล่อยตัวแปรว่างไว้ สคริปต์จะไม่ส่ง flag นั้นเลย
- **env ค้างจากการ debug** — `envs/plasmid-ml-py311` (5.3 GB) ที่เคยสร้างด้วยมือ ต้องลบทิ้ง
  (`conda env remove -p …`) ไม่งั้นดูแล env สองชุดและ sbatch อาจหาผิดตัว

---

## 5. สถานะสุดท้าย

```
envs/plasmid-ml   5.8G      db/platon     2.8G
envs/platon       1.1G      db/mobsuite   2.9G
envs/mobsuite     5.0G      .hf           160M
```

ตรวจว่าใช้ได้จริงด้วย:

```bash
cd "$PROJECT" && source server/env.sh
"$ENV_ML/bin/python" server/_check_ml_env.py       # torch / numpy / pandas / transformers
"$ENV_PLATON/bin/platon" --version
"$ENV_MOB/bin/mob_recon" --version
cat "$PROJECT/.cuda_tag"                           # CUDA_TAG / TORCH_SPEC / GLIBC ที่เลือกจริง
cat "$PLATON_DB/PROVENANCE.txt"                    # DB url + platon version + วันที่โหลด
```

> `torch.cuda.is_available() -> False` บน login node **ถูกต้อง** — login node ไม่มี GPU
> การตรวจจริงอยู่ใน `20_esm_embed.sh` ที่รันใน GPU allocation

---

## 6. บทเรียนที่ใช้ซ้ำได้

1. **ถือผล `check_qos` / `probe_server.sh` เป็นหลัก ไม่ใช่ตารางในคู่มือ** — ผู้ดูแลเปลี่ยนค่าได้
   และเราเจอมาแล้วหนึ่งจุด (8 ชั่วโมง vs 8 วัน)
2. **`Killed` ไม่เคยแปลว่า package มีปัญหา** — มันคือหน่วยความจำ ดู exit code 137 ก่อนเสมอ
3. **ข้อความ error ของเครื่องมือภายนอกเชื่อไม่ได้** — `mob_init` บอกว่าเน็ตมีปัญหา
   ทั้งที่เป็น `PATH` อ่าน errno จริง (`EACCES`) แล้วค่อยสรุป
4. **เงื่อนไข skip ต้องเป็น sentinel ที่เขียนหลังตรวจแล้ว** ไม่ใช่ "มีไฟล์อยู่" — ไม่งั้นของที่
   พังครึ่งทางจะไหลไปถึง job จริง
5. **แยกค่าเฉพาะเครื่องออกจากไฟล์ที่ tracked ตั้งแต่แรก** ประหยัดการแก้ conflict ทุกครั้งที่ pull
