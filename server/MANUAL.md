# คู่มือรันงานบน CU-AIM — ทีละขั้น

คู่มือนี้พาทำตั้งแต่ศูนย์จนได้ตารางผลลัพธ์ **ทำตามจากบนลงล่างได้เลย** ทุกบล็อกคำสั่ง copy-paste ได้
ถ้าอยากรู้ว่า*ทำไม*ออกแบบแบบนี้ให้อ่าน [README.md](README.md) คู่มือนี้ตอบแค่ *ต้องทำอะไร*

| | |
|---|---|
| **งานที่จะรัน** | เพิ่ม baseline Platon + MOB-suite และรัน ESM-2 แบบปลดคอขวดทั้งหมด |
| **เวลารวม** | ~2 ชม. งานคน (ส่วนใหญ่คือรอดาวน์โหลด) + ~8-14 ชม. งานเครื่อง |
| **พื้นที่** | ~24 GB ใน `/data/project/...` |
| **ต้องมีก่อน** | บัญชี CU-AIM ที่ public key ถูกอนุมัติแล้ว + Cisco AnyConnect VPN |

**แผนที่ทั้งหมด** — ขั้น 0-4 ทำครั้งเดียว ขั้น 5-6 คือรอบการรันจริง

```
0. ครั้งแรกสุด (VPN / ssh key / VS Code)
1. probe เครื่อง  ->  กรอก env.sh          [~10 นาที]
2. clone + ติดตั้ง env + ดาวน์โหลด DB      [~30-60 นาที, ส่วนใหญ่รอโหลด]
3. ส่ง dataset ขึ้น (รันบนโน้ตบุ๊ก)         [~20-40 นาที ขึ้นกับเน็ต]
4. ตรวจว่าข้อมูลมาครบ                      [~5 นาที]
5. submit งานทั้งหมด + ติดตาม               [~8-14 ชม. เครื่องทำ]
6. เก็บผลกลับมา
```

---

## ขั้น 0 — ครั้งแรกสุด (ทำครั้งเดียวต่อเครื่องที่ใช้)

**0.1 ต่อ VPN** — ติดตั้ง Cisco AnyConnect ตาม https://www.it.chula.ac.th/en/service/en-vpn/
แล้ว sign in ด้วย user/password ของจุฬา

> **ต้องต่อ VPN ทุกครั้งที่จะใช้เซิร์ฟเวอร์** ไม่ใช่แค่ครั้งแรก ถ้า `ssh` ค้างหรือ timeout
> ให้เช็ค VPN ก่อนอย่างอื่นเสมอ

**0.2 สร้าง SSH key** (ถ้ายังไม่มี) แล้วส่ง **ไฟล์ `.pub` เท่านั้น** ให้อาจารย์/ผู้ดูแลคลัสเตอร์

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_cuaim -C "<your-email>"
cat ~/.ssh/id_rsa_cuaim.pub        # ส่งบรรทัดนี้ไป
```

รอยืนยันว่า key ถูกเพิ่มแล้วก่อนลองต่อ

**0.3 ทดสอบต่อ**

```bash
ssh <user>@161.200.107.77 -i ~/.ssh/id_rsa_cuaim
```

**0.4 VS Code (ถ้าจะใช้)** — **ต้องเป็นรุ่น 1.85.2 เท่านั้น** รุ่นใหม่ใช้กับ CU-AIM ไม่ได้เพราะชนกับ
system library ของเครื่อง ไม่ต้องถอนรุ่นที่ใช้อยู่ — โหลด 1.85.2 แยกไว้อีกตัว **ปิด auto-update**
แล้วติดตั้ง extension `Remote-SSH` ในตัวนั้น จากนั้นใส่ใน `~/.ssh/config`:

```
Host cuaim
    HostName 161.200.107.77
    User <user>
    IdentityFile ~/.ssh/id_rsa_cuaim
```

---

## ขั้น 1 — probe เครื่อง แล้วกรอก `env.sh`

**ทำขั้นนี้ก่อนอย่างอื่นเสมอ** เพราะยังไม่รู้ 3 อย่าง: path โปรเจกต์ที่เขียนได้, พื้นที่ที่เหลือ,
และชื่อ partition จริง

`probe_server.sh` ไม่ต้องพึ่งไฟล์อื่นเลย ส่งขึ้นไปไฟล์เดียวแล้วรัน — **ยังไม่ต้อง clone**
เพราะ checkout ของ repo ~250 MB (CSV ที่ tracked) ซึ่งใหญ่เกินจะไปวางใน `/data/home`
ที่โควต้าเล็ก ต้องรู้ `PROJECT` ก่อนค่อย clone (ขั้น 2)

```bash
# รันบนโน้ตบุ๊ก: ส่ง probe script ขึ้นไป
scp -i ~/.ssh/id_rsa_cuaim "d:/senior project/server/probe_server.sh" \
    <user>@161.200.107.77:~/

# แล้ว ssh ขึ้นไปรัน
ssh <user>@161.200.107.77 -i ~/.ssh/id_rsa_cuaim
bash ~/probe_server.sh 2>&1 | tee ~/probe.txt
```

สคริปต์นี้ **read-only** ไม่ติดตั้งหรือเขียนอะไร อ่านผลที่หัวข้อ `SUMMARY` ท้ายสุด แล้วจดค่าเหล่านี้:

| ตัวแปร | หาจากหัวข้อ | ตัวอย่าง |
|---|---|---|
| `PROJECT` | §5 บรรทัด `WRITABLE` | `/data/project/naruemon/siwak_s_plasmid` |
| `CUAIM_ACCOUNT` | §1 `check_group` / `sacctmgr` | `naruemon` |
| `PART_CPU` / `PART_GPU` | §2 รายชื่อ partition | `cpu` / `gpu` |
| `GRES_GPU` | §2 รายการ GRES | `gpu:7g.40gb:1` |
| `MODULE_CONDA` / `MODULE_CUDA` | §4 | `Anaconda3` / `CUDA/11.8.0` |

**เช็คพื้นที่จาก §5**: ต้องมีว่าง **~24 GB** ถ้าไม่ถึง ไปอ่าน [ภาคผนวก C](#ภาคผนวก-c--พื้นที่ไม่พอ) ก่อน

**เช็ค §1 ว่าบัญชีมี QoS อะไร** คู่มือ CU-AIM ระบุ `cpu40h` (40 CPU) และ `gpu40g`
(1×7g.40gb, 16 CPU, 64 GB, 8 วัน) ซึ่งเป็นสองตัวที่งานนี้ใช้ ถ้า `check_qos` ไม่มีสองตัวนี้
ให้แก้ `QOS_CPU` / `QOS_GPU` / `GRES_GPU` ใน `env.sh` ตามที่มีจริง

**เช็ค §3 ว่า `MaxArraySize` ≥ 116** (งาน Platon/MOB เป็น job array 116 งาน)

**เช็ค §6 ว่า login node ออกเน็ตได้** ถ้าออกไม่ได้ ขั้น 2 จะโหลด DB ไม่ได้ → ต้องคุยกับผู้ดูแล

---

## ขั้น 2 — clone จริง + ติดตั้ง

```bash
# ใช้ PROJECT ที่ได้จากขั้น 1
export PROJECT=/data/project/<group>/<dir>      # <-- แก้เป็นของคุณ
mkdir -p "$PROJECT" && cd "$PROJECT"
git clone https://github.com/Professor-Shin/plasmid-class-private.git .
rm -f ~/probe_server.sh ~/probe.txt             # ไม่ต้องใช้แล้ว (probe.txt copy เก็บไว้ก่อนถ้าอยาก)
du -sh "$PROJECT"                               # ~250 MB (โค้ด + CSV ที่ tracked)
```

> repo เป็น private → `git clone` จะถาม GitHub credential ถ้าไม่สะดวกพิมพ์รหัสบนเซิร์ฟเวอร์
> ใช้ Personal Access Token หรือ `gh auth login` ก็ได้ อีกทางคือ clone บนโน้ตบุ๊กแล้ว
> `rsync -av --exclude .git` ขึ้นไป แต่การ clone ตรงสะดวกกว่าตอนจะ `git pull` อัปเดตภายหลัง

**แก้ `server/env.sh`** ด้วยค่าที่จดไว้จากขั้น 1:

```bash
nano server/env.sh      # หรือเปิดใน VS Code
```

แก้ 7 ค่าในหัวข้อ identity/Slurm/modules (ตรงกับ 7 แถวในตารางขั้น 1):

```bash
: "${PROJECT:=/data/project/naruemon/siwak_s_plasmid}"     # <-- ของคุณ
: "${CUAIM_ACCOUNT:=naruemon}"                             # <-- ของคุณ
: "${PART_CPU:=cpu}"
: "${PART_GPU:=gpu}"
: "${GRES_GPU:=gpu:7g.40gb:1}"
: "${MODULE_CONDA:=Anaconda3}"
: "${MODULE_CUDA:=CUDA/11.8.0}"
```

> `QOS_CPU` (`cpu40h`) และ `QOS_GPU` (`gpu40g`) มีค่าตั้งต้นให้แล้ว — แก้เฉพาะเมื่อ
> `check_qos` ในขั้น 1 บอกว่าบัญชีได้ชื่อ QoS อื่น

> ส่วนที่เหลือของ `env.sh` **อย่าลบ** โดยเฉพาะบล็อก `CONDA_ENVS_DIRS` / `HF_HOME` / `TMPDIR`
> ที่ย้าย cache ออกจาก `/data/home` ซึ่งโควต้าเล็กมาก — ถ้าไม่มีบรรทัดนั้น `conda create`
> จะลง `~/.conda/envs` และ home เต็มทันที (env 3 ตัว ~8 GB + DB ~7 GB)

**ติดตั้ง** — รันบน login node ได้ (เป็นแค่การดาวน์โหลดและติดตั้ง ไม่ใช่การคำนวณ):

```bash
cd "$PROJECT"
bash server/00_setup_envs.sh 2>&1 | tee server/logs/setup.txt
```

ใช้เวลา ~30-60 นาที ส่วนใหญ่รอโหลด Platon DB (1.6 GB) กับ MOB-suite DB สคริปต์ทำ 6 ขั้น:

```
1/6  conda env: plasmid-ml   (python 3.11 + torch CUDA ที่ glibc รองรับ + pin อื่นตามที่ตีพิมพ์)
2/6  conda env: platon       (ลาก prodigal, diamond, hmmer, blast, mummer มาด้วย)
3/6  conda env: mobsuite     (ลาก blast+, mash)
4/6  Platon DB               ~2.8 GB หลังแตกไฟล์
5/6  MOB-suite DB            mob_init
6/6  ESM-2 weights           โหลดไว้ล่วงหน้าเผื่อ compute node ไม่มีเน็ต
```

เสร็จแล้วสคริปต์เขียน `server/plasmid-ml-server.lock.txt` (ผลของ `pip freeze`) —
**commit ไฟล์นี้** เพราะเป็นหลักฐานว่าตัวเลขบนเซิร์ฟเวอร์มาจาก torch/numpy เวอร์ชันไหน
ใช้อ้างในส่วน Methods ของ paper และอธิบายได้ว่าทำไม regression diff ถึงไม่เป็นศูนย์พอดี
(torch 2.5.1 บนเซิร์ฟเวอร์ vs 2.11 บนโน้ตบุ๊ก)

> conda env ทั้งหมดอยู่ที่ `$PROJECT/envs/` ชื่อ `plasmid-ml` (ไม่ใช่ `plasmid-ml-py311`
> ที่อาจเผลอสร้างด้วยมือตอน debug) — `env.sh` ชี้ `ENV_ML` ไปที่ `plasmid-ml` ถ้าสร้างด้วยมือ
> ชื่ออื่นไว้ ให้ลบทิ้ง (`conda env remove -p $PROJECT/envs/<ชื่อนั้น>`) แล้วให้สคริปต์สร้าง
> `plasmid-ml` เอง จะได้ไม่ต้องดูแล env สองชุดและ sbatch หา env เจอ

> **เรื่อง glibc กับ torch (สำคัญ — เคยติดมาแล้ว):** CU-AIM ใช้ glibc 2.17 ซึ่งเป็น
> เหตุผลเดียวกับที่ VS Code ต้องเป็น ≤ 1.85.2 (ขั้น 0.4) — torch ≥ 2.6 และ VS Code ≥ 1.86
> ต่างก็ต้องการ glibc ≥ 2.28 ถ้า `pip install torch` ดึงรุ่นล่าสุด (เช่น 2.11) มันจะลงได้
> แต่ **โหลดไม่ขึ้น** `00_setup_envs.sh` จึงอ่าน glibc จาก `ldd --version` แล้ว**cap torch
> ที่ 2.5.1** (รุ่นสุดท้ายที่ build สำหรับ glibc 2.17) โดยอัตโนมัติ และเลือก cuXXX ตาม driver
> จาก `nvidia-smi` — เวอร์ชันที่เลือกจริงถูกบันทึกไว้ที่ `$PROJECT/.cuda_tag`
>
> ถ้า `pip` บ่นว่า pin ตัวใดต้องการ torch ≥ 2.6 (เช่น transformers รุ่นใหม่) ให้ผ่อน pin
> ตัวนั้นใน `server/requirements-plasmid-ml.txt` — โค้ดโปรเจกต์ไม่ได้ใช้ API ที่เปลี่ยน

**ถ้าหยุดกลางทาง รันซ้ำได้เลย** ทุกขั้นข้ามถ้า**ติดตั้งสำเร็จและผ่านการตรวจแล้ว** (เขียน
`.setup_ok` ไว้) — env ที่ solve ค้างหรือ torch ลงไม่สำเร็จจะถูกสร้างใหม่ ไม่ใช่ข้ามไปเฉย ๆ
การดาวน์โหลด DB ก็ resume (`curl -C -`)

### สิ่งที่เห็นแล้วไม่ต้องตกใจ

| ข้อความ | แปลว่า |
|---|---|
| `torch.cuda.is_available() -> False` | **ปกติ** login node ไม่มี GPU การตรวจจริงอยู่ในขั้น 5 |
| `nvidia-smi not on the login node (expected)` | ปกติ — บน login node ไม่มี GPU สคริปต์ใช้ cuXXX ที่ดีสุดเท่าที่ glibc ยอมเป็น fallback แล้วไปยืนยัน driver จริงในขั้น GPU; เช็ก `.cuda_tag` ว่าได้อะไร |
| `lm_head.* UNEXPECTED` ตอนโหลด ESM-2 | ปกติ เราใช้แค่ encoder ไม่ใช้ language-model head |
| `NOTE: the Platon documentation states...` | เตือนให้เช็คว่า Platon กับ DB เป็นรุ่นที่เข้ากัน (ดู [ภาคผนวก D](#ภาคผนวก-d--db-ของ-platon-ไม่ตรงรุ่น)) |

---

## ขั้น 3 — ส่ง dataset ขึ้น (รันบน**โน้ตบุ๊ก** ไม่ใช่บนเซิร์ฟเวอร์)

เปิด terminal บนโน้ตบุ๊ก (Git Bash) ที่โฟลเดอร์โปรเจกต์ **ต่อ VPN ไว้**:

```bash
cd "d:/senior project"
CUAIM_USER=<user> \
CUAIM_KEY=~/.ssh/id_rsa_cuaim \
PROJECT=/data/project/<group>/<dir> \
bash server/01_sync_from_laptop.sh
```

ส่ง ~2.5 GB: FASTA + CSV ของ `dataset_mixed`, metadata, และผล baseline เดิมที่ต้องใช้เทียบ
พร้อมสร้าง `server/dataset_md5.txt` **จากเครื่องที่ผลิตข้อมูล** เพื่อให้ขั้น 4 ตรวจได้

> **ทำไมไม่ rebuild dataset บนเซิร์ฟเวอร์** — step01/02 ของ `dataset pipeline1` ดึงจาก NCBI
> ซึ่ง `assembly_summary.txt` เปลี่ยนทุกสัปดาห์ ถ้า rebuild จะได้ dataset ไม่เหมือนเดิมและ
> ตัวเลขทั้งหมดในรายงานเทียบกันไม่ได้อีก

> **ถ้า Git Bash ไม่มี `rsync`** สคริปต์จะใช้ `scp` ให้อัตโนมัติ แต่ `scp` **resume ไม่ได้**
> ถ้า VPN หลุดกลางทางต้องรันใหม่ทั้งหมด ถ้าเน็ตไม่นิ่งแนะนำติดตั้ง rsync (MSYS2 `pacman -S rsync`)
> แล้วรันอีกครั้ง — `rsync` มี `--partial` ต่อจากที่ค้างได้

---

## ขั้น 4 — ตรวจว่าข้อมูลมาครบ

```bash
ssh <user>@161.200.107.77 -i ~/.ssh/id_rsa_cuaim
cd "$PROJECT"                      # <-- ต้อง cd มาที่นี่ก่อน sbatch ทุกครั้ง
sbatch server/02_verify_data.sh
squeue -u "$USER"                  # รอจนหาย (ไม่กี่นาที)
cat server/logs/verify-data-*.out
```

ต้องเห็น 3 อย่างผ่าน:

```
--- 1. checksums against the manifest built on the laptop
    all dataset checksums match the laptop
--- 2. the dataset pipeline's own independent re-derivation
    (ผลจาก verify_dataset.py)
--- 3. the counts this plan is sized for
train        79,094 fragments |  ... 
validation   16,891 fragments |  ...
test         16,944 fragments |  116 assemblies |  129 replicons |   416 plasmid (2.46%) ...
dataset matches the counts the Slurm requests were sized for
  Platon / MOB-suite array size = 116 (one task per test assembly)
```

> **ถ้าข้อ 3 ไม่ผ่าน** แปลว่า dataset ไม่ใช่ชุดเดิม → **หยุด** เพราะ array size (116) และเวลาที่ขอ
> คำนวณมาจากตัวเลขเหล่านี้ ต้องแก้ `N_SHARDS` ใน `submit_all.sh` ก่อนไปต่อ

> **ถ้าข้อ 1 ไม่ผ่าน** (md5 ไม่ตรง) ไฟล์เสียระหว่างส่ง → รันขั้น 3 ซ้ำ

---

## ขั้น 5 — submit งานทั้งหมด

```bash
cd "$PROJECT"
DRY_RUN=1 bash server/submit_all.sh      # ดูคำสั่ง sbatch ก่อน ไม่ submit จริง
bash server/submit_all.sh                # submit จริง
```

จะเห็นลำดับ job id พร้อม dependency:

```
10_shard               job 123456
    shards   116 (default; manifest.csv not written yet)
11_platon[116]         job 123457
12_mobsuite[116]       job 123458
13_smoke_tools         job 123459
14_collect_score       job 123460
20_esm_embed           job 123461
21_esm_merge           job 123462
22_esm_train           job 123463
30_report              job 123464
```

```
10_shard ──┬─ 11_platon[1-116%5]   (cpu40h) ──┬─ 14_collect_score ─┐
           ├─ 12_mobsuite[1-116%5] (cpu40h) ──┤                    ├─ 30_report
           └─ 13_smoke_tools (ประตูกั้น) ──────┘                    │
20_esm_embed (gpu40g) ── 21_esm_merge ── 22_esm_train (gpu40g) ────┘
```

สอง chain (เครื่องมือ / ESM-2) **เดินขนานกัน** เพราะใช้ทรัพยากรต่างกัน

### ติดตามงาน

```bash
squeue -u "$USER" -o '%.18i %.9P %.12j %.8T %.10M %.9l %R %b'
```

อ่านคอลัมน์สุดท้าย (`R` = reason) เมื่อ state เป็น `PD` (pending):

| reason | แปลว่า |
|---|---|
| `Dependency` | รอ job ก่อนหน้าเสร็จ — **ปกติ** |
| `Resources` / `Priority` | คิวยาว รอได้ |
| `QOSMaxCpuPerUserLimit` | ชน 40 CPU ของ `cpu40h` — ปกติสำหรับ array ที่ throttle `%5` |
| `DependencyNeverSatisfied` | **job ก่อนหน้า fail** → ดู log แล้ว `scancel` ตัวที่ค้าง |

ดูผลย้อนหลัง / exit code:

```bash
sacct -j <id> --format=JobID,JobName%18,State,ExitCode,Elapsed
sacct -u "$USER" --start=$(date -d '1 day ago' +%F) -o JobID,JobName%18,State,Elapsed
```

`ExitCode 0:0` = ผ่าน อย่างอื่น = ดู log

### log อยู่ไหน

```
server/logs/shard-<jobid>.out              ขั้นแบ่ง shard
server/logs/platon-<arrayid>_<task>.out    1 ไฟล์ต่อ shard
server/logs/mobsuite-<arrayid>_<task>.out
server/logs/smoke-tools-<jobid>.out        ประตูกั้น (สำคัญที่สุด ดูข้อถัดไป)
server/logs/esm-embed-<jobid>.out          ภาพรวม
server/logs/esm-embed-<jobid>-<split>-shard<N>.out   ละเอียดต่อ shard
server/logs/esm-train-<jobid>.out
server/logs/report_<วันเวลา>.md             ผลลัพธ์สุดท้าย
```

### จุดที่ต้องดูให้แน่ใจว่าผ่าน

**`13_smoke_tools`** คือประตูกั้น ถ้า fail `14_collect_score` จะไม่รัน มันตรวจ 4 อย่าง:

```
1/4  สร้าง input ทดสอบ 20 fragment (plasmid 10 / chromosome 10)
2/4  platon รันได้ + header ของ TSV เป็นอย่างไร
3/4  mob_recon รันได้ + header ของ contig_report
4/4  กฎ accuracy ที่คำนวณย้อนหลัง == platon --mode accuracy ของจริง
```

ต้องเห็นท้ายสุด:

```
PASS: the re-derived 'accuracy' rule reproduces Platon's own classification exactly.
=== smoke test passed - both tools work and the single-pass Platon design holds
```

**`22_esm_train`** ตรวจ 2 อย่างก่อนเทรน:

```
--- component equivalence checks (batched vs per-fragment, GPU vs CPU)
    all component checks passed
--- regression check against the laptop's published numbers
    PASS: the refactored code reproduces the published numbers ...
```

> ขั้น 3 ส่ง `esm2-attention/cache_mixed/` (~42 MB) ขึ้นมาด้วยแล้ว regression check จึง
> **รันเป็นค่าตั้งต้น** — จำเป็นมากขึ้นบนเซิร์ฟเวอร์เพราะ torch คนละเวอร์ชัน (2.5.1 บน
> เซิร์ฟเวอร์ vs 2.11 บนโน้ตบุ๊ก) และ GPU คนละรุ่น บนโน้ตบุ๊กผมได้ **diff 0.0000** เพราะ
> torch เดียวกัน แต่**บนเซิร์ฟเวอร์อย่าคาดหวัง 0.0000** — floating point บน GPU/torch
> ต่างรุ่นไม่ deterministic การตรวจใช้ tolerance = seed spread ของรอบเดิม (floor 0.02) ซึ่ง
> ครอบส่วนต่างนี้ได้ ถ้าผ่านแปลว่าพอร์ตมาถูก ข้ามได้ด้วย `SKIP_REGRESSION=1` หลังผ่านแล้ว

### เวลาที่คาด

| job | QoS | เวลาที่คาด |
|---|---|---|
| `10_shard` | `cpu40h` | ~1 นาที (ทดสอบบนโน้ตบุ๊กแล้ว 3 วินาที) |
| `11_platon[116]` | `cpu40h`, 8 cpu × 5 ขนาน | ~1.5-2 ชม. |
| `12_mobsuite[116]` | `cpu40h`, 8 cpu × 5 ขนาน | ~1.5-2.5 ชม. |
| `13_smoke_tools` | `cpu40h` | ~20-40 นาที |
| `14_collect_score` | `cpu40h` | ~2 นาที |
| `20_esm_embed` | `gpu40g` | ~2-6 ชม. (8.79 M ORFs / 2.13 G กรดอะมิโน) |
| `21_esm_merge` | `cpu40h` | ~5-15 นาที (เขียน 5.2 GiB) |
| `22_esm_train` | `gpu40g` | ~2-5 ชม. (4 variant × 2 regime × 5 seed) |

เวลาของสองเครื่องมือและ GPU เป็น**ประมาณการ** ยังไม่ได้ยืนยันกับเครื่องจริง ถ้า job ชน time limit
ดู [ภาคผนวก B](#ภาคผนวก-b--job-ตายหรือชน-time-limit) — ออกแบบให้ resume ได้ทุกตัว

---

## ขั้น 6 — เก็บผล

`30_report.sh` เขียน `server/logs/report_<วันเวลา>.md` ซึ่งเป็นตารางเดียวที่มีทุก baseline

```bash
ls -t server/logs/report_*.md | head -1          # ไฟล์ล่าสุด
```

ดึงกลับมาที่โน้ตบุ๊ก (รันบนโน้ตบุ๊ก):

```bash
scp -i ~/.ssh/id_rsa_cuaim \
  '<user>@161.200.107.77:/data/project/<group>/<dir>/server/logs/report_*.md' .

# ไฟล์ผลลัพธ์ที่ควรเก็บทั้งหมด
rsync -av -e "ssh -i ~/.ssh/id_rsa_cuaim" \
  '<user>@161.200.107.77:/data/project/<group>/<dir>/baseline/results_mixed/' \
  ./baseline/results_mixed_server/
rsync -av -e "ssh -i ~/.ssh/id_rsa_cuaim" \
  '<user>@161.200.107.77:/data/project/<group>/<dir>/esm2-attention/results_server/' \
  ./esm2-attention/results_server/
```

ไฟล์สำคัญที่เกิดขึ้น:

| ไฟล์ | คืออะไร |
|---|---|
| `baseline/results_mixed/tool_baselines.csv` | ตารางคะแนน Platon (3 mode) + MOB-recon |
| `baseline/results_mixed/tool_baselines.json` | รายละเอียดครบ รวม breakdown ตาม length bucket |
| `baseline/results_mixed/predictions_platon_test.csv` | ผลระดับ fragment + RDS + hallmark ทุกคอลัมน์ |
| `baseline/results_mixed/predictions_mobsuite_test.csv` | ผลระดับ fragment + cluster + mash distance |
| `esm2-attention/results_server/esm2_variants.csv` | ESM-2 ladder × 2 regime × checkpoint |
| `esm2-attention/results_server/esm2_by_length.csv` | ESM-2 แยกตาม length bucket |
| `esm2-attention/results_server/summary.md` | เทียบ ESM-2 กับ baseline ทั้งหมด |
| `esm2-attention/results_server/checkpoints/*.pt` | น้ำหนักโมเดลที่เทรนได้ (ของเดิมไม่เคยเก็บ) |
| `server/logs/report_*.md` | ตารางรวมทุกอย่างเป็น markdown |

### อ่านตาราง `tool_baselines.csv` อย่างไร

สองคอลัมน์ที่ต้องระวัง ไม่ใช่เอาไปใส่รายงานตรง ๆ:

- **`hard_label_score=True`** → AUROC/AUPRC ของแถวนั้นมาจาก**การตัดสิน** ไม่ใช่ความน่าจะเป็น
  (mob_recon ไม่คืนค่าความน่าจะเป็นเลย) ตัวเลขจึงถูกครอบงำโดย label แถวที่เทียบ AUROC/AUPRC
  ได้อย่างเป็นธรรมคือ `platon_rds@*` เพราะ RDS เป็นคะแนนต่อเนื่องจริง
- **`mcc_oracle`** → threshold จูนบน **test split เอง** (เพราะเครื่องมือไม่ต้องเทรน เราจึงไม่ได้
  รันบน validation) เป็น **upper bound** ไม่ใช่จุดทำงานที่เลือกได้ล่วงหน้า

และ **`policy`** มี 3 แบบต่อเครื่องมือ: `chromosome` (unclassified → chromosome, ตัวเลข headline),
`plasmid` (upper bound ของ recall), `exclude` (ตัดออกแล้วรายงาน coverage แยก)

### สิ่งที่ต้องเขียนในรายงานพร้อมตัวเลขเหล่านี้

fragment เป็น **linear sub-sequence** ของ replicon จึงไม่มี circularity และมักขาด oriT/rep
hallmark ที่ Platon mode `accuracy` พึ่งพา ส่วน mob_recon ก็ตัด contig ที่สั้นกว่า 1000 bp ทิ้ง
→ ทั้งคู่จะเอียงไปทาง chromosome/unclassified **นี่เป็นผลที่ถูกต้องและน่าสนใจ ไม่ใช่บั๊ก**:
เครื่องมือที่สมมติว่า input คือ assembled contig ทำงานแย่ลงกับ fragment และนั่นคือเหตุผลที่
ต้องรายงาน**ระดับ replicon** (`replicon_mcc`) ควบคู่กับระดับ fragment

---

## ภาคผนวก A — รันทีละขั้นด้วยมือ

`submit_all.sh` แค่ร้อย `sbatch` ให้ ถ้าจะ submit เองต้องใส่ flag ให้ครบ เพราะ `#SBATCH`
ในไฟล์อ่านตัวแปรไม่ได้ (เป็นแค่ค่าตั้งต้น):

```bash
cd "$PROJECT" && source server/env.sh

# เครื่องมือ
sbatch --partition=$PART_CPU --qos=$QOS_CPU --account=$CUAIM_ACCOUNT \
       --cpus-per-task=2 --mem=8G --time=01:00:00 server/10_shard.sh

sbatch --partition=$PART_CPU --qos=$QOS_CPU --account=$CUAIM_ACCOUNT \
       --cpus-per-task=$CPUS_PER_TOOL --mem=$MEM_PER_TOOL --time=$TIME_PER_TOOL \
       --array=1-116%$MAX_PARALLEL server/11_platon.sh

# ESM-2
sbatch --partition=$PART_GPU --qos=$QOS_GPU --gres=$GRES_GPU --account=$CUAIM_ACCOUNT \
       --cpus-per-task=$GPU_CPUS --mem=$GPU_MEM --time=$TIME_EMBED server/20_esm_embed.sh
```

**รันแค่บางส่วน**

```bash
bash server/submit_all.sh tools     # แค่ Platon + MOB-suite
bash server/submit_all.sh esm       # แค่ ESM-2
```

**รัน array ซ้ำแค่บาง task** (เช่น task 7 กับ 23 ตาย)

```bash
sbatch --partition=$PART_CPU --qos=$QOS_CPU --cpus-per-task=$CPUS_PER_TOOL \
       --mem=$MEM_PER_TOOL --time=$TIME_PER_TOOL --array=7,23 server/11_platon.sh
```

**ทดลองแบบ interactive** (ดีบั๊ก ไม่ใช่รันงานจริง — QoS `interactive-gpu` ให้ 2 ชม.)

```bash
srun -p $PART_GPU -c 4 --mem=8G --qos=interactive-gpu --gres=gpu:1g.5gb:1 --pty bash -l
cd "$PROJECT" && source server/env.sh && source server/lib.sh
cuaim_modules gpu && cuaim_activate "$ENV_ML"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## ภาคผนวก B — job ตายหรือชน time limit

**หลักการ: ทุกขั้นที่แพง resume ได้ ให้ submit สคริปต์เดิมซ้ำ อย่าเริ่มใหม่**

| job | resume ได้ยังไง |
|---|---|
| `11_platon` / `12_mobsuite` | shard ที่มี output แล้วจะถูก skip → submit array เดิมซ้ำ |
| `20_esm_embed` | shard ที่ cache แล้วจะถูก skip → `sbatch` ไฟล์เดิมซ้ำ เดินต่อจาก shard ถัดไป |
| `21_esm_merge` | split ที่ merge แล้วจะถูก skip |
| `22_esm_train` | **resume ไม่ได้** ต้องรันใหม่ทั้งหมด (~2-5 ชม.) |

เช็คว่าเหลืออะไร:

```bash
cd "$PROJECT" && source server/env.sh
# ESM: shard ไหนยังไม่เสร็จ
"$ENV_ML/bin/python" server/_shard_plan.py \
  --config esm2-attention/config_server.yaml --pending-only
# เครื่องมือ: shard ไหนยังไม่มี output
comm -23 <(cut -d, -f1 tools-baseline/shards/manifest.csv | tail -n +2 | sort) \
         <(ls tools-baseline/out/platon 2>/dev/null | sort)
```

**ถ้า `20_esm_embed` ชน 2 วันซ้ำ ๆ** ขอเวลามากขึ้น (`gpu40g` ให้ถึง 8 วัน) แต่คู่มือ CU-AIM
เตือนว่าขอใหญ่รอคิวนานกว่า — ขอเพิ่มทีละขั้น:

```bash
sbatch --partition=$PART_GPU --qos=$QOS_GPU --gres=$GRES_GPU \
       --cpus-per-task=$GPU_CPUS --mem=$GPU_MEM --time=4-00:00:00 server/20_esm_embed.sh
```

**ยกเลิกงาน**

```bash
scancel -f <jobid>              # -f หยุด job step ลูกทั้งหมดด้วย
scancel -u "$USER"              # ยกเลิกทุก job ของเรา
```

---

## ภาคผนวก C — พื้นที่ไม่พอ

พื้นที่ที่ต้องใช้: dataset 2.5 + Platon DB 2.8 + MOB DB ~4 + embeddings **5.2** + conda envs ~8
+ log ~1 = **~24 GB**

> ตัวเลข embeddings วัดจากข้อมูลจริง: ปลด `max_orfs` ได้ **8.79 M ORFs** จาก 112,929 fragments
> → float16 = 5.2 GiB แต่ float32 = 10.5 GiB ดังนั้น `store_float16: true` ใน
> `config_server.yaml` **จำเป็น** ไม่ใช่ทางเลือก

ตัดตามลำดับนี้ (ผลกระทบน้อย → มาก):

1. **ลบ shard หลัง merge** — `21_esm_merge.sh` ทำให้อยู่แล้ว (ตั้ง `KEEP_SHARDS=1` ถ้าไม่อยากลบ)
2. **ลบ shard FASTA ของเครื่องมือ** หลัง `14_collect_score` ผ่าน — คืน 367 MB
   ```bash
   rm -rf "$PROJECT/tools-baseline/shards"     # สร้างใหม่ได้ด้วย 10_shard.sh
   ```
3. **ลบ Platon DB** หลัง array เสร็จ — คืน 2.8 GB (เก็บ TSV ไว้แล้ว)
4. **ลด train set** — ใน `esm2-attention/config_server.yaml` เปลี่ยน
   ```yaml
   subsample: null
   ```
   เป็น
   ```yaml
   subsample:
     train: {plasmid: 2200, chromosome: 17800}      # ~20,000 จาก 79,094
     validation: {plasmid: 999999, chromosome: 999999}   # เต็ม split
     test: {plasmid: 999999, chromosome: 999999}         # เต็ม split
   ```
   (`999999` = "เอาเท่าที่มี" เพราะโค้ดใช้ `min(wanted, len(pool))` และ weight จะเป็น 1.0)
   ยังปลดคอขวด `max_orfs` และ class-ratio ได้ครบ เสียแค่ขนาด train set — ประหยัด
   embeddings ไปประมาณครึ่ง (train คิดเป็น ~70% ของ fragment ทั้งหมด)
5. **ใช้ Singularity แทน conda** — ประหยัด ~6 GB ตั้ง `TOOL_RUNTIME=singularity` ใน `env.sh`
   แล้วดึง image (ดูคู่มือ CU-AIM บทที่ 6):
   ```bash
   singularity pull "$PROJECT/containers/platon.sif" \
     docker://quay.io/biocontainers/platon:<tag>
   ```

---

## ภาคผนวก D — DB ของ Platon ไม่ตรงรุ่น

เอกสาร Platon ระบุชัดว่า **Platon รุ่นเก่าอ่าน DB รุ่นใหม่ไม่ได้** `00_setup_envs.sh` จึงบันทึก
เวอร์ชันไว้ข้าง DB:

```bash
cat "$PLATON_DB/PROVENANCE.txt"
```

ถ้าเวอร์ชันไม่เข้ากัน (อาการ: Platon fail ตอนอ่าน DB หรือ RDS เป็น NaN ทุก contig) ให้เช็ค
URL ของ DB รุ่นล่าสุดจาก https://github.com/oschwengers/platon แล้ว:

```bash
rm -rf "$PLATON_DB"
PLATON_DB_URL=<url ใหม่> bash server/00_setup_envs.sh
```

---

## ภาคผนวก E — ปัญหาที่พบบ่อย

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| `ssh` ค้าง / timeout | VPN ไม่ได้ต่อ | ต่อ Cisco AnyConnect |
| `Permission denied (publickey)` | key ยังไม่ถูกอนุมัติ หรือใส่ `-i` ผิดไฟล์ | ส่ง `.pub` ให้ผู้ดูแล / เช็ค path |
| `sbatch: error: Batch job submission failed: Invalid qos specification` | QoS ที่ใส่บัญชีไม่มีสิทธิ์ | `check_qos` แล้วแก้ `env.sh` |
| `slurmstepd: error: execve(): ... No such file` | submit จากโฟลเดอร์อื่น | `cd "$PROJECT"` ก่อน `sbatch` ทุกครั้ง |
| job หายไปเฉย ๆ ไม่มี log | โฟลเดอร์ `server/logs/` ไม่มี | `mkdir -p server/logs` (ปกติมีมาจาก git แล้ว) |
| `Disk quota exceeded` | ลืม `source server/env.sh` → cache ลง `/data/home` | ดู [ภาคผนวก C](#ภาคผนวก-c--พื้นที่ไม่พอ) + `du -sh ~/.conda ~/.cache` |
| `torch.cuda.is_available() is False` ใน job ที่ขอ GPU | wheel ไม่ตรง driver | `cat "$PROJECT/.cuda_tag"` แล้วติดตั้ง torch ใหม่ตาม `nvidia-smi` |
| `FATAL: ... missing required column(s)` | เครื่องมือเปลี่ยน schema ของ output | เทียบ `server/logs/tool_help_*.txt` แล้วเติม alias ใน `tools-baseline/toolbase/parse.py` |
| `FAIL: the following contigs were labelled differently` | กฎ Platon ที่คำนวณย้อนหลังไม่ตรงของจริง | แก้ `11_platon.sh` ใช้ `--mode accuracy` แทน `--characterize` แล้วให้คะแนนจาก FASTA ของ Platon ตรง ๆ |
| `FATAL: ... shard(s) produced no Platon TSV` | บาง array task ตาย | ดู log ของ task นั้น แล้ว resubmit แค่ task นั้น ([ภาคผนวก A](#ภาคผนวก-a--รันทีละขั้นด้วยมือ)) |
| `one or more splits could not be merged` | embedding ยังไม่ครบทุก shard | `sbatch server/20_esm_embed.sh` ซ้ำ |
| `FAIL: the refactor changed the published numbers` | โค้ดถูกแก้จนผลเปลี่ยน | **หยุด** อย่าตีความผลเป็นผลการทดลอง ดู `esm2/model.py`, `train_eval.py` เส้นทาง `batch_size==1` |

---

## เช็กลิสต์

ครั้งแรก:

- [ ] ต่อ VPN ได้ / `ssh` เข้าได้
- [ ] `probe_server.sh` ผ่าน จดค่าครบ 7 ตัว (รวม MODULE_CUDA) + glibc บอก torch รุ่นไหน
- [ ] พื้นที่ `/data/project` ≥ 24 GB
- [ ] `check_qos` มี `cpu40h` และ `gpu40g` (หรือแก้ `env.sh` ตามที่มี)
- [ ] `MaxArraySize` ≥ 116
- [ ] แก้ `env.sh` แล้ว (`PROJECT`, `CUAIM_ACCOUNT`, partition, GRES, module)
- [ ] `00_setup_envs.sh` จบครบ 6/6
- [ ] `01_sync_from_laptop.sh` ส่งครบ
- [ ] `02_verify_data.sh` ผ่านทั้ง 3 ข้อ (16,944 fragments / 116 assemblies)

ทุกรอบที่รัน:

- [ ] `cd "$PROJECT"` ก่อน `sbatch`
- [ ] `13_smoke_tools` ขึ้น `PASS` (ถ้าไม่ผ่าน ผลของ Platon ใช้ไม่ได้)
- [ ] `22_esm_train` ขึ้น `all component checks passed`
- [ ] ไม่มี job ไหน `State` เป็น `FAILED` / `TIMEOUT` ใน `sacct`
- [ ] `collect_*` ไม่เตือนเรื่อง shard หาย
- [ ] ได้ `server/logs/report_*.md` แล้ว copy กลับโน้ตบุ๊ก
