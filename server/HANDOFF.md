# Handoff — บริบทสำหรับถาม Claude ต่อบนเครื่องอื่น

> **วิธีใช้:** บนเครื่องที่รันงาน server อยู่ `git pull` แล้วเปิด Claude Code ในโฟลเดอร์
> โปรเจกต์ บอกว่า *"อ่าน server/HANDOFF.md กับ server/MANUAL.md แล้วช่วยดู error นี้ให้หน่อย"*
> แล้ววาง error มาได้เลย — Claude จะมี context พอช่วยดีบั๊กโดยไม่ต้องเล่าใหม่
>
> ไฟล์นี้ sync ผ่าน git เครื่องไหนก็เหมือนกัน แก้แล้ว commit/push เพื่อให้อีกเครื่องได้ด้วย

อัปเดตล่าสุด: 2026-10-26 (หลังรอบตรวจ MANUAL ↔ `survival-manual-cuaim.pdf`)

---

## กำลังทำอะไรอยู่

เพิ่ม baseline **Platon + MOB-suite** และรัน **ESM-2 แบบปลดคอขวด** บนคลัสเตอร์ **CU-AIM**
(Slurm + A100 MIG, glibc 2.17) งานทั้งหมดอยู่บน GitHub:
`Professor-Shin/plasmid-class-private` (private)

- รายละเอียดการรันทีละขั้น: **`server/MANUAL.md`**
- เหตุผลการออกแบบ: **`server/README.md`**
- แผนเต็ม (ไทย): `C:\Users\USER\.claude\plans\now-i-can-access-unified-whale.md` (ไฟล์นี้อยู่นอก repo)

## สภาพแวดล้อม CU-AIM ที่ยืนยันแล้ว

| | ค่า |
|---|---|
| เข้าถึง | VPN (Cisco AnyConnect) → `ssh <user>@161.200.107.77 -i ~/.ssh/id_rsa_cuaim` |
| PROJECT | `/data/project/naruemon/siwakorn-s_plasmid_class/plasmid-class-private` (= repo ที่ clone) |
| glibc | **2.17** — cap torch ที่ **2.5.1** (ยืนยันแล้วว่า `torch 2.5.1+cu124` โหลดขึ้น) |
| driver CUDA | **12.4** → ใช้ wheel `cu124` |
| python ของ env ML | **3.11** (ไม่ใช่ 3.13) |
| QoS | `cpu40h` (40 CPU) สำหรับ Platon/MOB, `gpu40g` (1×7g.40gb, 16 CPU) สำหรับ ESM-2 |
| conda env | อยู่ที่ `$PROJECT/envs/` ชื่อ **`plasmid-ml`**, `platon`, `mobsuite` |

## กับดักที่เจอมาแล้ว (ถ้า error ให้เช็กพวกนี้ก่อน)

1. **glibc 2.17 ↔ torch** — `pip install torch` รุ่นล่าสุด (≥2.6) ลงได้แต่**โหลดไม่ขึ้น** เพราะ
   เป็น manylinux_2_28 ต้อง cap ที่ 2.5.1 (`00_setup_envs.sh` ทำให้อัตโนมัติจาก `ldd --version`)
   เหตุผลเดียวกับที่ VS Code ต้อง ≤ 1.85.2
2. **venv ไม่ช่วยเรื่อง glibc** — glibc เป็น C library ของระบบ ใช้ร่วมกันหมด เราใช้ conda เพราะ
   Platon/MOB ต้องการ binary (prodigal, blast, diamond, hmmer, mash) ที่ pip/venv ลงไม่ได้
3. **env ชื่อ `plasmid-ml`** เท่านั้น — ถ้าเผลอสร้าง `plasmid-ml-py311` ด้วยมือ ให้ลบ แล้วให้
   `00_setup_envs.sh` สร้างเอง ไม่งั้น sbatch หา env ไม่เจอ
4. **`/data/home` โควต้าเล็ก** — `env.sh` export `CONDA_ENVS_DIRS`/`HF_HOME`/`TMPDIR`/`PIP_CACHE_DIR`
   ไป `$PROJECT` ถ้าลืม source env.sh → home เต็ม (`Disk quota exceeded`)
5. **PROJECT = repo** — `envs/ db/ .hf/ tmp/` อยู่ใน git tree แต่ gitignore ไว้แล้ว **อย่าลบ rule
   พวกนี้** ไม่งั้น `git add -A` จะดูด conda env หลาย GB เข้า git
6. **`cd "$PROJECT"` ก่อน `sbatch` เสมอ** — error `tee: ... No such file` = อยู่ผิด directory
7. **`00_setup_envs.sh` ต้องรันใน allocation ไม่ใช่บน login node เปล่า ๆ** — conda classic
   solver กิน RAM หลาย GB, login node ฆ่าทิ้งเป็น `Killed` / exit 137 (เจอตอน solve
   `mob_suite` บน `cuaim-frontend1`) ขอ `srun -p $PART_CPU --qos=$QOS_CPU -c 4 --mem=16G
   -t 2:00:00 --pty bash -l` ก่อน หรือลง `conda-libmamba-solver` ใน base — คู่มือ CU-AIM §2.1
   กัน login node ไว้ให้จัดการไฟล์/submit เท่านั้น
8. **job script ต้องขึ้นต้น `#!/bin/bash -l`** — `module` เป็น shell function จาก
   `/etc/profile.d` ซึ่ง non-login shell ไม่ source (ทุก listing ในคู่มือ CU-AIM ใช้ `-l`)
   `lib.sh` มี `cuaim_init_modules` กู้จาก `$MODULESHOME` ให้อีกชั้นแล้ว

## ลำดับงานบนเซิร์ฟเวอร์

```
0. git pull                                          # เอาโค้ดล่าสุด
1. bash server/probe_server.sh | tee server/logs/probe.txt   # แล้วกรอก env.sh
2. srun -p $PART_CPU --qos=$QOS_CPU -c 4 --mem=16G -t 2:00:00 --pty bash -l
   bash server/00_setup_envs.sh                      # conda envs + DB + ESM weights
                                                     # (ใน allocation! login node จะ OOM-kill)
3. (บนโน้ตบุ๊ก) bash server/01_sync_from_laptop.sh   # dataset ~2.5 GB
4. sbatch server/02_verify_data.sh                   # ตรวจข้อมูล
5. bash server/submit_all.sh                         # รันทั้ง pipeline
6. (ผลอยู่ใน server/logs/report_*.md)
```

job graph:
```
10_shard ─┬─ 11_platon[1-116%5]   (cpu40h) ─┬─ 14_collect_score ─┐
          ├─ 12_mobsuite[1-116%5] (cpu40h) ─┤                    ├─ 30_report
          └─ 13_smoke_tools (gate) ─────────┘                    │
20_esm_embed (gpu40g) ── 21_esm_merge ── 22_esm_train (gpu40g) ──┘
```

## สถานะ ณ ตอนนี้ (อัปเดตเมื่อคืบหน้า)

- [x] เขียนสคริปต์ + MANUAL ครบ, push แล้ว (commit `2b6d440`)
- [x] glibc/torch fix ยืนยันบนเครื่องจริง (`torch 2.5.1+cu124` โหลดได้)
- [x] ตรวจ MANUAL เทียบ `survival-manual-cuaim.pdf` + แก้ที่ขัดกัน (login node / `-l` /
      MODULE_CUDA / เหตุผล singularity)
- [x] `00_setup_envs.sh` รันครบ **6/6** (2026-10-26 07:57) — envs `plasmid-ml` 5.8G /
      `platon` 1.1G / `mobsuite` 5.0G, DB `platon` 2.8G / `mobsuite` 2.9G, ESM weights 160M
      - ทางที่ติดและวิธีแก้: OOM-kill ตอน solve บน login node → ใช้ micromamba ใน
        `$PROJECT/bin`; `mob_init` fail เพราะ `$ENV_MOB/bin` ไม่อยู่ใน PATH (commit `b72cd7a`)
      - **ค้าง: ลบ env `plasmid-ml-py311` (5.3 GB) ที่เผลอสร้างด้วยมือ** — ดูกับดักข้อ 3
- [x] dataset พร้อมแล้ว (2026-10-28) — โน้ตบุ๊กที่ผลิตข้อมูลใช้ไม่ได้ จึง **rebuild FASTA จาก
      NCBI** ด้วย `server/01b_rebuild_fasta_from_ncbi.py`: 112,929 fragments จาก 908 replicon
      / 500 assembly ทุก replicon ผ่าน sha256 ของ `replicons.csv` และทุก fragment ผ่าน
      length/GC/non-ACGT ของ split CSV
      - 2 assembly ถูกถอนจาก RefSeq ระหว่างทาง (`GCF_041228055.1`, `GCF_049902035.1`)
        กู้คืนจาก GenBank/ENA ด้วย accession INSDC (ตัด `NZ_`) — sha256 ตรงทั้งหมด
      - `verify_dataset.py` ผ่าน 21/21 ข้อ รวมข้อ re-derive จาก genome ต้นทาง
      - `_dataset_facts.py`: test 16,944 fragments / 116 assemblies / 129 replicons ✓
      - **ยังอย่าลบ `$PROJECT/tmp/ncbi_replicons/` (0.7 GB)** จนกว่า `02_verify_data.sh`
        แบบ sbatch จะผ่าน — check 7 ใช้แทน `source_fasta` ที่เป็น path ของโน้ตบุ๊ก
      - ที่หายไปกู้ไม่ได้: `esm2-attention/cache_mixed` (42 MB) → `22_esm_train` จะข้าม
        regression check เองและบอกเหตุผล
- [ ] `submit_all.sh` + ได้ `report_*.md`  ← **กำลังอยู่ขั้นนี้**
      **เครื่อง Linux (`/media/siwakorn/YUTHAPONG/plasmid-class-private`) ทำขั้นนี้ไม่ได้**
      มีแค่ CSV ที่ tracked — ขาด `train/validation/test.fasta` (~2.4 GB) และ
      `esm2-attention/cache_mixed` (42 MB) ซึ่งเป็น generated data ที่ gitignore ไว้
      → ต้องรันจากโน้ตบุ๊ก Windows `d:/senior project` ที่ผลิตข้อมูล
      → ถ้าเครื่องนั้นใช้ไม่ได้: CSV มี `sequence_id` (accession มี version) + `start`/`end`
        ครบทุก fragment จึง refetch จาก NCBI ทีละ accession แล้วตัดใหม่ได้ byte-identical
        (accession.version เป็น immutable — ต่างจากการรัน step01/02 ใหม่ที่จะสุ่ม genome ชุดใหม่)
- [ ] sync dataset + `02_verify_data.sh` ผ่าน

> อัปเดต checkbox ข้างบนเมื่อทำคืบหน้า แล้ว commit — จะได้รู้ว่าค้างตรงไหนเวลาสลับเครื่อง

## งานวิทยาศาสตร์ที่ค้างไว้ (ยังไม่ทำ รอหลังได้ผลจริง)

จาก review — 3 ข้อที่ควรทำเพื่อ paper (รายละเอียดในประวัติ chat):
1. **bootstrap CI ระดับ assembly** สำหรับ replicon MCC (test มี plasmid replicon 54 ตัว ยังน้อย)
2. **เช็ก DB leakage** — Platon/MOB DB สร้างจาก RefSeq อาจมี test genomes ปน ดู `mash_neighbor_distance ≈ 0`
3. **homology distance** — split คุมที่ Mash-ANI ≥ 0.95 แล้ว แต่ family-level (<95%) ยังควรวัด NN distance test↔train

## วิธีต่อ "ทั้งบทสนทนา" จริง ๆ (ถ้าต้องการ ไม่ใช่แค่ context)

ไฟล์ transcript เต็มอยู่ที่ (เครื่องโน้ตบุ๊กเดิมเท่านั้น):
`C:\Users\USER\.claude\projects\d--senior-project\8674eba7-b572-4328-b3b2-3ea800ea7742.jsonl`
ก็อปไปวางที่ path เดียวกันบนเครื่องใหม่ (+ โฟลเดอร์ `memory/`) แล้ว `claude --resume` ได้ แต่ path
โปรเจกต์ต้องตรงกันและห้ามเปิด 2 เครื่องพร้อมกัน — ส่วนใหญ่ไม่จำเป็น handoff doc นี้พอ

## ค่าจริงของ CU-AIM ที่ probe ยืนยันแล้ว (2026-10-26)

| | ค่า |
|---|---|
| account / user | `g_naruemon` / `naruemonp` |
| partition | `cpu`, `gpu`, `interactive-cpu`, `interactive-gpu` |
| QoS ที่บัญชีมี | `cpu40h`, `gpu10gh`, `gpu20gh`, `gpu40g`, `gpu40g24m`, `interactive-gpu` (**ไม่มี** `interactive-cpu`) |
| `cpu40h` | 40 CPU / 1 TB / **4 วัน** |
| `gpu40g` | 16 CPU / 1×7g.40gb / 64 GB / **8 ชั่วโมง** ← คู่มือเขียน 8 วัน **ผิด** |
| `gpu10gh` | 16 CPU / 1×2g.10gb / 64 GB / **4 วัน** (ทางเลือกเมื่องานยาวเกิน 8 ชม.) |
| `MaxArraySize` | 1001 (ต้องการ ≥116 ✓) |
| module ที่มี | `Anaconda3/2021.11` (conda 4.x, **ไม่มี libmamba**), `CUDA/11.8.0` (ไม่มี 12.x), `Apptainer/1.3.2`, `/usr/bin/singularity` |
| `mamba` | ไม่มีในระบบ → ลง micromamba ใน `$PROJECT/bin` (MANUAL ขั้น 2) |
| node | cpu: `cuaim-fat-[01-02]` (72+ CPU, 690 GB) · gpu: `cuaim-dgx-01` (A100 MIG 7g.40gb×5, 3g.20gb×6) |
| เน็ตจาก login node | zenodo / huggingface / anaconda ได้ครบ ✓ |
| พื้นที่ | `/data` เหลือ 914 TB ไม่มี quota ต่อ dir |

กับดักเพิ่ม:
9. **ค่าของเครื่องอยู่ใน `server/env.local.sh`** (gitignore) ไม่ใช่ `env.sh` — `git pull` จะไม่ชน
10. **`env.sh` ใช้ `: "${VAR:=...}"`** ค่าที่ export ค้างใน shell จะไม่ถูกทับเมื่อ source ซ้ำ
    → `unset` ก่อน หรือเปิด shell ใหม่
11. **`CUAIM_ACCOUNT` ว่าง ≠ ไม่ใส่** — `--account=` ว่างเปล่า Slurm ปฏิเสธ
12. **push จากเซิร์ฟเวอร์ไม่ได้** (credential ในบัญชีร่วมเป็นของ user `nanghtet` อ่านได้เขียนไม่ได้
    และไม่ควรเอา token ตัวเองไปวาง) ให้ commit บนเซิร์ฟเวอร์ แล้วดึงจากเครื่องตัวเองด้วย
    `git fetch ssh://naruemonp@161.200.107.77/<PROJECT path> main:<branch>` แล้ว push จากเครื่องนั้น
13. **`srun` ในลูป `while read` กิน stdin** — `srun` ส่ง stdin ต่อให้ task แล้วอ่านจนหมด
    ถ้ารายการที่วนอยู่บน stdin (`done <<<"$LIST"`) ลูปจะจบหลังรอบแรก **แล้ว job exit 0**
    เหมือนทำงานครบ (เจอกับ `20_esm_embed.sh`: ทำ 1 จาก 18 shard แล้ว merge ก็ fail ตาม)
    → อ่านรายการผ่าน fd 3 (`read -u 3` + `done 3<<<`) และใส่ `</dev/null` ให้ `srun`
    **เวลาสงสัยว่างานไม่ครบ ให้นับรอบจาก log อย่าดูแค่ exit code**
14. **job array ชน `QOSMaxSubmitJobPerUserLimit`** — `cpu40h` นับทุก element เป็น 1 job
    116 element จึงถูกปฏิเสธ → ใช้ `TOOLS_MODE=serial` หรือ `server/15_tools_serial.sh`
    ซึ่งวนทุก shard ใน job เดียว (5 ขนาน × 8 threads = 40 CPU เท่าเดิม)
15. **job หนึ่งจอง 40 CPU = เพดานทั้งหมด** → อีก job ขึ้น `QOSMaxCpuPerUserLimit` และรอ
    เป็นพฤติกรรมปกติ ไม่ใช่ error (platon กับ mobsuite จึงเดินทีละตัว)
