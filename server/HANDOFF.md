# Handoff — บริบทสำหรับถาม Claude ต่อบนเครื่องอื่น

> **วิธีใช้:** บนเครื่องที่รันงาน server อยู่ `git pull` แล้วเปิด Claude Code ในโฟลเดอร์
> โปรเจกต์ บอกว่า *"อ่าน server/HANDOFF.md กับ server/MANUAL.md แล้วช่วยดู error นี้ให้หน่อย"*
> แล้ววาง error มาได้เลย — Claude จะมี context พอช่วยดีบั๊กโดยไม่ต้องเล่าใหม่
>
> ไฟล์นี้ sync ผ่าน git เครื่องไหนก็เหมือนกัน แก้แล้ว commit/push เพื่อให้อีกเครื่องได้ด้วย

อัปเดตล่าสุด: commit `2b6d440` (2026-10-26)

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

## ลำดับงานบนเซิร์ฟเวอร์

```
0. git pull                                          # เอาโค้ดล่าสุด
1. bash server/probe_server.sh | tee server/logs/probe.txt   # แล้วกรอก env.sh
2. bash server/00_setup_envs.sh                      # conda envs + DB + ESM weights
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
- [ ] `00_setup_envs.sh` รันครบ 6/6 บนเซิร์ฟเวอร์  ← **กำลังอยู่ขั้นนี้**
- [ ] sync dataset + `02_verify_data.sh` ผ่าน
- [ ] `submit_all.sh` + ได้ `report_*.md`

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
