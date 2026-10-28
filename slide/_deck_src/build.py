"""Build the CU-AIM progress deck on the Siwak System CI.

Starts from the previous deck's package so the embedded Thai fonts, the theme
and the master come along unchanged, then replaces every slide.
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci import (BODY_Y, BLUE, FOOT_Y, GREEN, GREEN_D, GREY, INK, KANIT, L, LINE, MONO,
                MUTED, ORANGE, R, TAG_H, TAG_Y, THAI, TINT_B, TINT_G, TINT_N, TINT_O,
                TINT_Y, W, WHITE, YELLOW, card, header, para, pic, pill, run, shape,
                tags, text, title)
import blocks as B

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WORK = Path(sys.argv[3])

LOGO, SHIELD = "rId3", "rId4"      # media/image6.png, media/image49.png
TOTAL = 15
slides = []


def add(caption, tag_l, tag_r, body, notes, cover=False, foot=""):
    # The footer tags are set in JetBrains Mono, which carries no Thai glyphs, so
    # they are written out in ASCII per slide rather than derived from the caption.
    slides.append({"caption": caption, "tl": tag_l, "tr": tag_r, "foot": foot,
                   "body": body, "notes": notes, "cover": cover})


# ── 01 cover ─────────────────────────────────────────────────────────────────
add("", "[SYS_v2.0 // COVER]", "[GRID_ALIGN_OK]", [
    shape(7115175, 1717675, 4260850, 3422650, geom="roundRect", fill=WHITE,
          adj=5340, line_clr="DADCE0", dash="dash"),
    pic(SHIELD, 8293100, 2286000, 1905000, 1905000, "โลโก้ SIWAK"),
    text(7115175, 4775200, 4260850, TAG_H,
         [para([run("SPECIFICATION SHIELD // CU-AIM_RUN", 1200, GREY, MONO)], algn="ctr")]),
    pill(L, 1561465, 3327400, 434340, TINT_G,
         [run("● ", 1300, GREEN, THAI, b=1),
          run("FULL-SCALE RUN // CU-AIM", 1300, GREEN_D, THAI, b=1)]),
    text(L, 2154819, 6540600, 1599000,
         [para([run("ยกงานขึ้น HPC", 5200, INK, KANIT, b=1)], line=105000),
          para([run("แล้วรันเต็มสเกล", 5200, INK, KANIT, b=1)], line=105000)]),
    text(L, 4077335, 6278880, 711200,
         [para([run("ปลดคอขวด ORF · เทรนบนข้อมูล", 1800, MUTED, THAI),
                run("ทั้งหมด", 1800, GREEN_D, THAI, b=1),
                run(" · เพิ่มคู่เทียบที่คนใช้งานจริง", 1800, MUTED, THAI)], line=145000),
          para([run("สถานะ ณ 28 ต.ค. 2026 — รันเต็มสเกลจบครบทุกขั้นแล้ว", 1800, MUTED, THAI)],
               line=145000)]),
    shape(L, 5017135, 609600, 76200, geom="roundRect", fill=GREEN, adj=50000),
    shape(1498600, 5017135, 609600, 76200, geom="roundRect", fill=BLUE, adj=50000),
    shape(2184400, 5017135, 254000, 76200, geom="roundRect", fill=ORANGE, adj=50000),
    shape(2514600, 5017135, 254000, 76200, geom="roundRect", fill=YELLOW, adj=50000),
], "สไลด์ชุดนี้เป็นความคืบหน้าต่อจากรอบตอบความเห็นผู้เชี่ยวชาญ รอบที่แล้วฝั่งชุดข้อมูลแน่นแล้ว "
   "แต่ฝั่งโมเดลยังรันบนข้อมูลแค่ 1.8% รอบนี้คือการยกฝั่งโมเดลขึ้นมาอยู่ระดับเดียวกัน "
   "บนคลัสเตอร์ CU-AIM ย้ำกับผู้ฟังว่าผลโมเดลยังรันไม่เสร็จ ตัวเลขที่โชว์วันนี้คือสเกลและทรัพยากร ไม่ใช่คะแนน",
    cover=True)

# ── 02 what changed ──────────────────────────────────────────────────────────
add("สามเรื่องที่เปลี่ยน", "[SYS_v2.0 // SLIDE_02]", "[3_CHANGES]", [
    title("เปลี่ยนสามเรื่อง ", "ที่แพงเกินจะทำบนโน้ตบุ๊ก"),
    *B.cards_row([
        {"tag": "01", "color": ORANGE, "head": "ปลดเพดาน ORF",
         "body": "เดิมเก็บเฉพาะ ORF ที่ยาวที่สุด 10 ตัว ชิ้นยาวจึงถูกตัดทิ้งเกือบหมด "
                 "รอบนี้เก็บทุกตัวที่เรียกได้",
         "note": ("ผลคือ", "10 → ~78 ORF ต่อชิ้น")},
        {"tag": "02", "color": BLUE, "head": "เทรนบนข้อมูลทั้งหมด",
         "body": "เดิมสุ่มมา 2,000 ชิ้น และบังคับให้มีพลาสมิด 45% ซึ่งไม่ตรงกับตอนวัดผล",
         "note": ("ผลคือ", "79,094 ชิ้น · พลาสมิด 2.8% ตามจริง")},
        {"tag": "03", "color": GREEN, "head": "เทียบกับเครื่องมือจริง",
         "body": "เดิมเทียบกับ baseline ที่เราเขียนเอง รอบนี้รัน Platon และ MOB-recon "
                 "บนชุดทดสอบเดียวกัน",
         "note": ("ผลคือ", "116 shard ต่อเครื่องมือ")},
    ], BODY_Y, 3352800),
    *B.note_bar(5562600, [run("ทั้งสามข้อคือลำดับ 1–3 ในตาราง tuning ของรอบที่แล้ว ", 1300, MUTED, THAI),
                          run("ข้อที่เขียนไว้ว่า “ต้องขึ้น HPC”", 1300, INK, THAI, b=1)]),
 ], "สามข้อนี้ไม่ใช่ของใหม่ที่เพิ่งคิด แต่เป็นสามข้อแรกในตาราง tuning ที่เราจัดลำดับไว้เองเมื่อรอบที่แล้ว "
    "และทั้งสามข้อถูกระบุว่าต้องใช้ HPC จึงทำไม่ได้บนโน้ตบุ๊ก",
    foot="[3_CHANGES // ALL_FROM_TUNING_TABLE]")

# ── 03 before / after table ──────────────────────────────────────────────────
add("ตัวเลขก่อน–หลัง", "[SYS_v2.0 // SLIDE_03]", "[SCALE_DELTA]", [
    title("สเกลที่ต่างกัน ", "คนละระดับ"),
    *B.table_card(BODY_Y, 3352800,
                  [("รายการ", "l"), ("รอบที่แล้ว (โน้ตบุ๊ก)", "l"), ("รอบนี้ (CU-AIM)", "l"),
                   ("เท่า", "r")],
                  [[("ชิ้นส่วนที่ใช้เทรน", INK, 1), ("2,000", MUTED, 0), ("79,094", GREEN_D, 1), ("39.5×", MUTED, 0)],
                   [("ชิ้นส่วนที่ใช้ทดสอบ", INK, 1), ("1,016", MUTED, 0), ("16,944 เต็ม", GREEN_D, 1), ("16.7×", MUTED, 0)],
                   [("ORF ต่อชิ้น", INK, 1), ("10 (ชนเพดาน)", MUTED, 0), ("~78 (ไม่จำกัด)", GREEN_D, 1), ("7.8×", MUTED, 0)],
                   [("พลาสมิดตอนเทรน", INK, 1), ("45%", ORANGE, 0), ("2.8% = ของจริง", GREEN_D, 1), ("—", MUTED, 0)],
                   [("seed", INK, 1), ("2", MUTED, 0), ("5", GREEN_D, 1), ("2.5×", MUTED, 0)],
                   [("epoch", INK, 1), ("15 คงที่", MUTED, 0), ("≤60 + early stop", GREEN_D, 1), ("—", MUTED, 0)],
                   [("checkpoint", INK, 1), ("ไม่เก็บเลย", ORANGE, 0), ("เก็บทุกตัว", GREEN_D, 1), ("—", MUTED, 0)]],
                  [3505200, 2819400, 2667000, 965200]),
    *B.note_bar(5562600, [run("จุดที่มักถูกมองข้าม: ", 1300, ORANGE, THAI, b=1),
                          run("ตาราง ESM ของรอบที่แล้ววัดบน test แค่ 1,016 ชิ้น ไม่ใช่ 16,944 "
                              "— รอบนี้จึงเทียบกับ k-mer baseline ได้ตรง ๆ เป็นครั้งแรก", 1300, MUTED, THAI)],
                 fill=TINT_O, h=548640),
 ], "แถวที่อยากให้จำคือสองแถวบน ของเดิมไม่ใช่แค่เทรนน้อย แต่วัดผลบน test เพียง 1,016 ชิ้นด้วย "
    "เพราะ config เดิมสุ่ม test ลงมาเหลือ 600 โครโมโซมบวกพลาสมิดทั้งหมด รอบนี้ใช้ test เต็ม "
    "เลขจึงเทียบกับ composition baseline ได้โดยตรง",
    foot="[TRAIN 39.5x // TEST 16.7x]")

# ── 04 ORF bottleneck released ───────────────────────────────────────────────
add("คอขวดที่ปลดแล้ว", "[SYS_v2.0 // SLIDE_04]", "[max_orfs = null]", [
    title("ชิ้นยาวเคยถูกมองเห็นแค่ 20% ", "ตอนนี้เห็นหมด"),
    *B.stats_row([
        {"big": "8.79", "unit": " ล้าน", "color": GREEN,
         "label": "ORF ที่เรียกได้จากชิ้นส่วนทั้งชุด — เดิมถูกตัดเหลือชิ้นละ 10"},
        {"big": "2.12", "unit": " พันล้าน", "color": BLUE,
         "label": "กรดอะมิโนที่ต้องส่งผ่าน ESM-2 นี่คืองานจริงของ GPU"},
        {"big": "186", "unit": " ชิ้น", "color": YELLOW,
         "label": "ชิ้นที่ชนเพดาน 512 ORF คิดเป็น 0.165% และเป็นชิ้น 50-100 kb ทั้งหมด"},
        {"big": "5.3", "unit": " GiB", "color": ORANGE,
         "label": "ขนาด cache เวกเตอร์ เก็บเป็น float16 แทน float32"},
    ], BODY_Y, 2057400),
    *B.note_bar(4343400, [run("วัดจากข้อมูลจริง ไม่ใช่ประมาณการ ", 1300, GREEN_D, THAI, b=1),
                          run("— การเรียก ORF หกเฟรมใช้เวลารวมเพียง ~1.3 ชั่วโมง-ซีพียู "
                              "คอขวดจึงอยู่ที่ transformer ไม่ใช่การหายีน", 1300, MUTED, THAI)],
                 fill=TINT_G),
    *B.note_bar(5029200, [run("ข้อควรระวัง ", 1300, ORANGE, THAI, b=1),
                          run("เรายังไม่ได้ทำ gene prediction จริง (เช่น Prodigal) "
                              "แต่ใช้ช่วงระหว่าง stop codon หกเฟรม — ต่างกันประมาณ 3.9 เท่า "
                              "เก็บเป็น ablation ไว้ทีหลัง", 1300, MUTED, THAI)],
                 fill=TINT_O),
 ], "ตัวเลข 8.79 ล้านกับ 2.13 พันล้านมาจากการรันจริงบนชุดข้อมูล ไม่ได้ประมาณ "
    "ถ้าถูกถามว่าทำไมไม่ใช้ Prodigal ให้ตอบว่า ORF ที่เราใช้คือช่วงระหว่าง stop codon หกเฟรม "
    "ซึ่งให้เปปไทด์มากกว่ายีนจริงราว 3.9 เท่า การเปลี่ยนไปใช้ Prodigal จะเปลี่ยน input ของโมเดล "
    "จึงควรทำเป็นการทดลองเทียบ ไม่ใช่เปลี่ยนเงียบ ๆ",
    foot="[8.79M ORFs // 2.13G AMINO ACIDS]")

# ── 05 calibration ───────────────────────────────────────────────────────────
add("สัดส่วนคลาสที่เคยไม่ตรง", "[SYS_v2.0 // SLIDE_05]", "[pos_weight // resample]", [
    title("เทรนด้วยสัดส่วนไหน ", "ต้องตรงกับตอนวัด"),
    *B.two_panel(BODY_Y, 2971800,
                 {"color": ORANGE, "kicker": "BEFORE", "head": "เทรน 45% · วัด 2.5%",
                  "lines": [("โมเดลเห็นพลาสมิดบ่อยกว่าโลกจริง 18 เท่า", ""),
                            ("จุดตัดที่ดีที่สุดไปอยู่ที่ 0.76–0.92", "ไม่ใช่ 0.5"),
                            ("ต้องจูน threshold ทีหลังทุกครั้ง", "")]},
                 {"color": GREEN, "kicker": "AFTER", "head": "สองสภาพการเทรน",
                  "lines": [("full", "เทรนบน 2.8% ตามจริง ใช้ pos_weight รับภาระความไม่สมดุล — เป็นตัวหลัก"),
                            ("balanced", "สุ่มให้ 1:1 แบบเดิม เก็บไว้เทียบกับเลขที่เคยรายงาน"),
                            ("รายงานทั้งสองสภาพเคียงกัน", "")]}),
    *B.note_bar(5334000, [run("ทำไมต้องเก็บแบบเดิมไว้ด้วย ", 1300, BLUE, THAI, b=1),
                          run("เพราะถ้าเปลี่ยนทั้งข้อมูลและสภาพการเทรนพร้อมกัน "
                              "เราจะไม่รู้ว่าเลขที่เปลี่ยนไปมาจากอะไร", 1300, MUTED, THAI)],
                 fill=TINT_B),
 ], "นี่คือลำดับ 2 ในตาราง tuning และเป็นข้อที่ถูกที่สุด แต่กระทบความน่าเชื่อถือของตัวเลขมากที่สุด "
    "การเก็บ regime balanced ไว้ด้วยคือการควบคุมตัวแปร ไม่ใช่ความลังเล",
    foot="[REGIMES: full | balanced]")

# ── 06 dataset rebuilt ───────────────────────────────────────────────────────
add("ชุดข้อมูลสร้างใหม่", "[SYS_v2.0 // SLIDE_06]", "[REBUILT_FROM_NCBI]", [
    title("สร้างชุดข้อมูลใหม่จาก NCBI ", "แล้วพิสูจน์ว่าเหมือนเดิม"),
    *B.cards_row([
        {"tag": "WHY", "color": BLUE, "head": "ของเดิมผูกกับเครื่องเดียว",
         "body": "replicons.csv เก็บ path ของโน้ตบุ๊กที่สร้างข้อมูล พอย้ายเครื่องก็ไล่กลับไปหาต้นทางไม่ได้",
         "note": ("ทำใหม่", "909 replicon · 500 assembly")},
        {"tag": "HOW", "color": GREEN, "head": "ตรวจสี่ชั้น",
         "body": "sha256 ต่อ replicon · ความยาวและ GC ต่อชิ้น · ยอดรวมต่อ split · URL ที่บันทึกไว้ตอนโหลดครั้งแรก",
         "note": ("ผลตรวจ", "21/21 ข้อผ่าน")},
        {"tag": "EDGE", "color": ORANGE, "head": "RefSeq ถอน 6 รายการ",
         "body": "assembly ถูกถอนหลังเราโหลดครั้งแรก กู้คืนผ่านฝาแฝดฝั่ง GenBank และ ENA",
         "note": ("ยืนยันแล้ว", "sha256 ตรงกับของเดิมทุกตัว")},
    ], BODY_Y, 3352800),
    *B.note_bar(5562600, [run("ได้อะไรเพิ่ม ", 1300, GREEN_D, THAI, b=1),
                          run("ตั้งแต่นี้ใครก็สร้างชุดข้อมูลนี้ขึ้นมาใหม่ได้จากสคริปต์เดียว "
                              "โดยไม่ต้องมีไฟล์จากเครื่องเรา", 1300, MUTED, THAI)], fill=TINT_G),
 ], "ประเด็นนี้ไม่ใช่เรื่องเทคนิคล้วน ๆ แต่เป็นเรื่องความน่าเชื่อถือของงาน "
    "ถ้ากรรมการถามว่า ข้อมูลนี้ทำซ้ำได้ไหม คำตอบตอนนี้คือได้ และมีการตรวจสี่ชั้นยืนยัน "
    "รวมถึงกรณีที่ NCBI ถอน record ออกไปแล้วซึ่งเรากู้กลับมาได้ครบ",
    foot="[909 REPLICONS // 500 ASSEMBLIES]")

# ── 07 pipeline ──────────────────────────────────────────────────────────────
add("ลำดับงานบนคลัสเตอร์", "[SYS_v2.0 // SLIDE_07]", "[SLURM // DEPENDENCY_CHAIN]", [
    title("งานเจ็ดขั้น ", "ร้อยกันด้วย dependency"),
    *B.flow([("00", "ติดตั้ง env", BLUE),
             ("02", "ตรวจข้อมูล", BLUE),
             ("10", "แบ่ง shard", GREEN),
             ("11·12", "Platon + MOB", GREEN),
             ("20", "ฝัง ESM-2", ORANGE),
             ("21·22", "รวม + เทรน", ORANGE),
             ("30", "ออกรายงาน", YELLOW)], BODY_Y, 1143000),
    *B.table_card(3657600, 1524000,
                  [("ทรัพยากรที่ขอ", "l"), ("พาร์ทิชัน / QoS", "l"), ("เพดานเวลา", "l"), ("หมายเหตุ", "l")],
                  [[("งานเครื่องมือเทียบ", INK, 1), ("cpu · cpu40h", MUTED, 0), ("4 วัน", MUTED, 0),
                    ("40 CPU ต่อหนึ่งงาน", MUTED, 0)],
                   [("ฝังเวกเตอร์ + เทรน", INK, 1), ("gpu · gpu40g", MUTED, 0), ("8 ชั่วโมง", ORANGE, 1),
                    ("A100 40 GB หนึ่งใบ + 16 CPU", MUTED, 0)]],
                  [2895600, 2667000, 1828800, 2565400], band=411480),
    *B.note_bar(5562600, [run("ทุกขั้นเขียนให้รันซ้ำได้ ", 1300, GREEN_D, THAI, b=1),
                          run("shard ที่เสร็จแล้วจะถูกข้าม งานที่ชนเพดานเวลาจึงส่งใหม่ได้โดยไม่เสียของ",
                              1300, MUTED, THAI)], fill=TINT_G, h=411480),
 ], "ภาพนี้ไว้ตอบคำถามว่ารันยังไง จุดสำคัญคือทุกขั้น resume ได้ "
    "ถ้างานถูกฆ่าเพราะหมดเวลา ส่งใหม่แล้วมันไปต่อจากจุดเดิม ไม่ใช่เริ่มใหม่ทั้งหมดแบบรอบโน้ตบุ๊ก",
    foot="[RESUMABLE // SKIPS FINISHED SHARDS]")

# ── 08 obstacles ─────────────────────────────────────────────────────────────
add("อุปสรรคและวิธีแก้", "[SYS_v2.0 // SLIDE_08]", "[4_TRAPS // ALL_FIXED]", [
    title("สี่กับดักที่เจอจริง ", "และวิธีแก้"),
    *B.cards_row([
        {"tag": "OOM", "color": ORANGE, "head": "ติดตั้งแล้วถูกฆ่า",
         "body": "ตัวแก้สมการของ conda กินแรมจนเกินโควตาของ login node",
         "note": ("แก้โดย", "ขอ allocation ด้วย srun + micromamba")},
        {"tag": "QOS", "color": BLUE, "head": "คิวปฏิเสธ job array",
         "body": "array 116 ช่องถูกนับเป็น 116 งาน ซึ่งเกินเพดานของ QoS",
         "note": ("แก้โดย", "รวมเป็นหนึ่งงานต่อเครื่องมือ")},
        {"tag": "STDIN", "color": GREEN, "head": "ลูปจบหลังรอบแรก",
         "body": "srun กลืน stdin ไปทั้งก้อน ลูปจึงจบหลัง shard แรกแต่คืนค่า 0 เหมือนสำเร็จ",
         "note": ("แก้โดย", "อ่านรายการผ่าน fd 3")},
        {"tag": "ENV", "color": YELLOW, "head": "ไม่มีคำสั่ง module",
         "body": "module เป็นฟังก์ชันของ shell จึงหายไปในสคริปต์ที่ไม่ใช่ login shell",
         "note": ("แก้โดย", "bootstrap ใน lib.sh")},
    ], BODY_Y, 3352800),
    *B.note_bar(5562600, [run("บทเรียนที่แพงที่สุดคือข้อสาม ", 1300, ORANGE, THAI, b=1),
                          run("งานจบด้วยสถานะสำเร็จทั้งที่ทำไปแค่ 1 ใน 18 — "
                              "ตั้งแต่นี้จะนับรอบจาก log ไม่ดูแค่ exit code", 1300, MUTED, THAI)],
                 fill=TINT_O),
 ], "สี่ข้อนี้บันทึกไว้ใน HANDOFF.md แล้ว ข้อที่สามคือข้อที่อันตรายที่สุดเพราะไม่มีสัญญาณเตือน "
    "งานขึ้นสถานะ COMPLETED ทั้งที่ embed ไปแค่ shard เดียว แล้วขั้นถัดไปก็รันบน cache ที่ไม่ครบ",
    foot="[4_TRAPS // LOGGED IN HANDOFF.md]")

# ── 09 resources ─────────────────────────────────────────────────────────────
add("ทรัพยากรที่ใช้จริง", "[SYS_v2.0 // SLIDE_09]", "[MEASURED // NOT_ESTIMATED]", [
    title("ใช้เครื่องไปเท่าไหร่ ", "วัดจาก log จริง"),
    *B.stats_row([
        {"big": "11.2", "unit": " นาที", "color": BLUE,
         "label": "ต่อหนึ่ง shard ของ train (6,591 ชิ้น) นิ่งมาก คลาดเคลื่อน ±15%"},
        {"big": "18", "unit": " shard", "color": GREEN,
         "label": "แบ่งเป็น train 12 · validation 3 · test 3 เพื่อให้รันซ้ำได้"},
        {"big": "~3", "unit": " ชั่วโมง", "color": ORANGE,
         "label": "เวลาฝังเวกเตอร์ทั้งชุด บน A100 หนึ่งใบ จากเพดาน 8 ชั่วโมง"},
        {"big": "40", "unit": " รอบเทรน", "color": YELLOW,
         "label": "4 variant × 2 สภาพ × 5 seed — ขั้นที่กินเวลามากที่สุด"},
    ], BODY_Y, 2514600),
    *B.note_bar(4876800, [run("รายงานบันทึกเองอัตโนมัติ ", 1300, GREEN_D, THAI, b=1),
                          run("จำนวน record · CPU-hours · GPU-hours · หน่วยความจำสูงสุด · พื้นที่ดิสก์ "
                              "ดึงจากฐานบัญชีของ Slurm ไม่ต้องจดมือ", 1300, MUTED, THAI)], fill=TINT_G),
    *B.note_bar(5562600, [run("ทำไมต้องเก็บ ", 1300, BLUE, THAI, b=1),
                          run("ส่วนวิธีการของรายงานต้องบอกได้ว่าใช้ทรัพยากรเท่าไหร่ "
                              "และข้อมูลนี้จะหายไปเมื่องานหลุดจากฐานบัญชี", 1300, MUTED, THAI)],
                 fill=TINT_B),
 ], "ตัวเลข 11.2 นาทีมาจากการจับเวลาระหว่าง shard ใน log จริง 11 ค่า ไม่ได้เดา "
    "จากตรงนี้ประมาณต่อได้ว่าทั้งงานจะจบเมื่อไหร่ ส่วนขั้นเทรนคือขั้นที่เสี่ยงชนเพดาน 8 ชั่วโมงที่สุด",
    foot="[MEASURED FROM LOGS // NOT ESTIMATED]")

# ── 10 published tools ───────────────────────────────────────────────────────
add("คู่เทียบที่เพิ่มเข้ามา", "[SYS_v2.0 // SLIDE_10]", "[PLATON // MOB-RECON]", [
    title("เทียบกับเครื่องมือ ", "ที่คนใช้งานจริง"),
    *B.table_card(BODY_Y, 2743200,
                  [("แบบจำลอง", "l"), ("MCC ชิ้นส่วน", "r"), ("MCC ระดับ replicon", "r"),
                   ("MCC 1–2 kb", "r"), ("MCC 50–100 kb", "r")],
                  [[("MOB-recon", INK, 1), ("0.440", INK, 1), ("0.505", MUTED, 0),
                    ("0.511", GREEN_D, 1), ("0.341", MUTED, 0)],
                   [("Platon (โหมด accuracy)", INK, 1), ("0.372", INK, 0), ("0.563", MUTED, 0),
                    ("0.305", MUTED, 0), ("0.349", MUTED, 0)],
                   [("k-mer 5 + GC + ori", INK, 1), ("0.190", MUTED, 0), ("0.667", MUTED, 0),
                    ("0.099", MUTED, 0), ("0.251", MUTED, 0)],
                   [("ESM-2 เปล่า", INK, 1), ("0.254", MUTED, 0), ("0.857", MUTED, 0),
                    ("0.204", MUTED, 0), ("0.495", MUTED, 0)],
                   [("ESM-2 + GC + ori", INK, 1), ("0.272", INK, 1), ("0.889", GREEN_D, 1),
                    ("0.125", ORANGE, 0), ("0.528", GREEN_D, 1)]],
                  [2743200, 1828800, 1828800, 1778000, 1778000]),
    *B.note_bar(5029200, [run("อ่านตารางนี้สองทิศทาง ", 1300, ORANGE, THAI, b=1),
                          run("ระดับชิ้นส่วนเรายังแพ้ MOB-recon (", 1300, MUTED, THAI),
                          run("0.272", 1300, MUTED, MONO),
                          run(" ต่อ ", 1300, MUTED, THAI),
                          run("0.440", 1300, MUTED, MONO),
                          run(") แต่ระดับ replicon เราชนะชัด (", 1300, MUTED, THAI),
                          run("0.889", 1300, GREEN_D, MONO),
                          run(" ต่อ ", 1300, MUTED, THAI),
                          run("0.505", 1300, MUTED, MONO),
                          run(") และแนวโน้มตามความยาวกลับทางกัน "
                              "เครื่องมือเก่งที่ชิ้นสั้น เราเก่งที่ชิ้นยาว",
                              1300, MUTED, THAI)], fill=TINT_O, h=548640),
    *B.note_bar(5715000, [run("ทำไมเครื่องมือจึงไม่แย่อย่างที่คิด ", 1300, BLUE, THAI, b=1),
                          run("precision ของ MOB-recon สูงถึง ", 1300, MUTED, THAI),
                          run("0.988", 1300, MUTED, MONO),
                          run(" แต่ recall แค่ ", 1300, MUTED, THAI),
                          run("0.199", 1300, MUTED, MONO),
                          run(" คือตอบน้อยแต่แม่น", 1300, MUTED, THAI)],
                 fill=TINT_B, h=411480),
 ], "ประโยคสำคัญของสไลด์นี้คือเราไม่ได้ชนะทุกช่อง ระดับชิ้นส่วนเดี่ยว MOB-recon ยังดีกว่า "
    "เพราะมันตอบเฉพาะเวลาที่มั่นใจ precision 0.988 แต่ recall 0.199 "
    "ขณะที่โมเดลเราต้องตอบทุกชิ้น เมื่อรวมคะแนนขึ้นมาเป็นระดับโมเลกุลด้วยการโหวตจากหลายชิ้น "
    "ข้อดีของเราจึงโผล่ออกมา 0.889 ต่อ 0.505 และแนวโน้มตามความยาวกลับทางกันจริง "
    "ซึ่งเป็นหลักฐานว่าสองวิธีจับสัญญาณต่างกัน ไม่ใช่วิธีเดียวกันที่เก่งกว่า",
    foot="[MOB-RECON 0.440 // ESM2 0.889 REPLICON]")

# ── 11 what the report gives ─────────────────────────────────────────────────
add("รายงานที่จะได้", "[SYS_v2.0 // SLIDE_11]", "[12_TABLES + PROVENANCE]", [
    title("รายงานหนึ่งไฟล์ ", "ที่ถอดตัวเลขมาจากผลโดยตรง"),
    *card(L, BODY_Y, 5181600, 3048000, BLUE),
    text(L + 304800, BODY_Y + 254000, 4572000, 300000,
         [para([run("TABLES", 1200, BLUE, MONO)])]),
    *B.bullets(L + 304800, BODY_Y + 566420, 4572000,
               [("ชุดข้อมูล", "ต่อ split · การกระจายความยาว"),
                ("Composition baseline", "GC · ori · k-mer บน test เต็ม"),
                ("เครื่องมือจริง", "Platon 3 โหมด · MOB-recon × 3 นโยบาย"),
                ("ESM-2 ladder", "4 variant × 2 สภาพ × 5 seed"),
                ("แยกตามความยาว", "ทั้งฝั่งเครื่องมือและฝั่งโมเดล")], color=BLUE, step=457200),
    *card(6197600, BODY_Y, 5181600, 3048000, GREEN),
    text(6502400, BODY_Y + 254000, 4572000, 300000,
         [para([run("PROVENANCE", 1200, GREEN_D, MONO)])]),
    *B.bullets(6502400, BODY_Y + 566420, 4572000,
               [("จำนวน record", "ชิ้นส่วน · ORF · กรดอะมิโน · ขนาด cache"),
                ("ทรัพยากร", "CPU-hours · GPU-hours · หน่วยความจำสูงสุด"),
                ("พื้นที่ดิสก์", "แยกตามส่วนของงาน"),
                ("เครื่องและเวอร์ชัน", "CPU · RAM · torch · CUDA"),
                ("ทั้งหมดดึงอัตโนมัติ", "ไม่ได้พิมพ์เอง")], color=GREEN_D, step=457200),
    *B.note_bar(5562600, [run("สองคอลัมน์ที่ห้ามคัดลอกลงรายงานตรง ๆ ", 1300, ORANGE, THAI, b=1),
                          run("hard_label_score — คะแนนจากคำตัดสิน ไม่ใช่ความน่าจะเป็น · "
                              "mcc_oracle — จูนจุดตัดบน test เอง เป็นเพดานบน ไม่ใช่ผลที่ใช้ได้จริง",
                              1300, MUTED, THAI)], fill=TINT_O, h=640080),
 ], "สองคอลัมน์ท้ายสไลด์เป็นกับดักของรายงานอัตโนมัติ ถ้าหยิบไปใส่ตารางโดยไม่อธิบาย "
    "จะดูเหมือนเราได้คะแนนสูงกว่าความจริง ต้องเขียนกำกับทุกครั้ง",
    foot="[12_TABLES // AUTO-TRANSCRIBED]")

# ── 12 status ────────────────────────────────────────────────────────────────
add("เวลาและทรัพยากรที่ใช้จริง", "[SYS_v2.0 // SLIDE_12]", "[28_OCT_2026 // 09:31 DONE]", [
    title("ทั้งงานจบใน ", "วันเดียว"),
    *B.table_card(BODY_Y, 2743200,
                  [("ขั้นงาน", "l"), ("job", "l"), ("เวลาที่ใช้จริง", "l"), ("ทรัพยากรที่จับได้", "l")],
                  [[("Platon + MOB-recon", INK, 1), ("411700-01", GREY, 0),
                    ("1:08:16", INK, 1), ("40 CPU · RSS สูงสุด 15.9 GiB", MUTED, 0)],
                   [("ฝังเวกเตอร์ ESM-2", INK, 1), ("411706", GREY, 0),
                    ("3:00:31", INK, 1), ("A100 7g.40gb · 16 CPU · 3.0 ชม.-GPU", MUTED, 0)],
                   [("รวม cache", INK, 1), ("411707", GREY, 0),
                    ("0:05:06", MUTED, 0), ("4 CPU · 99 MiB", MUTED, 0)],
                   [("เทรน ladder 40 fit", INK, 1), ("411708", GREY, 0),
                    ("3:09:22", INK, 1), ("A100 · 3.2 ชม.-GPU · เพดาน 8 ชม.", GREEN_D, 1)],
                   [("ออกรายงาน", INK, 1), ("411709", GREY, 0),
                    ("0:00:09", MUTED, 0), ("4 CPU · อ่านไฟล์ผลล้วน", MUTED, 0)]],
                  [3048000, 1219200, 1828800, 3860800]),
    *B.note_bar(5029200, [run("ต้นทุนรวมของรันนี้ ", 1300, GREEN_D, THAI, b=1),
                          run("147.5", 1300, MUTED, MONO),
                          run(" ชั่วโมง-ซีพียูที่จอง (ใช้จริง ", 1300, MUTED, THAI),
                          run("46.6", 1300, MUTED, MONO),
                          run(") · ", 1300, MUTED, THAI),
                          run("6.3", 1300, MUTED, MONO),
                          run(" ชั่วโมง-จีพียู · cache ", 1300, MUTED, THAI),
                          run("5.3 GiB", 1300, MUTED, MONO),
                          run(" · ทุกตัวเลขดึงจาก sacct อัตโนมัติ ไม่ได้พิมพ์เอง",
                              1300, MUTED, THAI)], fill=TINT_G, h=548640),
    *B.note_bar(5715000, [run("ความเสี่ยงที่เตรียมไว้แล้วไม่ได้ใช้ ", 1300, BLUE, THAI, b=1),
                          run("ขั้นเทรน resume ไม่ได้และมีเพดาน ", 1300, MUTED, THAI),
                          run("8", 1300, MUTED, MONO),
                          run(" ชั่วโมง จบที่ ", 1300, MUTED, THAI),
                          run("3:09", 1300, MUTED, MONO),
                          run(" จึงเหลือที่ว่าง", 1300, MUTED, THAI)], fill=TINT_B, h=411480),
 ], "สไลด์นี้เดิมเป็นสไลด์ประมาณการ ตอนนี้เป็นของจริงแล้ว ตัวเลขทุกตัวมาจาก sacct "
    "ไม่ใช่การจับเวลาด้วยมือ ประเด็นที่ควรพูดคือเพดาน 8 ชั่วโมงของคิว GPU "
    "เราเตรียมแผนย้ายคิวไว้แล้วแต่ไม่ต้องใช้ เพราะ epoch จริงเร็วกว่าที่กลัวราวสองเท่า "
    "ส่วนช่องว่างระหว่าง 147.5 กับ 46.6 ชั่วโมง-ซีพียู คืองานที่ถือคอร์ไว้รอ GPU ซึ่งเป็นธรรมชาติของขั้นนี้",
    foot="[147.5 CPU-H // 6.3 GPU-H // 5.3 GIB]")

# ── 13 trust ─────────────────────────────────────────────────────────────────
add("อะไรเชื่อได้ อะไรยัง", "[SYS_v2.0 // SLIDE_13]", "[HONESTY_TABLE]", [
    title("แยกให้ชัด ", "ก่อนถูกถาม"),
    *B.two_panel(BODY_Y, 3352800,
                 {"color": GREEN, "kicker": "เชื่อได้แล้ว", "head": "วัดบน test เต็ม 16,944 ชิ้น",
                  "lines": [("ระดับ replicon MCC 0.889 ชนะเครื่องมือจริงทั้งสองตัว", ""),
                            ("ยิ่งชิ้นส่วนยาวยิ่งแม่น 0.13 ที่ 1–2 kb เป็น 0.53 ที่ 50–100 kb", ""),
                            ("GC อย่างเดียวได้ MCC 0.019 คือเท่าเดาสุ่ม", ""),
                            ("ชุดข้อมูลสร้างซ้ำได้ ตรวจผ่าน 21 ข้อ", "")]},
                 {"color": ORANGE, "kicker": "ยังตอบไม่ได้", "head": "ผลออกแล้วแต่ยังสรุปไม่ได้",
                  "lines": [("ระดับชิ้นส่วนเดี่ยวยังแพ้ MOB-recon 0.272 ต่อ 0.440", ""),
                            ("อันดับระหว่าง variant ห่างกันแค่ 0.254–0.276 ยังไม่มีช่วงความเชื่อมั่น", ""),
                            ("ช่วงความเชื่อมั่นระดับโมเลกุล — พลาสมิดใน test มีแค่ 54 ตัว", ""),
                            ("186 ชิ้นถูกตัด ORF ที่เพดาน 512 ทั้งหมดเป็นชิ้น 50–100 kb", "")]},
                 step=533400),
    *B.note_bar(5486400, [run("สามข้อที่ต้องทำต่อ ก่อนจะเรียกตัวเลขพวกนี้ว่าผลสรุป ", 1300, BLUE, THAI, b=1),
                          run("bootstrap ช่วงความเชื่อมั่นระดับ assembly · ตรวจว่าจีโนมทดสอบ"
                              "ไปอยู่ในฐานข้อมูลของเครื่องมือหรือไม่ · วัดความคล้ายระหว่าง train กับ test",
                              1300, MUTED, THAI)], fill=TINT_B, h=640080),
 ], "สไลด์นี้คือเกราะของเรา พูดเองก่อนที่จะถูกจับได้ สามข้อที่ควรพูดด้วยปากตัวเอง "
    "หนึ่ง ระดับชิ้นส่วนเดี่ยวเรายังแพ้ MOB-recon สอง ความต่างระหว่าง variant เล็กกว่าที่จะฟันธงได้ "
    "เพราะยังไม่มีช่วงความเชื่อมั่น และสาม พลาสมิดในชุดทดสอบมีแค่ 54 โมเลกุล "
    "การรายงานคะแนนระดับโมเลกุลเป็นเลขเดี่ยวจึงไม่พอ",
    foot="[3 WINS // 4 OPEN QUESTIONS]")

# ── 14 roadmap ───────────────────────────────────────────────────────────────
add("งานที่เหลือ", "[SYS_v2.0 // SLIDE_14]", "[ROADMAP // 3_DONE_OF_8]", [
    title("เหลืออะไร ", "และอันไหนถูกที่สุด"),
    *B.cards_row([
        {"tag": "ถูก · ทำได้เลย", "color": GREEN, "head": "คำนวณจากไฟล์ผล",
         "body": "เปลี่ยนวิธีรวมคะแนนเป็นระดับโมเลกุล จากเฉลี่ยความน่าจะเป็น เป็นเฉลี่ย logit "
                 "หรือถ่วงน้ำหนักตามความยาว · ทำช่วงความเชื่อมั่นแบบ bootstrap",
         "note": ("ต้นทุน", "ไม่ต้องเทรนใหม่เลย")},
        {"tag": "ถูก · เทรนใหม่", "color": BLUE, "head": "ปรับตัวโมเดล",
         "body": "เพิ่ม regularization ตามหลักฐาน overfitting · ขยายน้ำหนักฟีเจอร์ non-coding "
                 "ที่ตอนนี้เป็นแค่ 12 ค่าในเวกเตอร์ 328 มิติ",
         "note": ("ต้นทุน", "ใช้ cache เดิมได้")},
        {"tag": "แพง · ต้อง GPU", "color": ORANGE, "head": "เปลี่ยนตัวตั้งต้น",
         "body": "ขยับไป ESM-2 ตัวใหญ่ขึ้นหรือ fine-tune · ใช้ Prodigal เป็นการทดลองเทียบ · "
                 "ตรวจหา ori จริงแทนร่องรอยองค์ประกอบ",
         "note": ("ต้นทุน", "ฝังเวกเตอร์ใหม่ทั้งชุด")},
    ], BODY_Y, 3352800),
    *B.note_bar(5562600, [run("ข้อเสนอ ", 1300, GREEN_D, THAI, b=1),
                          run("ปิดงานด้วยชุดข้อมูล 500 assembly นี้ แล้วเขียนการขยายไปทั้ง 66,230 assembly "
                              "ไว้ในหัวข้องานต่อ พร้อมตัวเลขต้นทุนที่คำนวณไว้แล้ว", 1300, MUTED, THAI)],
                 fill=TINT_G),
 ], "ตาราง tuning เดิมมี 8 ข้อ รอบนี้ปิดไปสามข้อแรกซึ่งเป็นข้อที่แพงที่สุด "
    "ที่เหลือส่วนใหญ่ถูกและทำบนโน้ตบุ๊กได้ ถ้าต้องเลือกทำต่อข้อเดียว ผมเลือกวิธีรวมคะแนนระดับโมเลกุล "
    "เพราะคำนวณจากไฟล์ผลที่จะได้วันนี้เลย",
    foot="[3_OF_8_DONE]")

# ── 15 summary ───────────────────────────────────────────────────────────────
add("สรุป", "[SYS_v2.0 // SLIDE_15]", "[END]", [
    title("สรุป ", "รอบนี้ได้อะไร"),
    *B.cards_row([
        {"tag": "01", "color": GREEN, "head": "ฝั่งโมเดลตามทันฝั่งข้อมูล",
         "body": "เดิมข้อสรุปเรื่องชุดข้อมูลวัดบนข้อมูลเต็ม แต่ข้อสรุปเรื่องโมเดลวัดบน 1.8% "
                 "รอบนี้ทั้งสองฝั่งอยู่บนข้อมูลเดียวกันแล้ว",
         "note": ("หมายความว่า", "เลขทุกตัวเทียบกันได้")},
        {"tag": "02", "color": BLUE, "head": "มีคู่เทียบที่คนรู้จัก",
         "body": "Platon และ MOB-recon รันบนชุดทดสอบเดียวกัน แยกผลตามความยาวชิ้นส่วน",
         "note": ("หมายความว่า", "งานมีจุดอ้างอิงภายนอก")},
        {"tag": "03", "color": ORANGE, "head": "ทำซ้ำได้ทั้งเส้น",
         "body": "ชุดข้อมูลสร้างใหม่จาก NCBI ได้ · ทุกขั้นรันซ้ำได้ · รายงานบันทึกทรัพยากรเอง",
         "note": ("หมายความว่า", "ส่งต่อให้คนอื่นทำต่อได้")},
    ], BODY_Y, 3352800),
    *B.note_bar(5562600, [run("ก้าวถัดไป ", 1300, GREEN_D, THAI, b=1),
                          run("อ่านรายงานที่จะออกวันนี้ เลือกเรื่องเล่าหลักจากตารางแยกตามความยาว "
                              "แล้วเติมช่วงความเชื่อมั่นก่อนเขียนบท", 1300, MUTED, THAI)], fill=TINT_G),
 ], "ปิดด้วยสามข้อนี้ และย้ำว่าสิ่งที่เปลี่ยนไปไม่ใช่แค่ตัวเลขใหญ่ขึ้น "
    "แต่เป็นการทำให้ข้อสรุปทุกข้อวางอยู่บนฐานเดียวกันและตรวจสอบย้อนได้",
    foot="[END // CU-AIM_RUN]")


# ── package ──────────────────────────────────────────────────────────────────
def build():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        z.extractall(WORK)

    for d in ("ppt/slides", "ppt/notesSlides"):
        for f in (WORK / d).glob("*.xml"):
            f.unlink()
        for f in (WORK / d / "_rels").glob("*.rels"):
            f.unlink()

    notes_tpl = (WORK / "ppt/notesSlides/notesSlide.tpl")
    hdr = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
           '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8F9FA"/></a:solidFill>'
           '</p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name="Shape 1"/>'
           '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
           '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
           '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')
    ftr = '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'

    for i, s in enumerate(slides, 1):
        parts = list(tags(s["tl"], s["tr"], FOOT_Y)) if s["cover"] else []
        if s["cover"]:
            parts = list(tags(s["tl"], s["tr"]))
            parts += tags("[112,929 FRAGMENTS // 8.79M ORFs]", "[BKK_EMERALD // GOOGLE_BLUE]",
                          FOOT_Y)
        else:
            parts = list(tags(s["tl"], s["tr"]))
            parts += header(s["caption"], i, TOTAL, LOGO)
            parts += tags(s["foot"], f"[{i:02d}_OF_{TOTAL}]", FOOT_Y)
        parts += s["body"]
        (WORK / f"ppt/slides/slide{i}.xml").write_text(hdr + "".join(parts) + ftr,
                                                       encoding="utf-8")
        img = ('<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/'
               'officeDocument/2006/relationships/image" Target="../media/image49.png"/>'
               if s["cover"] else "")
        (WORK / f"ppt/slides/_rels/slide{i}.xml.rels").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/'
            f'2006/relationships/notesSlide" Target="../notesSlides/notesSlide{i}.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/image" Target="../media/image6.png"/>'
            f'{img}</Relationships>', encoding="utf-8")
        write_notes(i, s["notes"])

    if notes_tpl.exists():
        notes_tpl.unlink()
    rewrite_presentation()
    drop_embedded_fonts()
    rewrite_content_types()
    pack()


NOTES_TPL = None


def write_notes(i, body):
    global NOTES_TPL
    if NOTES_TPL is None:
        NOTES_TPL = ORIG_NOTES
    xml = re.sub(r"(<p:ph idx=\"1\" type=\"body\"/>.*?<a:t>).*?(</a:t>)",
                 lambda m: m.group(1) + _esc(body) + m.group(2), NOTES_TPL, flags=re.S)
    (WORK / f"ppt/notesSlides/notesSlide{i}.xml").write_text(xml, encoding="utf-8")
    (WORK / f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
        '</Relationships>', encoding="utf-8")


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rewrite_presentation():
    p = WORK / "ppt/_rels/presentation.xml.rels"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r'<Relationship Id="[^"]+" Type="[^"]*/slide" Target="slides/[^"]+"/>', "", s)
    new = "".join(
        f'<Relationship Id="rId{100 + i}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1))
    p.write_text(s.replace("</Relationships>", new + "</Relationships>"), encoding="utf-8")

    q = WORK / "ppt/presentation.xml"
    t = q.read_text(encoding="utf-8")
    lst = "".join(f'<p:sldId id="{255 + i}" r:id="rId{100 + i}"/>'
                  for i in range(1, len(slides) + 1))
    t = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", f"<p:sldIdLst>{lst}</p:sldIdLst>", t, flags=re.S)
    q.write_text(t, encoding="utf-8")


def drop_embedded_fonts():
    """Remove the font embedding inherited from the Google Slides export.

    The export subsets each font to the glyphs the OLD deck used, so new text
    renders those fonts' missing letters as tofu - "ORF" and "A100" came out as
    empty boxes. Without the embedding, PowerPoint and Google Slides fall back to
    the installed Kanit / Noto Sans Thai (both are Google Fonts) instead."""
    for f in (WORK / "ppt/fonts").glob("*.fntdata"):
        f.unlink()
    (WORK / "ppt/fonts").rmdir()
    q = WORK / "ppt/presentation.xml"
    t = q.read_text(encoding="utf-8")
    t = re.sub(r"<p:embeddedFontLst>.*?</p:embeddedFontLst>", "", t, flags=re.S)
    t = t.replace(' embedTrueTypeFonts="1"', "").replace(' saveSubsetFonts="1"', "")
    q.write_text(t, encoding="utf-8")
    p = WORK / "ppt/_rels/presentation.xml.rels"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r'<Relationship Id="[^"]+" Type="[^"]*/font" Target="fonts/[^"]+"/>', "", s)
    p.write_text(s, encoding="utf-8")


def rewrite_content_types():
    p = WORK / "[Content_Types].xml"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r'<Override ContentType="[^"]*presentationml\.(?:slide|notesSlide)\+xml" '
               r'PartName="/ppt/(?:slides|notesSlides)/[^"]+"/>', "", s)
    new = "".join(
        f'<Override ContentType="application/vnd.openxmlformats-officedocument.'
        f'presentationml.slide+xml" PartName="/ppt/slides/slide{i}.xml"/>'
        f'<Override ContentType="application/vnd.openxmlformats-officedocument.'
        f'presentationml.notesSlide+xml" PartName="/ppt/notesSlides/notesSlide{i}.xml"/>'
        for i in range(1, len(slides) + 1))
    p.write_text(s.replace("</Types>", new + "</Types>"), encoding="utf-8")


def pack():
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(WORK.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(WORK).as_posix())


with zipfile.ZipFile(SRC) as _z:
    ORIG_NOTES = _z.read("ppt/notesSlides/notesSlide2.xml").decode("utf-8")

build()
print(f"wrote {OUT}  ({len(slides)} slides)")
