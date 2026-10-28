"""Composite blocks: the repeatable slide furniture of the Siwak CI."""
import re

from ci import (GREEN, GREEN_D, BLUE, ORANGE, YELLOW, INK, MUTED, GREY, LINE, WHITE,
                TINT_G, TINT_B, TINT_O, TINT_Y, TINT_N, KANIT, THAI, MONO,
                L, R, W, BODY_Y, accent, card, para, pill, run, shape, text)

GAP = 203200
ACCENTS = [ORANGE, BLUE, GREEN, YELLOW]
TINTS = {ORANGE: TINT_O, BLUE: TINT_B, GREEN: TINT_G, GREEN_D: TINT_G, YELLOW: TINT_Y}
LABEL = {ORANGE: ORANGE, BLUE: BLUE, GREEN: GREEN_D, YELLOW: "B06000"}


def cols(n, gap=GAP):
    w = (W - gap * (n - 1)) // n
    return w, [L + i * (w + gap) for i in range(n)]


def cards_row(items, y, h, y_tag=None):
    """items: [{tag, head, body, note, color}] - the deck's workhorse block."""
    n = len(items)
    w, xs = cols(n)
    out = []
    for it, x in zip(items, xs):
        c = it.get("color", BLUE)
        out += card(x, y, w, h, c)
        ix, iw = x + 304800, w - 609600
        # fixed rhythm: the heading gets room for two lines, so a long one
        # cannot land on top of the body text below it
        out.append(text(ix, y + 254000, iw, 256540,
                        [para([run(it.get("tag", ""), 1300, LABEL.get(c, c), MONO)])]))
        out.append(text(ix, y + 600000, iw, 620000,
                        [para([run(it["head"], 1800, INK, KANIT, b=1)], line=110000)]))
        out.append(text(ix, y + 1280000, iw, 1100000,
                        [para([run(it["body"], 1300, MUTED, THAI)], line=145000)]))
        if it.get("note"):
            nh = 711200 if len(it["note"][1]) > 46 else 520700
            ny = y + h - nh - 254000
            label = run(it["note"][0] + " ", 1300, LABEL.get(c, c), THAI, b=1)
            out.append(pill(ix, ny, iw, nh, TINTS.get(c, TINT_N),
                            [label, run(it["note"][1], 1300, INK, THAI)]))
    return out


# A figure set in the Thai face depends on that font's Latin coverage, and the
# installed Noto Sans Thai has none: digits come out letter-spaced and the
# decimal point disappears entirely. Splitting the cell lets every number take
# the mono face while the Thai words keep theirs.
_FIG = re.compile(r"[0-9][0-9A-Za-z.,:\u00d7/%\-]*")


def figure_runs(txt, sz, clr, bold=0):
    out, i = [], 0
    for m in _FIG.finditer(txt):
        if m.start() > i:
            out.append(run(txt[i:m.start()], sz, clr, THAI, b=bold))
        fig = m.group(0).rstrip(".,:-/")
        out.append(run(fig, sz, clr, MONO, b=bold))
        i = m.start() + len(fig)
    if i < len(txt):
        out.append(run(txt[i:], sz, clr, THAI, b=bold))
    return out or [run(txt, sz, clr, THAI, b=bold)]


def table_card(y, h, head, rows, widths, band=457200, row_h=None):
    """A bordered data table: tinted header band, hairline between rows."""
    out = card(L, y, W, h)
    out.append(shape(L, y, W, band, geom="roundRect", fill=TINT_N, adj=12000))
    out.append(shape(L, y + band // 2, W, band // 2, fill=TINT_N))
    xs, acc = [], L + 304800
    for cw in widths:
        xs.append(acc)
        acc += cw
    for x, cw, cell in zip(xs, widths, head):
        algn = "l" if cell[1] == "l" else cell[1]
        out.append(text(x, y + 114300, cw - 203200, 256540,
                        [para([run(cell[0], 1200, MUTED, THAI, b=1)], algn=algn)]))
    rh = row_h or (h - band) // max(len(rows), 1)
    for i, r in enumerate(rows):
        ry = y + band + i * rh
        if i:
            out.append(shape(L + 304800, ry, W - 609600, 9525, fill=LINE))
        for x, cw, cell, spec in zip(xs, widths, r, head):
            txt, clr, bold = cell if isinstance(cell, tuple) else (cell, INK, 0)
            runs = ([run(txt, 1300, clr, MONO, b=bold)] if clr == GREY
                    else figure_runs(txt, 1300, clr, bold))
            out.append(text(x, ry + (rh - 256540) // 2, cw - 203200, 300000,
                            [para(runs, algn=spec[1])]))
    return out


def stats_row(items, y, h):
    """Big-number callouts: value in Kanit, caption under it."""
    n = len(items)
    w, xs = cols(n)
    out = []
    for it, x in zip(items, xs):
        c = it.get("color", GREEN)
        out += card(x, y, w, h, c)
        ix, iw = x + 304800, w - 609600
        out.append(text(ix, y + 304800, iw, 600000,
                        [para([run(it["big"], 3600, LABEL.get(c, c), KANIT, b=1),
                               run(it.get("unit", ""), 1600, MUTED, THAI)], line=100000)]))
        out.append(text(ix, y + 1000000, iw, 800000,
                        [para([run(it["label"], 1300, MUTED, THAI)], line=145000)]))
    return out


def two_panel(y, h, left_p, right_p, step=600000):
    """Before / after, or do / don't - two half-width cards."""
    w, xs = cols(2)
    out = []
    for p, x in zip((left_p, right_p), xs):
        c = p["color"]
        out += card(x, y, w, h, c)
        ix, iw = x + 304800, w - 609600
        out.append(text(ix, y + 254000, iw, 300000,
                        [para([run(p["kicker"], 1200, LABEL.get(c, c), MONO)])]))
        out.append(text(ix, y + 600000, iw, 340000,
                        [para([run(p["head"], 1900, INK, KANIT, b=1)], line=110000)]))
        ly = y + 1080000
        for line in p["lines"]:
            runs = [run("—  ", 1300, LABEL.get(c, c), THAI, b=1)]
            if isinstance(line, tuple):
                runs += figure_runs(line[0] + " ", 1300, INK, 1)
                runs += figure_runs(line[1], 1300, MUTED)
            else:
                runs += figure_runs(line, 1300, MUTED)
            out.append(text(ix, ly, iw, 560000, [para(runs, line=145000)]))
            ly += step
    return out


def flow(steps, y, h):
    """Numbered pipeline chips with arrows between them."""
    n = len(steps)
    arrow = 190500
    w = (W - arrow * (n - 1)) // n
    out = []
    for i, (tag, label, c) in enumerate(steps):
        x = L + i * (w + arrow)
        out += card(x, y, w, h, c)
        ix, iw = x + 190500, w - 381000
        out.append(text(ix, y + 190500, iw, 240000,
                        [para([run(tag, 1100, LABEL.get(c, c), MONO)])]))
        out.append(text(ix, y + 495300, iw, 700000,
                        [para([run(label, 1300, INK, THAI, b=1)], line=130000)]))
        if i < n - 1:
            out.append(text(x + w, y + h // 2 - 160000, arrow, 320000,
                            [para([run("›", 1800, GREY, THAI, b=1)], algn="ctr")]))
    return out


def bullets(x, y, w, items, color=GREEN_D, sz=1300, step=520700, lead=145000):
    out, cy = [], y
    for it in items:
        runs = [run("●  ", 900, color, THAI, b=1)]
        if isinstance(it, tuple):
            runs += figure_runs(it[0] + " ", sz, INK, 1) + figure_runs(it[1], sz, MUTED)
        else:
            runs += figure_runs(it, sz, MUTED)
        out.append(text(x, cy, w, step, [para(runs, line=lead)]))
        cy += step
    return out


def note_bar(y, runs, fill=TINT_N, h=520700):
    return [pill(L, y, W, h, fill, runs, pad=304800)]
