"""Shape primitives in the Siwak System CI, emitting the same DrawingML dialect
as the source deck so the embedded Kanit / Noto Sans Thai / JetBrains Mono keep
working and nothing has to be re-themed."""

# --- palette (CI guideline slide 03) -----------------------------------------
GREEN, GREEN_D = "00C853", "00A045"
BLUE, ORANGE, YELLOW = "1A73E8", "FF5722", "FBBC05"
INK, MUTED, GREY = "202124", "5F6368", "BDC1C6"
LINE, BG, WHITE = "E8EAED", "F8F9FA", "FFFFFF"
TINT_G, TINT_B, TINT_O, TINT_Y, TINT_N = "E6F7EE", "EAF2FE", "FFF3F0", "FEF7E0", "F1F3F4"

KANIT, THAI, MONO = "Kanit", "Noto Sans Thai", "JetBrains Mono"

# --- grid ---------------------------------------------------------------------
SW, SH = 12192000, 6858000
L, R = 812800, 11379200
W = R - L                      # 10566400
TAG_Y, TAG_H = 279400, 238760
HEAD_Y = 628650
RULE_Y = 1028700
TITLE_Y, TITLE_H = 1320800, 500380
BODY_Y = 2100580
FOOT_Y = 6238240

_ids = [100]


def nid():
    _ids[0] += 1
    return _ids[0]


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def run(text, sz=1400, color=INK, font=THAI, b=0, i=0):
    return {"t": text, "sz": sz, "c": color, "f": font, "b": b, "i": i}


def _run_xml(r):
    sp = ' xml:space="preserve"' if r["t"] != r["t"].strip() else ""
    f = esc(r["f"])
    return (f'<a:r><a:rPr b="{r["b"]}" i="{r["i"]}" lang="en-US" sz="{r["sz"]}" u="none" '
            f'cap="none" strike="noStrike"><a:solidFill><a:srgbClr val="{r["c"]}"/></a:solidFill>'
            f'<a:latin typeface="{f}"/><a:ea typeface="{f}"/><a:cs typeface="{f}"/>'
            f'<a:sym typeface="{f}"/></a:rPr><a:t{sp}>{esc(r["t"])}</a:t></a:r>')


def para(runs, algn="l", line=140000, before=0, after=0):
    """One <a:p>. `runs` is a list from run()."""
    if not runs:
        runs = [run("", 1200)]
    head = runs[0]
    body = "".join(_run_xml(r) for r in runs)
    return (f'<a:p><a:pPr indent="0" lvl="0" marL="0" marR="0" rtl="0" algn="{algn}">'
            f'<a:lnSpc><a:spcPct val="{line}"/></a:lnSpc>'
            f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef>'
            f'<a:spcAft><a:spcPts val="{after}"/></a:spcAft>'
            f'<a:buClr><a:srgbClr val="{head["c"]}"/></a:buClr>'
            f'<a:buSzPts val="{head["sz"]}"/><a:buFont typeface="{esc(head["f"])}"/><a:buNone/>'
            f'</a:pPr>{body}'
            f'<a:endParaRPr b="0" i="0" sz="{head["sz"]}" u="none" cap="none" strike="noStrike">'
            f'<a:solidFill><a:schemeClr val="dk1"/></a:solidFill><a:latin typeface="Calibri"/>'
            f'<a:ea typeface="Calibri"/><a:cs typeface="Calibri"/><a:sym typeface="Calibri"/>'
            f'</a:endParaRPr></a:p>')


SHADOW = ('<a:effectLst><a:outerShdw blurRad="76200" rotWithShape="0" algn="bl" '
          'dir="5400000" dist="12700"><a:srgbClr val="202124"><a:alpha val="5882"/>'
          '</a:srgbClr></a:outerShdw></a:effectLst>')


def shape(x, y, w, h, paras="", geom="rect", fill=None, shadow=False,
          anchor="t", pad=0, dash=None, adj=5340, line_clr=None):
    """A text box, a card, or a filled block - the same <p:sp> either way."""
    if geom == "roundRect":
        g = (f'<a:prstGeom prst="roundRect"><a:avLst>'
             f'<a:gd fmla="val {adj}" name="adj"/></a:avLst></a:prstGeom>')
    else:
        g = f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
    f = (f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>') if fill else "<a:noFill/>"
    if line_clr:
        dash_xml = f'<a:prstDash val="{dash}"/>' if dash else ""
        ln = (f'<a:ln cap="flat" cmpd="sng" w="9525"><a:solidFill>'
              f'<a:srgbClr val="{line_clr}"/></a:solidFill>{dash_xml}<a:round/></a:ln>')
    else:
        ln = "<a:ln><a:noFill/></a:ln>"
    eff = SHADOW if shadow else ""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="Shape {nid()}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{int(x)}" y="{int(y)}"/>'
            f'<a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>{g}{f}{ln}{eff}</p:spPr>'
            f'<p:txBody><a:bodyPr anchorCtr="0" anchor="{anchor}" bIns="{pad}" lIns="{pad}" '
            f'spcFirstLastPara="1" rIns="{pad}" wrap="square" tIns="{pad}"><a:noAutofit/>'
            f'</a:bodyPr><a:lstStyle/>{paras or para([run("", 1200)])}</p:txBody></p:sp>')


def text(x, y, w, h, paras, anchor="t"):
    return shape(x, y, w, h, "".join(paras), anchor=anchor)


def accent(x, y, w, h, color):
    """The 6px left edge bar the CI asks for (`Structural Framing`).

    Copied as a custom path from the source deck rather than drawn as a
    rectangle, so the bar's own corners stay rounded against the card's."""
    h = int(h)
    w = int(w)
    p = (f'<a:custGeom><a:rect b="b" l="l" r="r" t="t"/><a:pathLst>'
         f'<a:path extrusionOk="0" h="{h}" w="{w}">'
         f'<a:moveTo><a:pt x="152400" y="{h}"/></a:moveTo>'
         f'<a:lnTo><a:pt x="152400" y="{h}"/></a:lnTo>'
         f'<a:cubicBezTo><a:pt x="68232" y="{h}"/><a:pt x="0" y="{h-68232}"/>'
         f'<a:pt x="0" y="{h-152400}"/></a:cubicBezTo>'
         f'<a:lnTo><a:pt x="0" y="152400"/></a:lnTo>'
         f'<a:cubicBezTo><a:pt x="0" y="68232"/><a:pt x="68232" y="0"/>'
         f'<a:pt x="152400" y="0"/></a:cubicBezTo>'
         f'<a:lnTo><a:pt x="152400" y="0"/></a:lnTo>'
         f'<a:cubicBezTo><a:pt x="96288" y="0"/><a:pt x="50800" y="68232"/>'
         f'<a:pt x="50800" y="152400"/></a:cubicBezTo>'
         f'<a:lnTo><a:pt x="50800" y="{h-152400}"/></a:lnTo>'
         f'<a:cubicBezTo><a:pt x="50800" y="{h-68232}"/><a:pt x="96288" y="{h}"/>'
         f'<a:pt x="152400" y="{h}"/></a:cubicBezTo><a:close/></a:path></a:pathLst></a:custGeom>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="Shape {nid()}"/><p:cNvSpPr/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/>'
            f'<a:ext cx="{w}" cy="{h}"/></a:xfrm>{p}'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln>'
            f'</p:spPr><p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="0" lIns="0" '
            f'spcFirstLastPara="1" rIns="0" wrap="square" tIns="0"><a:noAutofit/></a:bodyPr>'
            f'<a:lstStyle/>{para([run("", 1200)])}</p:txBody></p:sp>')


def card(x, y, w, h, color=None, fill=WHITE):
    out = [shape(x, y, w, h, geom="roundRect", fill=fill, shadow=True)]
    if color:
        out.append(accent(x, y, w, h, color))
    return out


def pic(rid, x, y, w, h, descr=""):
    return (f'<p:pic><p:nvPicPr><p:cNvPr descr="{esc(descr)}" id="{nid()}" '
            f'name="Picture {nid()}"/><p:cNvPicPr preferRelativeResize="0"/><p:nvPr/>'
            f'</p:nvPicPr><p:blipFill rotWithShape="1"><a:blip r:embed="{rid}">'
            f'<a:alphaModFix/></a:blip><a:srcRect b="0" l="0" r="0" t="0"/><a:stretch/>'
            f'</p:blipFill><p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/>'
            f'<a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
            f'<a:ln><a:noFill/></a:ln></p:spPr></p:pic>')


def pill(x, y, w, h, fill, runs, algn="l", pad=114300):
    return shape(x, y, w, h, para(runs, algn=algn, line=120000),
                 geom="roundRect", fill=fill, anchor="ctr", pad=pad, adj=18000)


# --- page furniture -----------------------------------------------------------
def tags(left, right, y=TAG_Y):
    return [text(L, y, 5886450, TAG_H, [para([run(left, 1200, GREY, MONO)])]),
            text(6800850, y, 4578350, TAG_H,
                 [para([run(right, 1200, GREY, MONO)], algn="r")])]


def header(caption, page, total, logo_rid):
    return [
        pic(logo_rid, L, 609600, 228600, 228600, "โลโก้ SIWAK"),
        text(1143000, HEAD_Y, 1320800, 256540, [para([run("SIWAK", 1300, INK, THAI, b=1)])]),
        text(3479800, 641350, 5232400, 256540,
             [para([run(caption, 1300, MUTED, THAI)], algn="ctr")]),
        text(10058400, HEAD_Y, 1320800, 256540,
             [para([run(f"{page:02d} / {total}", 1300, BLUE, THAI, b=1)], algn="r")]),
        shape(L, RULE_Y, W, 10160, fill=LINE),
    ]


def title(plain, colored, color=BLUE, sz=3400, y=TITLE_Y):
    runs = [run(plain, sz, INK, KANIT, b=1)]
    if colored:
        runs.append(run(colored, sz, color, KANIT, b=1))
    return text(L, y, W, TITLE_H, [para(runs, line=110000)])
