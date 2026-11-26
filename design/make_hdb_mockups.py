"""Builds two mockups from the approved violet mockup, recoloured with the HDB facade palette:
  design/mockup-hdb.html       white surfaces, as on the blocks
  design/mockup-hdb-grey.html  the same colours, with every white changed to medium grey

Each colour has one job, and nothing is decorative:
  blue    Oikos itself: buttons, links, the active section, your messages, the key figure (HDB blue, #1A88D4)
  orange  "check this": short lease, notes to check, the alerts badge, and nothing else (HDB orange, #F5A55F)
  green / red  good or bad for you on prices and changes, as before; the middle of the price scale is neutral grey
Home types (planned, previewed on the price card): blue HDB, purple condo, gold landed.
Run: python design/make_hdb_mockups.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "mockup-violet.html").read_text(encoding="utf-8")

# Picked from the photo of an HDB facade, plus the dark window frames. Home-type colours for the planned HDB / condo / landed tags.
HDB = "--hdb-blue:#1A88D4; --hdb-orange:#F5A55F; --frame:#2B3036; --t-hdb:#1A88D4; --t-condo:#7B4FE0; --t-landed:#E0A400;"

LIGHT = {
    "white": f"""[data-theme="light"] {{ {HDB} --bg:#F5F5F2; --panel:#FFFFFF; --soft:#E6F1FA; --line:#E1E1DC; --ink:#1F2328; --dim:#5A5F66; --faint:#8A8F96;
      --violet:#1A88D4; --violet-ink:#0F5E99; --on-violet:#FFFFFF; --wall:#FFFFFF;
      --warn:#F5A55F; --warn-ink:#9A5A0C; --warn-bg:#FDF0E2;
      --good:#15803d; --good-bg:#E7F5EC; --bad:#C4261F; --bad-bg:#FCEAE8; --mid:#9AA0A6; --mid-bg:#EEEFF0; }}""",
    # Medium grey replaces every white. Text and the good/bad colours are darkened so they still read on grey;
    # text on the blue buttons is dark instead of white.
    "grey": f"""[data-theme="light"] {{ {HDB} --bg:#9EA0A3; --panel:#A9ABAE; --soft:#B6B8BB; --line:#86898D; --ink:#14171A; --dim:#2D3135; --faint:#43474C;
      --violet:#1A88D4; --violet-ink:#0B416E; --on-violet:#101316; --wall:#A9ABAE;
      --warn:#F5A55F; --warn-ink:#5E3304; --warn-bg:#C9A880;
      --good:#0B4D23; --good-bg:#8FBC9C; --bad:#7E1510; --bad-bg:#C79A96; --mid:#5B6066; --mid-bg:#B6B8BB; }}""",
}
DARK = f"""[data-theme="dark"] {{ {HDB} --bg:#14171A; --panel:#1D2125; --soft:#1E2B37; --line:#2E343A; --ink:#ECEDEE; --dim:#A3A9B0; --faint:#79808A;
      --violet:#4BA8EC; --violet-ink:#8CC8F5; --on-violet:#0C1116; --wall:#1D2125; --t-condo:#A88BFF; --t-landed:#F2C230;
      --warn:#F5A55F; --warn-ink:#F5A55F; --warn-bg:#33240F;
      --good:#4ade80; --good-bg:#14301f; --bad:#FF6B62; --bad-bg:#3A1A18; --mid:#79808A; --mid-bg:#262B30; }}"""

EXTRA = """
  /* blue is Oikos; orange is "check this"; green and red are good or bad for you. No decorative colour. */
  .hero::before { display:none; }
  .hero h1 .accent { color:var(--hdb-blue); }
  .when { background:var(--frame); color:#F5F5F2; }
  .tile.key { background:var(--hdb-blue); border-color:var(--hdb-blue); color:var(--on-violet); } .tile.key .k { color:var(--on-violet); opacity:.8; }
  .mark { background:var(--hdb-blue); }
  #nav button.on { box-shadow:inset 3px 0 0 var(--hdb-blue); }
  .tag { background:var(--soft); color:var(--violet-ink); }
  .you { background:var(--hdb-blue); color:var(--on-violet); }
  .trust { background:var(--soft); }
  /* orange: things to check */
  .badge { min-width:20px; height:20px; padding:0 6px; border-radius:99px; background:var(--warn); color:#1F2328; font-size:11.5px; font-weight:700; display:inline-grid; place-items:center; }
  .warnchip { display:inline-block; margin-left:6px; font-size:11.5px; font-weight:700; padding:1px 8px; border-radius:99px; background:var(--warn-bg); color:var(--warn-ink); }
  .warn { color:var(--warn-ink); font-weight:600; }
  .note { border-left:4px solid var(--warn); background:var(--warn-bg); border-radius:0 12px 12px 0; padding:10px 14px; font-size:14px; }
  .note b { color:var(--warn-ink); }
  /* home types (planned): blue HDB, purple condo, gold landed */
  .type { display:inline-block; vertical-align:2px; margin-left:8px; font-size:11px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; padding:2px 8px; border-radius:6px; color:#fff; }
  .type.hdb { background:var(--t-hdb); } .type.condo { background:var(--t-condo); } .type.landed { background:var(--t-landed); color:#1F2328; }
  .palette { display:flex; gap:6px; align-items:center; margin-top:8px; font-size:11.5px; color:var(--dim); }
  .palette i { width:18px; height:18px; border-radius:4px; box-shadow:inset 0 0 0 1px var(--line); }
"""


def rep(s, a, b, count=1):
    assert s.count(a) >= count, a[:80]
    return s.replace(a, b, count)


def build(kind, out, label):
    s = src
    start = s.index('[data-theme="light"]')
    end = s.index("* { box-sizing:border-box; }")
    s = s[:start] + LIGHT[kind] + "\n  " + DARK + "\n  " + s[end:]
    s = s.replace("</style>", EXTRA + ("  /* on grey, the bright blue is too faint for words: the headline uses the deep blue */\n"
                                        "  [data-theme=\"light\"] .hero h1 .accent { color:var(--violet-ink); }\n" if kind == "grey" else "") + "</style>", 1)
    s = s.replace("<title>Oikos mockup: violet</title>", f"<title>Oikos mockup: {label}</title>")
    s = s.replace("/* Direction A: electric violet for the brand", "/* HDB palette: blue for Oikos, orange for things to check; was: electric violet for the brand")
    # orange where something needs checking
    s = rep(s, "<span>Alerts<sup>05</sup></span><span></span>", '<span>Alerts<sup>05</sup></span><span class="badge">3</span>')
    s = rep(s, '<div class="kv"><span>Lease left: 54 years</span>', '<div class="kv"><span>Lease left: 54 years<span class="warnchip">Short lease</span></span>')
    s = rep(s, '        <div class="acts"><button class="btn primary">Save flat</button>',
            '        <div class="note"><b>Check this:</b> 54 years of lease left. If it won\'t last until the youngest buyer turns 95, you may be able to use less CPF and borrow less.</div>\n'
            '        <div class="acts"><button class="btn primary">Save flat</button>')
    s = rep(s, '<td class="bad">54 years</td>', '<td class="warn">54 years, short lease</td>')
    # the planned home-type tag
    # the ticker keeps green for a rise and red for a fall (the source mockup had them grey)
    s = rep(s, '`<span class="dim">${c >= 0 ? "▲" : "▼"} ${Math.abs(c).toFixed(1)}%</span>`',
            '`<span class="${c >= 0 ? "good" : "bad"}">${c >= 0 ? "▲" : "▼"} ${Math.abs(c).toFixed(1)}%</span>`')
    s = rep(s, '<div class="name">Blk 177 Ang Mo Kio Ave 4</div>', '<div class="name">Blk 177 Ang Mo Kio Ave 4<span class="type hdb">HDB</span></div>')
    s = rep(s, '<div class="banner">Mockup · sample figures',
            f'<div class="banner">Mockup · {label} <span class="palette"><i style="background:var(--hdb-blue)"></i><i style="background:var(--hdb-orange)"></i><i style="background:var(--wall)"></i>'
            '<span style="margin-left:6px">Types:</span><span class="type hdb" style="margin:0">HDB</span><span class="type condo" style="margin:0">Condo</span><span class="type landed" style="margin:0">Landed</span></span>')
    (HERE / out).write_text(s, encoding="utf-8")


build("white", "mockup-hdb.html", "HDB colours")
build("grey", "mockup-hdb-grey.html", "HDB colours, grey")
print("made design/mockup-hdb.html and design/mockup-hdb-grey.html")
