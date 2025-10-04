# Polygon-Factory
Generate customizable 3D solid polygons with holes, nut pockets, and export to STL for 3D printing.
polyGenV5 — STL Polygon Generator with Radial & Vertical Holes (M4/M5)

Generates hollow polygonal crowns (triangles, hexagons, octagons, etc.) in STL format with:

Radial holes (on sides)

Hex pockets for nuts (countersinks)

Vertical holes at vertices (optional)

BIG mode --B which doubles radial holes (2 per side) and adds vertical holes at edge midpoints (in addition to vertices)

💡 Perfect for mechanical projects, modular fastenings, accessory mounts, etc.

✨ Features

N sides of your choice (≥ 3)

Size selectable via --size_mode:

side_length = edge length

across_flats = distance between flats

circum_diameter = circumscribed diameter

Parametric Z thickness and radial width

Holes sized for M4/M5 (diameter + printing clearance)

Hex pockets (top/bottom and radial) with configurable depth

Generates 5 STL files in cascade (see below)

🔧 Requirements

Python 3.9–3.12

CadQuery 2.x

Quick install:

python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install "cadquery>=2.3,<3"

⚡ Quick Usage (PowerShell)
py -3.10 .\polyGenV5.py --N 6 --A 12.5 --size_mode side_length `
  --H 1.5 --w 1.5 `
  --hole_d 5.0 --hole_clear 2.0 `
  --nut_af 8.0 --nut_depth 4.0 `
  --vertex_holes `
  --D 4 `
  --B `
  --out mypolygon.stl


Units & conventions (IMPORTANT)

--A, --w, --H are in cm

Everything else (holes, clearances, hex pockets) is in mm

--size_mode side_length ⇒ A = edge length

📂 Generated Files (from --out mypolygon.stl)

Base: mypolygon.stl — Hollow prism + radial holes (1 per side)

Radial upgrade: mypolygon_geminiUpgrade.stl — Adds hex pockets for radial holes

Radial upgrade with depth D: mypolygon_geminiUpgrade_D{D}mm.stl — Same as 2 but radial hex pockets depth = D mm

Vertex holes: mypolygon_vertexHoles.stl — Adds vertical holes at vertices + top/bottom hex pockets

BIG object: mypolygon_bigObject.stl — 2 radial holes per side + duplicated radial hex pockets

If --vertex_holes is active: mypolygon_bigObject_vertexHoles.stl — Adds vertical holes at edge midpoints in addition to vertices

⚙️ CLI Options

Geometry & size

--N (int, required) : number of sides (≥ 3)

--A (float, cm, required) : dimension according to --size_mode

--size_mode : side_length | across_flats | circum_diameter (default: across_flats)

--w (float, cm, required) : radial width (thickness between apothems)

--H (float, cm, required) : Z thickness (prism height)

--rotate (deg) : global rotation around Z

--z_offset (mm) : Z offset for radial hole plane

Radial holes (sides)

--hole_d (mm) : nominal diameter (M4 ≈ 4.5; M5 ≈ 5.0)

--hole_clear (mm) : clearance (0.3–0.7 mm typical FDM; 2.0 mm if you want very loose)

Hex pockets for nuts (radial & vertical)

--nut_af (mm) : across-flats of nut (M4 = 7.0; M5 = 8.0)

--nut_clear (mm) : clearance on flats (e.g., 0.3–0.7)

--nut_depth (mm) : depth per face (radial + top/bottom)

--D (mm) : also generate STL where only radial hex pockets have depth = D

Vertical holes (top/bottom)

--vertex_holes (bool) : add vertical holes at vertices + top/bottom hex pockets

--vertex_hole_d (mm) : nominal diameter (default = hole_d)

--vertex_hole_clear (mm) : clearance (default = hole_clear)

--vertex_depth (mm) : depth if not through (default = through)

BIG mode --B (5th file)

Doubles radial holes → 2 per side (positions 1/3 and 2/3)

If --vertex_holes is active: also adds vertical holes at edge midpoints (top/bottom hex pockets included)

🛠️ Printing Tips (Bambu Studio)

To reduce weight (g):

Infill: 15–20 % (Gyroid / Lines)

Walls: 2 (3 if needed)

Top/Bottom layers: 3–4

Layer height: 0.20 mm (or 0.28 mm for faster prints)

Troubleshooting

“w too large compared to A” → reduce --w or increase --A

Hex pockets merging → decrease --nut_depth or increase --w

Clearance too tight/loose → adjust --hole_clear / --nut_clear
