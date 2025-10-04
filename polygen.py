#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import argparse
import cadquery as cq
import os

# =============================================================================
# Basic Geometry
# =============================================================================

def regular_polygon_circ_diameter_from_A(N: int, A_mm: float, size_mode: str) -> float:
    """
    Returns the circumscribed circle diameter of a regular N-gon, according to size_mode convention.
    - across_flats : A = 2 * apothem => R_circ = apothem / cos(pi/N)
    - circum_diameter : A = circumscribed diameter
    - side_length : A = edge length => R_circ = s / (2 sin(pi/N))
    """
    if N < 3:
        raise ValueError("N must be >= 3")
    if size_mode == "across_flats":
        R_in = A_mm / 2.0
        R_circ = R_in / math.cos(math.pi / N)
        return 2.0 * R_circ
    elif size_mode == "circum_diameter":
        return A_mm
    elif size_mode == "side_length":
        s = A_mm
        R_circ = s / (2.0 * math.sin(math.pi / N))
        return 2.0 * R_circ
    else:
        raise ValueError("unknown size_mode")


# -----------------------------------------------------------------------------
# SHELL CONSTRUCTION ONLY (object 1, without radial holes)
# -----------------------------------------------------------------------------
def build_shell_only(
    N: int,
    A_cm: float,
    w_cm: float,
    H_cm: float,
    size_mode: str = "across_flats",
    z_offset_mm: float = 0.0,
    rotate_deg: float = 0.0,
):
    # cm -> mm
    A_mm = A_cm * 10.0
    w_mm = w_cm * 10.0
    H_mm = H_cm * 10.0

    if N < 3:
        raise ValueError("N must be >= 3")
    if A_mm <= 0 or w_mm <= 0 or H_mm <= 0:
        raise ValueError("A, w, H must be > 0")

    Dcirc_out = regular_polygon_circ_diameter_from_A(N, A_mm, size_mode)
    R_circ_out = Dcirc_out / 2.0
    R_in_out = R_circ_out * math.cos(math.pi / N)       # outer apothem
    R_in_in = R_in_out - w_mm                           # inner apothem
    if R_in_in <= 0:
        raise ValueError("w too large compared to A")
    R_circ_in = R_in_in / math.cos(math.pi / N)

    # outer edge length
    s_edge_out = 2.0 * R_in_out * math.tan(math.pi / N)

    # hollow solid without holes
    wp = cq.Workplane("XY")
    if abs(rotate_deg) > 1e-9:
        wp = wp.transformed(rotate=(0, 0, rotate_deg))
    shell = wp.polygon(N, 2.0 * R_circ_out).extrude(H_mm)
    shell = (
        shell.faces(">Z").workplane()
        .polygon(N, 2.0 * R_circ_in)
        .cutBlind(-H_mm)
    )

    angle_offset = math.radians(rotate_deg)
    angles_mid_edges = [2.0 * math.pi * (k + 0.5) / N + angle_offset for k in range(N)]
    angles_vertices  = [2.0 * math.pi *  k        / N + angle_offset for k in range(N)]
    z_center = H_mm / 2.0 + float(z_offset_mm)

    geom_params = {
        "N": N,
        "H_mm": H_mm,
        "w_mm": w_mm,
        "R_in_out": R_in_out,
        "R_in_in": R_in_in,
        "R_circ_out": R_circ_out,
        "R_circ_in": R_circ_in,
        "z_center": z_center,
        "angles_mid_edges": angles_mid_edges,
        "angles_vertices": angles_vertices,
        "angle_offset": angle_offset,
        "s_edge_out_mm": s_edge_out,
    }
    return shell, geom_params


def build_model(
    N: int,
    A_cm: float,
    w_cm: float,
    H_cm: float,
    size_mode: str = "across_flats",
    hole_d_mm: float = 4.5,
    hole_clear_mm: float = 0.5,  # clearance on holes (mm)
    z_offset_mm: float = 0.0,
    rotate_deg: float = 0.0,
):
    """
    Builds the hollow polygonal prism + radial holes (1 per side).
    Returns (solid, geom_params) where geom_params contains all useful dimensions.
    """
    # cm -> mm
    A_mm = A_cm * 10.0
    w_mm = w_cm * 10.0
    H_mm = H_cm * 10.0

    if N < 3:
        raise ValueError("N must be >= 3")
    if A_mm <= 0 or w_mm <= 0 or H_mm <= 0:
        raise ValueError("A, w, H must be > 0")
    if abs(z_offset_mm) > H_mm / 2:
        raise ValueError("z_offset_mm must be in [-H/2, H/2]")

    # Polygonal geometry
    Dcirc_out = regular_polygon_circ_diameter_from_A(N, A_mm, size_mode)
    R_circ_out = Dcirc_out / 2.0
    R_in_out = R_circ_out * math.cos(math.pi / N)  # outer apothem
    R_in_in = R_in_out - w_mm                      # inner apothem
    if R_in_in <= 0:
        raise ValueError("w too large compared to A")
    R_circ_in = R_in_in / math.cos(math.pi / N)

    # outer edge length
    s_edge_out = 2.0 * R_in_out * math.tan(math.pi / N)

    # hollow solid
    wp = cq.Workplane("XY")
    if abs(rotate_deg) > 1e-9:
        wp = wp.transformed(rotate=(0, 0, rotate_deg))
    solid = wp.polygon(N, 2.0 * R_circ_out).extrude(H_mm)
    solid = (
        solid.faces(">Z").workplane()
        .polygon(N, 2.0 * R_circ_in)
        .cutBlind(-H_mm)
    )

    # Angles: mid-edges and vertices (include global rotation)
    angle_offset = math.radians(rotate_deg)
    angles_mid_edges = [2.0 * math.pi * (k + 0.5) / N + angle_offset for k in range(N)]
    angles_vertices  = [2.0 * math.pi *  k        / N + angle_offset for k in range(N)]
    z_center = H_mm / 2.0 + z_offset_mm

    # Radial cylindrical holes (M4 by default) with clearance
    margin = 2.0
    L = w_mm + margin
    eff_hole_d = hole_d_mm + max(0.0, hole_clear_mm)  # effective diameter with clearance
    r_cyl = eff_hole_d / 2.0
    r_axis = (R_in_out + R_in_in) / 2.0

    for a in angles_mid_edges:
        dx, dy = math.cos(a), math.sin(a)
        r_start = r_axis - L / 2.0
        p0 = cq.Vector(r_start * dx, r_start * dy, z_center)
        dir_vec = cq.Vector(dx, dy, 0.0)  # radial direction
        cyl = cq.Solid.makeCylinder(r_cyl, L, p0, dir_vec)
        solid = solid.cut(cyl)

    geom_params = {
        "N": N,
        "H_mm": H_mm,
        "w_mm": w_mm,
        "R_in_out": R_in_out,  # outer apothem
        "R_in_in": R_in_in,    # inner apothem
        "R_circ_out": R_circ_out,  # outer circ radius (outer vertex)
        "R_circ_in": R_circ_in,    # inner circ radius (inner vertex)
        "z_center": z_center,
        "angles_mid_edges": angles_mid_edges,
        "angles_vertices": angles_vertices,
        "angle_offset": angle_offset,
        "s_edge_out_mm": s_edge_out,   # outer edge length
    }
    return solid, geom_params


# =============================================================================
# TOOLS: radial plane + radial hex pockets
# =============================================================================

def wp_radial_plane(radius: float, angle: float, z: float) -> cq.Workplane:
    """
    Creates a local plane whose normal points radially (outwards),
    centered at (radius, angle, z). We fix xDir = global Z for a stable orientation.
    """
    origin = cq.Vector(radius * math.cos(angle), radius * math.sin(angle), z)
    normal = cq.Vector(math.cos(angle), math.sin(angle), 0.0)  # pure radial
    xdir = cq.Vector(0, 0, 1)  # keep global Z as local X axis
    plane = cq.Plane(origin=origin, xDir=xdir, normal=normal)
    return cq.Workplane(plane)


def wp_radial_plane_with_tangent_offset(radius: float, angle: float, z: float, t_offset: float) -> cq.Workplane:
    """
    Variant: same radial plane but offset by t_offset along the
    tangential direction of the edge: t_hat = (-sin a, +cos a).
    """
    ca, sa = math.cos(angle), math.sin(angle)
    t_x, t_y = -sa, +ca  # unit tangent (counterclockwise sense)
    x = radius * ca + t_offset * t_x
    y = radius * sa + t_offset * t_y
    origin = cq.Vector(x, y, z)
    normal = cq.Vector(ca, sa, 0.0)
    xdir = cq.Vector(0, 0, 1)
    plane = cq.Plane(origin=origin, xDir=xdir, normal=normal)
    return cq.Workplane(plane)


def add_m4_hex_countersinks(
    solid_in: cq.Workplane,
    geom_params: dict,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
    nut_depth_mm: float = 1.8,
) -> cq.Workplane:
    """
    Adds hexagonal pockets for M4 nuts on both sides of each radial hole.
    Robust orientation via a dedicated radial plane.
    """
    print("Adding radial hex pockets ...")
    # across-flats + clearance -> circumscribed diameter of the hexagon
    D_hex = (nut_af_mm + max(0.0, nut_clear_mm)) / math.cos(math.pi / 6.0)

    R_face_out = geom_params["R_in_out"]  # outer apothem
    R_face_in  = geom_params["R_in_in"]   # inner apothem
    z_center   = geom_params["z_center"]
    angles     = geom_params["angles_mid_edges"]
    w_mm       = geom_params["w_mm"]

    if 2.0 * nut_depth_mm > w_mm:
        print(f"WARNING: 2*nut_depth ({2.0*nut_depth_mm:.2f} mm) > w ({w_mm:.2f} mm) : radial pockets may meet.")

    solid = solid_in
    eps = 0.05  # mm : avoid coplanar faces
    for a in angles:
        # Outside : extrusion towards inside (negative)
        wp_out = wp_radial_plane(R_face_out + eps, a, z_center)
        hex_out = wp_out.polygon(6, D_hex).extrude(-nut_depth_mm)
        solid = solid.cut(hex_out)

        # Inside : extrusion towards outside (positive)
        wp_in = wp_radial_plane(R_face_in - eps, a, z_center)
        hex_in = wp_in.polygon(6, D_hex).extrude(+nut_depth_mm)
        solid = solid.cut(hex_in)

    return solid.clean()


def add_m4_hex_countersinks_with_depth(
    solid_in: cq.Workplane,
    geom_params: dict,
    d_mm: float,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
) -> cq.Workplane:
    """
    Hex milling variant for M4 nuts, with custom depth d_mm.
    (applied on radial pockets)
    """
    print(f"Radial hex pockets with custom depth = {d_mm:.2f} mm ...")
    nut_depth_mm = float(d_mm)
    D_hex = (nut_af_mm + max(0.0, nut_clear_mm)) / math.cos(math.pi / 6.0)

    R_face_out = geom_params["R_in_out"]
    R_face_in  = geom_params["R_in_in"]
    z_center   = geom_params["z_center"]
    angles     = geom_params["angles_mid_edges"]
    w_mm       = geom_params["w_mm"]

    if 2.0 * nut_depth_mm > w_mm:
        print(f"WARNING: 2*D ({2.0*nut_depth_mm:.2f} mm) > w ({w_mm:.2f} mm) : radial pockets may meet.")

    solid = solid_in
    eps = 0.05  # mm
    for a in angles:
        wp_out = wp_radial_plane(R_face_out + eps, a, z_center)
        hex_out = wp_out.polygon(6, D_hex).extrude(-nut_depth_mm)
        solid = solid.cut(hex_out)

        wp_in = wp_radial_plane(R_face_in - eps, a, z_center)
        hex_in = wp_in.polygon(6, D_hex).extrude(+nut_depth_mm)
        solid = solid.cut(hex_in)

    return solid.clean()


# =============================================================================
# STEP 4: VERTICAL HOLES (VERTICES / EDGES) + HEX POCKETS (TOP & BOTTOM)
# =============================================================================

def add_vertical_holes_with_hex_at_angles(
    solid_in: cq.Workplane,
    geom_params: dict,
    angles_list,
    hole_d_mm: float = 4.5,
    hole_clear_mm: float = 0.5,
    through: bool = True,
    depth_mm: float = None,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
    nut_depth_mm: float = 1.8,
    center_ref: str = "circum",  # "circum" (vertices) or "apoth" (edge centers)
) -> cq.Workplane:
    """
    Adds, for each angle in angles_list:
    1) A vertical hole (Z axis) centered radially:
       - center_ref="circum" -> r_center = (R_circ_out + R_circ_in) / 2 (ideal for VERTICES)
       - center_ref="apoth"  -> r_center = (R_in_out + R_in_in) / 2   (ideal for EDGE CENTERS)
    2) Two hexagonal pockets (top and bottom) like for radial holes (same parameters).

    - If through = False, depth_mm must be > 0 (drill depth).
    """
    H_mm = geom_params["H_mm"]
    R_circ_out = geom_params["R_circ_out"]
    R_circ_in  = geom_params["R_circ_in"]
    R_in_out   = geom_params["R_in_out"]
    R_in_in    = geom_params["R_in_in"]

    if not through:
        if depth_mm is None or depth_mm <= 0:
            raise ValueError("If 'through' is False, 'depth_mm' must be > 0.")

    solid = solid_in

    # Effective dimensions
    eff_hole_d = hole_d_mm + max(0.0, hole_clear_mm)
    r_cyl = eff_hole_d / 2.0

    # Hex (top/bottom): across-flats + clearance -> circumscribed diameter
    D_hex = (nut_af_mm + max(0.0, nut_clear_mm)) / math.cos(math.pi / 6.0)

    # Heights / margins
    epsZ = 0.05
    h_cyl = (H_mm + 2 * epsZ) if through else float(depth_mm)

    # Check hex pocket in Z
    if 2.0 * nut_depth_mm > H_mm:
        print(f"WARNING: 2*nut_depth ({2.0*nut_depth_mm:.2f} mm) > H ({H_mm:.2f} mm) : top/bottom pockets may meet.")

    for a in angles_list:
        if center_ref == "apoth":
            r_center = 0.5 * (R_in_out + R_in_in)       # center in thickness (ref apothems)
        else:
            r_center = 0.5 * (R_circ_out + R_circ_in)   # center in thickness (ref circ radii)

        x = r_center * math.cos(a)
        y = r_center * math.sin(a)

        # 1) VERTICAL DRILLING
        p0 = cq.Vector(x, y, H_mm + epsZ)  # base of the cylinder above the top face
        dir_vec = cq.Vector(0.0, 0.0, -1.0)
        cyl = cq.Solid.makeCylinder(r_cyl, h_cyl, p0, dir_vec)
        solid = solid.cut(cyl)

        # 2) HEX POCKETS (TOP & BOTTOM)
        # Top: extrude towards -Z
        wp_top = cq.Workplane("XY").workplane(offset=H_mm + epsZ).center(x, y)
        hex_top = wp_top.polygon(6, D_hex).extrude(-nut_depth_mm)
        solid = solid.cut(hex_top)

        # Bottom: extrude towards +Z
        wp_bot = cq.Workplane("XY").workplane(offset=-epsZ).center(x, y)
        hex_bot = wp_bot.polygon(6, D_hex).extrude(+nut_depth_mm)
        solid = solid.cut(hex_bot)

    return solid.clean()


def add_vertex_vertical_holes_with_hex(
    solid_in: cq.Workplane,
    geom_params: dict,
    hole_d_mm: float = 4.5,
    hole_clear_mm: float = 0.5,
    through: bool = True,
    depth_mm: float = None,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
    nut_depth_mm: float = 1.8,
) -> cq.Workplane:
    """Original version: vertical holes at vertices + top/bottom hex pockets."""
    angles_v = geom_params["angles_vertices"]
    return add_vertical_holes_with_hex_at_angles(
        solid_in, geom_params, angles_v,
        hole_d_mm, hole_clear_mm, through, depth_mm,
        nut_af_mm, nut_clear_mm, nut_depth_mm,
        center_ref="circum",  # vertices: reference circumscribed radii
    )

    


# =============================================================================
# MODE BIG OBJECT (radial : 2 trous par cote) + poches radiales dupliquees
# =============================================================================

def build_model_big(
    N: int,
    A_cm: float,
    w_cm: float,
    H_cm: float,
    size_mode: str = "across_flats",
    hole_d_mm: float = 4.5,
    hole_clear_mm: float = 0.5,
    z_offset_mm: float = 0.0,
    rotate_deg: float = 0.0,
    delta_override=None,
):
    """
    'BIG' variant: same polygonal prism, but RADIAL HOLES:
    - 2 per side, placed at ±(s_edge_out/6) along the tangent of the edge
      (=> positions at 1/3 and 2/3 of the edge).
    """
    # Reuse base geometry from build_model for the volume
    solid, geom = build_model(
        N, A_cm, w_cm, H_cm,
        size_mode=size_mode,
        hole_d_mm=0.0,        # temporarily neutralize hole diameter to NOT create holes here
        hole_clear_mm=0.0,    # (recalculated below)
        z_offset_mm=z_offset_mm,
        rotate_deg=rotate_deg
    )

    # Recreate radial holes, but with 2 per slice
    A_mm = A_cm * 10.0
    w_mm = w_cm * 10.0
    H_mm = H_cm * 10.0

    Dcirc_out = regular_polygon_circ_diameter_from_A(N, A_mm, size_mode)
    R_circ_out = Dcirc_out / 2.0
    R_in_out = R_circ_out * math.cos(math.pi / N)
    R_in_in = R_in_out - w_mm
    R_circ_in = R_in_in / math.cos(math.pi / N)

    s_edge_out = 2.0 * R_in_out * math.tan(math.pi / N)   # outer edge length
    if delta_override is not None:
        delta = delta_override
    else:
        delta = s_edge_out / 6.0

    # Angles
    angles_mid_edges = geom["angles_mid_edges"]
    z_center = geom["z_center"]

    # Radial drillings
    margin = 2.0
    L = w_mm + margin
    eff_hole_d = hole_d_mm + max(0.0, hole_clear_mm)
    r_cyl = eff_hole_d / 2.0
    r_axis = (R_in_out + R_in_in) / 2.0

    for a in angles_mid_edges:
        dx, dy = math.cos(a), math.sin(a)            # radial normal
        tx, ty = -math.sin(a), +math.cos(a)          # tangent
        r_start = r_axis - L / 2.0

        for t_off in (-delta, +delta):
            # Tangentially offset start point
            x0 = r_start * dx + t_off * tx
            y0 = r_start * dy + t_off * ty
            print(f"Coordinates of a radial hole (x, y, z): ({x0:.3f}, {y0:.3f}, {z_center:.3f})")
            p0 = cq.Vector(x0, y0, z_center)
            dir_vec = cq.Vector(dx, dy, 0.0)
            cyl = cq.Solid.makeCylinder(r_cyl, L, p0, dir_vec)
            solid = solid.cut(cyl)

    # Update some useful parameters / delta for BIG mode
    geom_big = dict(geom)
    geom_big.update({
        "s_edge_out_mm": s_edge_out,
        "big_tangent_offsets_mm": (-delta, +delta),
        "R_in_out": R_in_out,
        "R_in_in": R_in_in,
        "R_circ_out": R_circ_out,
        "R_circ_in": R_circ_in,
        "H_mm": H_mm,
        "w_mm": w_mm,
    })
    return solid, geom_big


def add_m_hex_countersinks_radial_big(
    solid_in: cq.Workplane,
    geom_params: dict,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
    nut_depth_mm: float = 1.8,
) -> cq.Workplane:
    """
    Radial hex pockets for BIG mode: duplicate pockets at both
    tangential positions ±delta (matching the 2 holes per side).
    """
    print("[BIG] Radial hex pockets (x2 per side) ...")
    D_hex = (nut_af_mm + max(0.0, nut_clear_mm)) / math.cos(math.pi / 6.0)

    R_face_out = geom_params["R_in_out"]
    R_face_in  = geom_params["R_in_in"]
    z_center   = geom_params["z_center"]
    angles     = geom_params["angles_mid_edges"]
    w_mm       = geom_params["w_mm"]
    t_offsets  = geom_params.get("big_tangent_offsets_mm", (0.0,))

    if 2.0 * nut_depth_mm > w_mm:
        print(f"2*nut_depth ({2.0*nut_depth_mm:.2f} mm) > w ({w_mm:.2f} mm): radial pockets may overlap.")

    solid = solid_in
    eps = 0.05

    for a in angles:
        for t_off in t_offsets:
            # Outer
            wp_out = wp_radial_plane_with_tangent_offset(R_face_out + eps, a, z_center, t_off)
            hex_out = wp_out.polygon(6, D_hex).extrude(-nut_depth_mm)
            solid = solid.cut(hex_out)

            # Inner
            wp_in = wp_radial_plane_with_tangent_offset(R_face_in - eps, a, z_center, t_off)
            hex_in = wp_in.polygon(6, D_hex).extrude(+nut_depth_mm)
            solid = solid.cut(hex_in)

    return solid.clean()


# =============================================================================

# -------------------- V6: assembly (bevel) helper --------------------
def compute_scale_top_from_assembly_angle(geom_params: dict, assembly_angle_deg: float) -> float:
    """
    Compute the scaling factor of the top face (scale_top) to apply
    to the outer prism in order to achieve the requested assembly angle.
    geom_params must contain at least 'R_circ_out' and 'H_mm'.
    """
    R = geom_params.get("R_circ_out", None)
    H = geom_params.get("H_mm", None)
    if R is None or H is None:
        raise ValueError("geom_params must contain 'R_circ_out' and 'H_mm'")

    theta_deg = (180.0 - float(assembly_angle_deg)) / 2.0
    theta_rad = math.radians(theta_deg)
    # radial inset = H * tan(theta)
    inset = 0.0 if abs(theta_rad) < 1e-12 else float(H) * math.tan(theta_rad)
    scale_top = (R - inset) / R if R > 1e-9 else 1.0
    # clamp to avoid collapse
    scale_top = max(0.01, min(1.0, scale_top))
    return scale_top


def build_tapered_prism_from_geom(geom_params: dict, scale_top: float):
    """
    Build a truncated polygonal prism (cadquery Workplane / solid)
    corresponding to the external geometry contained in geom_params,
    with the top face reduced by scale_top.
    - keeps the inner hollow (R_circ_in) by subtracting it to obtain
      a ring if geom_params contains 'R_circ_in'.
    Returns a cadquery object (compatible with cq.exporters.export).
    """
    N = int(geom_params.get("N", 6))
    H_mm = float(geom_params.get("H_mm", 1.0))
    R_out = float(geom_params.get("R_circ_out", 10.0))
    D_bottom = 2.0 * R_out
    D_top = D_bottom * float(scale_top)

    # build truncated outer prism (loft between two polygons)
    tapered_outer = cq.Workplane("XY").polygon(N, D_bottom).workplane(offset=H_mm).polygon(N, D_top).loft()

    # if we have an inner radius, recreate the inner part and subtract it
    R_in = geom_params.get("R_circ_in", None)
    if R_in is not None and float(R_in) > 1e-6:
        D_bottom_in = 2.0 * float(R_in)
        D_top_in = D_bottom_in * float(scale_top)
        tapered_inner = cq.Workplane("XY").polygon(N, D_bottom_in).workplane(offset=H_mm).polygon(N, D_top_in).loft()
        # cut to obtain ring (same logic as build_model with extrude/cut)
        tapered_outer = tapered_outer.cut(tapered_inner)

    return tapered_outer
# ----------------------------------------------------------------------

def apply_assembly_angle(mesh, angle_deg):
    """
    Apply a bevel on external faces to allow assembly
    at a certain angle (e.g. 120°).
    The operation slightly reduces the top face to create the desired angle.
    """
    import numpy as np
    from math import radians, tan

    angle_rad = radians(angle_deg)
    # reduction factor proportional to angle
    shrink_factor = tan(angle_rad / 2.0)

    vertices = mesh.vertices.copy()
    zmax = np.max(vertices[:, 2])
    zmin = np.min(vertices[:, 2])
    height = zmax - zmin

    for v in vertices:
        # relative distance from base
        rel_z = (v[2] - zmin) / height
        scale = 1.0 - (shrink_factor * rel_z)
        v[0] *= scale
        v[1] *= scale

    mesh.vertices = vertices
    return mesh


# =============================================================================

# -------------------- V6: assembly (bevel) helper --------------------
def compute_scale_top_from_assembly_angle(geom_params: dict, assembly_angle_deg: float) -> float:
    """
    Compute the scaling factor of the top face (scale_top) to apply
    to the outer prism to obtain the requested assembly angle.
    geom_params must contain at least 'R_circ_out' and 'H_mm'.
    """
    R = geom_params.get("R_circ_out", None)
    H = geom_params.get("H_mm", None)
    if R is None or H is None:
        raise ValueError("geom_params must contain 'R_circ_out' and 'H_mm'")

    theta_deg = (180.0 - float(assembly_angle_deg)) / 2.0
    theta_rad = math.radians(theta_deg)
    # radial inset = H * tan(theta)
    inset = 0.0 if abs(theta_rad) < 1e-12 else float(H) * math.tan(theta_rad)
    scale_top = (R - inset) / R if R > 1e-9 else 1.0
    # clamp to avoid collapse
    scale_top = max(0.01, min(1.0, scale_top))
    return scale_top


def build_tapered_prism_from_geom(geom_params: dict, scale_top: float):
    """
    Build a truncated polygonal prism (cadquery Workplane / solid)
    corresponding to the external geometry contained in geom_params
    with the top face reduced by scale_top.
    - keeps the inner hollow (R_circ_in) by subtracting it to obtain
      a ring if geom_params contains 'R_circ_in'.
    Returns a cadquery object (compatible with cq.exporters.export).
    """
    N = int(geom_params.get("N", 6))
    H_mm = float(geom_params.get("H_mm", 1.0))
    R_out = float(geom_params.get("R_circ_out", 10.0))
    D_bottom = 2.0 * R_out
    D_top = D_bottom * float(scale_top)

    # build truncated outer prism (loft between two polygons)
    tapered_outer = cq.Workplane("XY").polygon(N, D_bottom).workplane(offset=H_mm).polygon(N, D_top).loft()

    # if we have an inner radius, recreate the inner part and subtract it
    R_in = geom_params.get("R_circ_in", None)
    if R_in is not None and float(R_in) > 1e-6:
        D_bottom_in = 2.0 * float(R_in)
        D_top_in = D_bottom_in * float(scale_top)
        tapered_inner = cq.Workplane("XY").polygon(N, D_bottom_in).workplane(offset=H_mm).polygon(N, D_top_in).loft()
        # cut to obtain the ring (same logic as build_model with extrude/cut)
        tapered_outer = tapered_outer.cut(tapered_inner)

    return tapered_outer
# ----------------------------------------------------------------------

def apply_assembly_angle(mesh, angle_deg):
    """
    Apply a bevel on external faces to allow assembly
    at a certain angle (e.g. 120°).
    The operation slightly reduces the top face to create the desired angle.
    """
    import numpy as np
    from math import radians, tan

    angle_rad = radians(angle_deg)
    # reduction factor proportional to angle
    shrink_factor = tan(angle_rad / 2.0)

    vertices = mesh.vertices.copy()
    zmax = np.max(vertices[:, 2])
    zmin = np.min(vertices[:, 2])
    height = zmax - zmin

    for v in vertices:
        # relative distance from base
        rel_z = (v[2] - zmin) / height
        scale = 1.0 - (shrink_factor * rel_z)
        v[0] *= scale
        v[1] *= scale

    mesh.vertices = vertices
    return mesh


# ---------------------------------------------------------------------------
# V6b: reference of a side facet (k) after truncated machining (scale_top)
# ---------------------------------------------------------------------------
def _side_face_frame(N: int, k: int, geom: dict, scale_top: float, z: float = None):
    """
    Returns a dict with:
      - a_mid : mid-edge angle (XY)
      - p_out : point on the OUTER face at level z (mid-edge)
      - p_in  : point on the INNER face at level z (mid-edge)
      - n_hat : unit outward normal of the facet (constant on the face)
      - t_hat : unit tangent along the edge (horizontal XY)
      - u_hat : unit "upward" direction in the plane of the face
      - t_eff_mm : projected thickness along n_hat (useful for machining lengths)
    """
    a_off = float(geom.get("angle_offset", 0.0))
    H = float(geom["H_mm"])
    if z is None:
        z = float(geom.get("z_center", H/2.0))

    # circumscribed radii ext./int. at bottom, and "top" versions after scale_top
    Rb_out = float(geom["R_circ_out"])
    Rb_in  = float(geom["R_circ_in"])
    Rt_out = Rb_out * float(scale_top)
    Rt_in  = Rb_in  * float(scale_top)

    # vertex angles k and k+1
    a1 = 2.0*math.pi * k / N + a_off
    a2 = 2.0*math.pi * (k+1) / N + a_off
    a_mid = 0.5*(a1 + a2)

    # Four edge points (bottom/top) OUTER side
    B1 = cq.Vector(Rb_out*math.cos(a1), Rb_out*math.sin(a1), 0.0)
    B2 = cq.Vector(Rb_out*math.cos(a2), Rb_out*math.sin(a2), 0.0)
    T1 = cq.Vector(Rt_out*math.cos(a1), Rt_out*math.sin(a1), H)
    T2 = cq.Vector(Rt_out*math.cos(a2), Rt_out*math.sin(a2), H)

    # Same for INNER side
    b1 = cq.Vector(Rb_in*math.cos(a1), Rb_in*math.sin(a1), 0.0)
    b2 = cq.Vector(Rb_in*math.cos(a2), Rb_in*math.sin(a2), 0.0)
    t1 = cq.Vector(Rt_in*math.cos(a1), Rt_in*math.sin(a1), H)
    t2 = cq.Vector(Rt_in*math.cos(a2), Rt_in*math.sin(a2), H)

    # Edge tangent (horizontal)
    t_vec = (B2 - B1)
    t_hat = t_vec.normalized()

    # Upward direction on the face (from mid-bottom to mid-top)
    M_b_out = 0.5*(B1 + B2)
    M_t_out = 0.5*(T1 + T2)
    u_vec = (M_t_out - M_b_out)
    u_hat = u_vec.normalized()

    # Outward normal = t̂ × û (ensures outward orientation)
    n_hat = t_hat.cross(u_hat).normalized()
    # Verify orientation (towards outward radial ≈ (cos a_mid, sin a_mid, 0))
    if n_hat.dot(cq.Vector(math.cos(a_mid), math.sin(a_mid), 0.0)) < 0:
        n_hat = (-n_hat)

    # Point at level z (linear interpolation)
    lam = z / H
    p_out = M_b_out + lam * (M_t_out - M_b_out)

    # Same for inner side
    M_b_in = 0.5*(b1 + b2)
    M_t_in = 0.5*(t1 + t2)
    p_in = M_b_in + lam * (M_t_in - M_b_in)

    # projected thickness along n̂ (faces ~ parallel)
    t_eff_mm = abs((p_out - p_in).dot(n_hat))

    return dict(a_mid=a_mid, p_out=p_out, p_in=p_in, n_hat=n_hat, t_hat=t_hat, u_hat=u_hat, t_eff_mm=t_eff_mm)

# ---------------------------------------------------------------------------
# V6b: perpendicular drillings to side facets (axis = n_hat)
# ---------------------------------------------------------------------------
def drill_side_holes_perpendicular(
    solid_in: cq.Workplane,
    geom: dict,
    scale_top: float,
    hole_d_mm: float,
    hole_clear_mm: float,
    t_offsets_mm = (0.0,),   # e.g. (0,) or (-delta, +delta) for BIG
    z: float = None,
) -> cq.Workplane:
    H = float(geom["H_mm"])
    if z is None:
        z = float(geom.get("z_center", H/2.0))
    eff_d = float(hole_d_mm) + max(0.0, float(hole_clear_mm))
    r_cyl = eff_d / 2.0

    eps = 0.05  # mm
    solid = solid_in
    N = int(geom["N"])

    for k in range(N):
        frame = _side_face_frame(N, k, geom, scale_top, z)
        p_out = frame["p_out"]
        p_in  = frame["p_in"]
        n_hat = frame["n_hat"]
        t_hat = frame["t_hat"]
        L = frame["t_eff_mm"] + 2.0*eps  # traverse full thickness

        for t_off in t_offsets_mm:
            # base slightly outside the face, offset along the edge
            p0 = p_out + cq.Vector(t_off*t_hat.x, t_off*t_hat.y, t_off*t_hat.z) + eps * n_hat
            dir_vec = cq.Vector(-n_hat.x, -n_hat.y, -n_hat.z)  # inward

            cyl = cq.Solid.makeCylinder(r_cyl, L, p0, dir_vec)
            solid = solid.cut(cyl)
            print(f"Coordinates of a perpendicular hole (x, y, z): ({p0.x:.3f}, {p0.y:.3f}, {p0.z:.3f})")

    return solid.clean()

# ---------------------------------------------------------------------------
# V6b: interior hex pocket ONLY on INNER SIDE, oriented according to same normal n̂
# ---------------------------------------------------------------------------
def add_interior_hex_pockets_perp_to_outer(
    solid_in: cq.Workplane,
    geom: dict,
    scale_top: float,
    nut_af_mm: float = 7.0,
    nut_clear_mm: float = 0.5,
    nut_depth_mm: float = 1.8,
    t_offsets_mm = (0.0,),
    z: float = None,
) -> cq.Workplane:
    H = float(geom["H_mm"])
    if z is None:
        z = float(geom.get("z_center", H/2.0))
    # across-flats -> circumscribed diameter (like rest of V6)
    D_hex = (float(nut_af_mm) + max(0.0, float(nut_clear_mm))) / math.cos(math.pi/6.0)

    eps = 0.05
    solid = solid_in
    N = int(geom["N"])

    for k in range(N):
        frame = _side_face_frame(N, k, geom, scale_top, z)
        p_in  = frame["p_in"]
        n_hat = frame["n_hat"]
        t_hat = frame["t_hat"]
        u_hat = frame["u_hat"]
        t_eff = frame["t_eff_mm"]

        # safety: pocket depth <= thickness
        if 2*nut_depth_mm > t_eff:
            print(f"[Warning] [k={k}] 2*nut_depth ({2*nut_depth_mm:.2f} mm) > t_eff ({t_eff:.2f} mm): interior pockets may overlap.")

        for t_off in t_offsets_mm:
            origin = p_in + cq.Vector(t_off*t_hat.x, t_off*t_hat.y, t_off*t_hat.z) - eps * n_hat  # slightly "inside" cavity
            # local plane: normal = n̂ (perpendicular to face), xDir = û (stable)
            plane_in = cq.Plane(origin=origin, xDir=cq.Vector(u_hat.x, u_hat.y, u_hat.z), normal=cq.Vector(n_hat.x, n_hat.y, n_hat.z))
            wp_in = cq.Workplane(plane_in)
            # milling from inside to outside (+n̂ direction)
            hex_in = wp_in.polygon(6, D_hex).extrude(float(nut_depth_mm))
            solid = solid.cut(hex_in)

    return solid.clean()


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Hollow polygonal crown with radial M4 holes, hex pockets and (optional) vertical holes at vertices."
    )
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--A", type=float, required=True, help="in cm (see --size_mode)")
    p.add_argument("--w", type=float, required=True, help="in cm (radial width)")
    p.add_argument("--H", type=float, required=True, help="in cm (Z thickness)")
    p.add_argument("--out", type=str, default="polygon_radial_hole.stl")
    p.add_argument("--size_mode", choices=["across_flats", "circum_diameter", "side_length"], default="across_flats")

    # Radial M4 holes: nominal diameter + clearance
    p.add_argument("--hole_d", type=float, default=4.5, help="Nominal diameter in mm (M4: 4.3 tight, 4.5 nominal)")
    p.add_argument("--hole_clear", type=float, default=0.5, help="Extra clearance added to diameter (mm) for FDM printing (e.g. 0.3–0.7)")
    p.add_argument("--z_offset", type=float, default=0.0, help="mm, + upwards (center of radial holes)")
    p.add_argument("--rotate", type=float, default=0.0, help="° around Z (global polygon rotation)")

    # Radial hex nut pockets: dimensions and clearance
    p.add_argument("--nut_af", type=float, default=7.0, help="Distance across flats of nut (mm). M4 ISO4032 = 7.0")
    p.add_argument("--nut_clear", type=float, default=0.5, help="Clearance on nut flats (mm), e.g. 0.3–0.7")
    p.add_argument("--nut_depth", type=float, default=1.8, help="Machining depth per side (mm) for radial hex pockets and vertices")

    # Step 3b: custom depth D (3rd file)
    p.add_argument("--D", type=float, default=None, help="Hex milling depth (mm) to generate a 3rd version (radial)")

    # Step 4: vertical holes at vertices + hex pockets in Z
    p.add_argument("--vertex_holes", action="store_true", help="Add a vertical hole + hex pockets above/below each vertex.")
    # retro-compat alias if you already had --top_holes:
    p.add_argument("--top_holes", dest="vertex_holes", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--vertex_hole_d", type=float, default=None, help="Nominal diameter (mm) of vertex holes (default = hole_d).")
    p.add_argument("--vertex_hole_clear", type=float, default=None, help="Clearance (mm) of vertex holes (default = hole_clear).")
    p.add_argument("--vertex_depth", type=float, default=None, help="Depth (mm) if non-through. Otherwise through.")

    # >>> NEW: 5th BIG OBJECT file <<<
    p.add_argument("--B", dest="big_object", action="store_true",
                   help="Enable 'Big Object' mode: 2 radial holes per side (at 1/3 and 2/3) + (if --vertex_holes) vertical holes at mid-sides. Generates a 5th file.")
    p.add_argument("--Alignment", type=float, default=None,
                   help="If specified, forces tangential offset of BigObject holes to align with an object of size A (cm).")

    # >>> V6
    p.add_argument("--assembly_angle", type=float, default=None,
                   help="Apply a global assembly angle (e.g. 120 => tilted faces). If absent, classic mode.")
    # --- V6 options (side pocket perpendicular) ---
    p.add_argument("--v6_side_pocket", action="store_true",
                   help="Add interior aligned hex pocket (6th file). Default: disabled.")
    p.add_argument("--v6_side_pocket_depth", type=float, default=None,
                   help="Hex pocket depth (mm) for 6th file (default = --nut_depth).")

    # specific for radial holes
    p.add_argument("--R", dest="radial_holes", action="store_true", default=True,
                   help="Enable radial holes (default: enabled).")
    p.add_argument("--no-R", dest="radial_holes", action="store_false",
                   help="Disable radial holes.")
    args = p.parse_args()

    # Step 1: base model (hollow prism + radial holes)
    part_base, geom_params = build_model(
        N=args.N,
        A_cm=args.A,
        w_cm=args.w,
        H_cm=args.H,
        size_mode=args.size_mode,
        hole_d_mm=args.hole_d if args.radial_holes else 0.0,
        hole_clear_mm=args.hole_clear if args.radial_holes else 0.0,
        z_offset_mm=args.z_offset,
        rotate_deg=args.rotate,
    )

    # Step 2: export base STL
    cq.exporters.export(part_base, args.out)
    print(f"[OK] Base version generated: {args.out}")

    # Step 3: radial hex pockets (parametric) + export
    part_upgraded = add_m4_hex_countersinks(
        part_base,
        geom_params,
        nut_af_mm=args.nut_af,
        nut_clear_mm=args.nut_clear,
        nut_depth_mm=args.nut_depth
    )
    base_name, extension = os.path.splitext(args.out)
    out_upgraded_name = f"{base_name}_geminiUpgrade{extension}"
    cq.exporters.export(part_upgraded, out_upgraded_name)
    print(f"[OK] Upgraded version (radial) generated: {out_upgraded_name}")

    # Step 3b (optional): radial hex pockets with depth D + export (3rd file)
    if args.D is not None:
        if args.D <= 0:
            raise ValueError("--D must be > 0 mm")
        part_upgraded_D = add_m4_hex_countersinks_with_depth(
            part_base,
            geom_params,
            args.D,
            nut_af_mm=args.nut_af,
            nut_clear_mm=args.nut_clear
        )
        D_tag = f"{args.D:.2f}".rstrip('0').rstrip('.')  # format 4.00 -> 4
        out_upgraded_D_name = f"{base_name}_geminiUpgrade_D{D_tag}mm{extension}"
        cq.exporters.export(part_upgraded_D, out_upgraded_D_name)
        print(f"[OK] Upgraded version (radial, D={D_tag} mm) generated: {out_upgraded_D_name}")

    # Step 4 (optional): vertical holes at vertices + top/bottom hex pockets (4th file)
    if args.vertex_holes:
        v_hole_d = args.vertex_hole_d if args.vertex_hole_d is not None else args.hole_d
        v_hole_clear = args.vertex_hole_clear if args.vertex_hole_clear is not None else args.hole_clear
        through = True
        v_depth = None
        if args.vertex_depth is not None and args.vertex_depth > 0:
            through = False
            v_depth = float(args.vertex_depth)
        part_vertex = add_vertex_vertical_holes_with_hex(
            part_upgraded,
            geom_params,
            hole_d_mm=v_hole_d,
            hole_clear_mm=v_hole_clear,
            through=through,
            depth_mm=v_depth,
            nut_af_mm=args.nut_af,
            nut_clear_mm=args.nut_clear,
            nut_depth_mm=args.nut_depth,
        )
        out_vertex_name = f"{base_name}_vertexHoles{extension}"
        cq.exporters.export(part_vertex, out_vertex_name)
        print(f"[OK] 4th file (vertices) generated: {out_vertex_name}")

    # =========================
    # 5th FILE: BIG OBJECT
    # =========================
    if args.big_object:
        # 5a) BIG base: 2 radial holes per side
        delta_override = None
        if args.Alignment is not None:
            # compute edge length of reference
            s_edge_ref = 2.0 * (regular_polygon_circ_diameter_from_A(args.N, args.Alignment * 10.0,
                                                                     args.size_mode) / 2.0 * math.cos(
                math.pi / args.N)) * math.tan(math.pi / args.N)
            delta_override = s_edge_ref / 6.0

        # 5a bis: small hack on hole placement
        big_base, geom_big = build_model_big(
            N=args.N,
            A_cm=args.A,
            w_cm=args.w,
            H_cm=args.H,
            size_mode=args.size_mode,
            hole_d_mm=args.hole_d if args.radial_holes else 0.0,
            hole_clear_mm=args.hole_clear if args.radial_holes else 0.0,
            z_offset_mm=args.z_offset,
            rotate_deg=args.rotate,
            delta_override=delta_override,
        )
        # 5b) BIG radial hex pockets (duplicated)
        big_upgraded = add_m_hex_countersinks_radial_big(
            big_base,
            geom_big,
            nut_af_mm=args.nut_af,
            nut_clear_mm=args.nut_clear,
            nut_depth_mm=args.nut_depth
        )

        # 5c) (Optional) vertical holes:
        #     - at vertices (ref "circum")
        #     - AND at mid-edges (ref "apoth")  <-- requested FIX
        out_big_name = f"{base_name}_bigObject{extension}"
        if args.vertex_holes:
            v_hole_d = args.vertex_hole_d if args.vertex_hole_d is not None else args.hole_d
            v_hole_clear = args.vertex_hole_clear if args.vertex_hole_clear is not None else args.hole_clear
            through = True
            v_depth = None
            if args.vertex_depth is not None and args.vertex_depth > 0:
                through = False
                v_depth = float(args.vertex_depth)

            # First, vertex holes
            big_with_vertices = add_vertical_holes_with_hex_at_angles(
                big_upgraded,
                geom_big,
                geom_big["angles_vertices"],
                hole_d_mm=v_hole_d,
                hole_clear_mm=v_hole_clear,
                through=through,
                depth_mm=v_depth,
                nut_af_mm=args.nut_af,
                nut_clear_mm=args.nut_clear,
                nut_depth_mm=args.nut_depth,
                center_ref="circum",
            )
            # Then, vertical holes at mid-edges
            big_final = add_vertical_holes_with_hex_at_angles(
                big_with_vertices,
                geom_big,
                geom_big["angles_mid_edges"],
                hole_d_mm=v_hole_d,
                hole_clear_mm=v_hole_clear,
                through=through,
                depth_mm=v_depth,
                nut_af_mm=args.nut_af,
                nut_clear_mm=args.nut_clear,
                nut_depth_mm=args.nut_depth,
                center_ref="apoth",
            )
            out_big_name = f"{base_name}_bigObject_vertexHoles{extension}"
        else:
            big_final = big_upgraded

        cq.exporters.export(big_final, out_big_name)
        print(f"[OK] 5th file (BIG OBJECT) generated: {out_big_name}")
        # --- V6: MACHINING LOGIC (intersection with tapered envelope) ---
if args.assembly_angle is not None:
    try:
        # a) Rebuild a clean shell (object 1 without holes)
        clean_shell, g_env = build_shell_only(
            N=args.N, A_cm=args.A, w_cm=args.w, H_cm=args.H,
            size_mode=args.size_mode, z_offset_mm=args.z_offset, rotate_deg=args.rotate
        )

        # b) Compute scale_top from assembly angle
        scale_top = compute_scale_top_from_assembly_angle(g_env, args.assembly_angle)

        # c) Build tapered envelope
        envelope = build_tapered_prism_from_geom(g_env, scale_top)

        # d) Machining: intersect -> object 6 "machined" (clean, no old radial holes)
        try:
            machined = clean_shell.intersect(envelope)
        except Exception as inter_err:
            print(f"[ERROR] Intersection failed: {inter_err}")
            # Export debug solids
            cq.exporters.export(clean_shell, "debug_shell.stl")
            cq.exporters.export(envelope, "debug_envelope.stl")
            raise

        # e) Edge offsets (BIG object ±delta) else (0,)
        t_offs = (0.0,)
        if args.big_object:
            s_edge = g_env["s_edge_out_mm"]
            delta = s_edge / 6.0
            t_offs = (-delta, +delta)

        # f) Side holes perpendicular to facets (+ optional internal hex pockets)
        machined = drill_side_holes_perpendicular(
            machined, g_env, scale_top,
            hole_d_mm=args.hole_d, hole_clear_mm=args.hole_clear,
            t_offsets_mm=t_offs, z=g_env.get("z_center", g_env["H_mm"] / 2.0),
        )

        if args.v6_side_pocket:
            depth = args.v6_side_pocket_depth if args.v6_side_pocket_depth is not None else args.nut_depth
            machined = add_interior_hex_pockets_perp_to_outer(
                machined, g_env, scale_top,
                nut_af_mm=args.nut_af, nut_clear_mm=args.nut_clear, nut_depth_mm=depth,
                t_offsets_mm=t_offs, z=g_env.get("z_center", g_env["H_mm"] / 2.0),
            )

        # g) (Optional) Vertical holes after machining
        if args.vertex_holes:
            v_hole_d = args.vertex_hole_d if args.vertex_hole_d is not None else args.hole_d
            v_hole_clear = args.vertex_hole_clear if args.vertex_hole_clear is not None else args.hole_clear
            through = True
            v_depth = None
            if args.vertex_depth is not None and args.vertex_depth > 0:
                through = False
                v_depth = float(args.vertex_depth)

            # Vertex holes:
            machined = add_vertical_holes_with_hex_at_angles(
                machined, g_env, g_env["angles_vertices"],
                hole_d_mm=v_hole_d, hole_clear_mm=v_hole_clear,
                through=through, depth_mm=v_depth,
                nut_af_mm=args.nut_af, nut_clear_mm=args.nut_clear, nut_depth_mm=args.nut_depth,
                center_ref="circum",
            )
            # (and if --B) mid-edge centers:
            if args.big_object:
                machined = add_vertical_holes_with_hex_at_angles(
                    machined, g_env, g_env["angles_mid_edges"],
                    hole_d_mm=v_hole_d, hole_clear_mm=v_hole_clear,
                    through=through, depth_mm=v_depth,
                    nut_af_mm=args.nut_af, nut_clear_mm=args.nut_clear, nut_depth_mm=args.nut_depth,
                    center_ref="apoth",
                )

        # h) Export final machined solid
        out_tapered = f"{base_name}_v6_machined_{int(args.assembly_angle)}deg{extension}"
        cq.exporters.export(machined, out_tapered)
        print(f"[OK] V6 file generated (machined CLEAN + perp holes, angle={args.assembly_angle}°): {out_tapered}")

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate V6 (machined clean): {e}")
        traceback.print_exc()
