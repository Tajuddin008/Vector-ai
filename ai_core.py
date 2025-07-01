import os
import torch
import numpy as np
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.lines as mlines
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import ezdxf
from ezdxf import const
from skimage import measure
from skimage.draw import polygon, line, rectangle, disk, polygon_perimeter, line_aa
from scipy.ndimage import binary_dilation, binary_opening, binary_closing
from skimage.morphology import disk as create_disk
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiLineString, box
from shapely.ops import unary_union
from PIL import Image, ImageDraw
from enum import Enum
import math
import copy
from collections import deque
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random

# ==================== CONFIGURATION ====================
CONFIG = {
    "weights_path": r"C:\Users\shaikot\sam_weights\sam_vit_h_4b8939.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_type": "vit_h",
    "output_dir": "outputs",
    "icon_dir": "icons",
    "max_dimension": 1400,
    "points_per_side": 48,
    "pred_iou_thresh": 0.95,
    "stability_score_thresh": 0.96,
    "min_mask_region_area": 500,
    "min_room_area": 3500,
    "min_wall_area": 1000, # For DXF export based on template
    "CONTAINMENT_THRESHOLD": 0.85,
    "simplify_tolerance": 1.0,
    "IDEALIZATION_GRID_SPACING": 10,
    "GRID_HISTOGRAM_THRESHOLD": 7,
    "RDP_TOLERANCE": 2.0,
    "SNAP_DISTANCE_PIXELS": 15,
    "WALL_THICKNESS_PIXELS": 10,
    "HATCH_SPACING": 8,
    "HATCH_ANGLE": 45,
    "DOOR_LEAF_THICKNESS_PIXELS": 2,
    "DOOR_ARC_THICKNESS_PIXELS": 1,
    "BACKGROUND_COLOR": [255, 255, 255],
    "WALL_COLOR": [0, 0, 0],
    "WALL_HATCH_COLOR": [180, 180, 180],
    "dxf_scale": 0.025,
    "FURNITURE_THICKNESS_PIXELS": 2,
    "DOOR_COLOR": [255, 0, 0],
    "WINDOW_COLOR": [0, 0, 255],
    "FURNITURE_COLOR": [128, 0, 128],
    "MAX_UNDO_STEPS": 50,
    "CIRCLE_WALL_SEGMENTS": 32,
    "ARC_WALL_SEGMENTS": 20,
    "SELECTION_COLOR": "#0080FFFF",
    "CROSSING_SELECTION_EDGE_COLOR": "#FF0000",
    "CROSSING_SELECTION_FILL_COLOR": "#00FF00",
    "WINDOW_SELECTION_EDGE_COLOR": "#0000FF",
    "WINDOW_SELECTION_FILL_COLOR": "#0080FF",
    "PREVIEW_COLOR": "#FF00FFFF",
    "HOVER_COLOR": "#FF8C00", # Using a distinct orange for hover
    "SEGMENT_EDIT_SELECTION_COLOR": [255, 255, 0, 128],
    "HANDLE_SIZE_PIXELS": 8,
    "HANDLE_COLOR": "#FF8C00",
    "STRAIGHT_LINE_TOLERANCE": 0.998,
    "MIN_LINE_SEGMENT_LENGTH": 10,
    "HYBRID_ORTHO_ANGLE_TOLERANCE": 7.0,
    "MINIMALIST_MIN_ROOM_AREA": 12000,
    "TRACE_PLUS_STRAIGHT_LINE_TOLERANCE": 0.9995,
    "TRACE_PLUS_RDP_TOLERANCE": 1.5,
}

class DrawingMode(Enum):
    SELECT = "select"
    WALL = "wall"
    LINE = "line"
    DOOR = "door"
    WINDOW = "window"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ARC_WALL = "arc_wall"
    SPLIT_WALL = "split_wall"

class ViewMode(Enum):
    VECTOR = "Vector"
    SEGMENTATION = "Segmentation"
    ORIGINAL = "Original"

class SelectionBoxMode(Enum):
    WINDOW = "window"
    CROSSING = "crossing"

class DragMode(Enum):
    NONE = "none"
    MOVE = "move"
    STRETCH = "stretch"

class VectorizationMode(Enum):
    IDEALIZE = "Idealize (Orthogonal)"
    HYBRID = "Hybrid (Rectilinear + Curves)"
    ARCHITECTURAL_CLEAN = "Architectural Clean (Hybrid + Symmetry)"
    TRACE = "Trace (Smooth Curves & Lines)"
    VECTOR_TRACE_PLUS = "Vector Trace+ (AI-Assisted Vectorization)"
    SIMPLIFY = "Simplify (Organic)"
    RAW = "Raw (Pixel-Perfect)"
    TACTILE = "Tactile (High Detail with Texture)"
    MINIMALIST = "Minimalist Interior (Semantic Cleanup)"

# ==================== HELPER FUNCTIONS ====================
def load_model_secure():
    print("⚙️ Loading SAM model...")
    if not os.path.exists(CONFIG["weights_path"]): raise FileNotFoundError(f"Model weights not found at '{CONFIG['weights_path']}'")
    try:
        model = sam_model_registry[CONFIG["model_type"]]()
        state_dict = torch.load(CONFIG["weights_path"], map_location=CONFIG["device"], weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device=CONFIG["device"])
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def process_image(image):
    ratio = min(CONFIG["max_dimension"]/image.size[0], CONFIG["max_dimension"]/image.size[1])
    new_size = (int(image.size[0]*ratio), int(image.size[1]*ratio))
    return np.array(image.resize(new_size, Image.LANCZOS))

def process_contour(contour):
    poly = Polygon(contour).simplify(CONFIG["simplify_tolerance"], preserve_topology=True)
    if not poly.is_valid: poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon): return np.array(max(poly.geoms, key=lambda p: p.area).exterior.coords) if poly.geoms else np.array([])
    return np.array(poly.exterior.coords) if isinstance(poly, Polygon) else np.array([])

def filter_and_get_rooms(masks, min_area_override=None):
    if not masks: return []
    non_empty_masks = [m for m in masks if m['segmentation'].any()]
    if not non_empty_masks: return []
    sorted_masks = sorted(non_empty_masks, key=lambda m: m['area'], reverse=True)
    
    min_area = min_area_override if min_area_override is not None else CONFIG["min_room_area"]

    final_room_masks, claimed_area = [], np.zeros_like(sorted_masks[0]['segmentation'], dtype=bool)
    for m in sorted_masks:
        if m['area'] < min_area: continue
        if np.sum(m['segmentation'] & claimed_area) / m['area'] > CONFIG["CONTAINMENT_THRESHOLD"]: continue
        final_room_masks.append(m); claimed_area |= m['segmentation']
    return final_room_masks

def calculate_global_orthogonal_grid(final_room_masks, image_shape):
    all_y, all_x = [], []
    for mask_data in final_room_masks:
        for contour in measure.find_contours(mask_data['segmentation'], 0.5):
            c = process_contour(contour)
            if c.size > 0: all_y.extend(c[:, 0]); all_x.extend(c[:, 1])
    if not all_x or not all_y: return np.array([]), np.array([])
    spacing = CONFIG["IDEALIZATION_GRID_SPACING"]
    y_hist, y_bins = np.histogram(all_y, bins=np.arange(0, image_shape[0] + spacing, spacing))
    x_hist, x_bins = np.histogram(all_x, bins=np.arange(0, image_shape[1] + spacing, spacing))
    dominant_y = y_bins[:-1][y_hist > CONFIG["GRID_HISTOGRAM_THRESHOLD"]] + spacing / 2
    dominant_x = x_bins[:-1][x_hist > CONFIG["GRID_HISTOGRAM_THRESHOLD"]] + spacing / 2
    return dominant_y, dominant_x

def _draw_hatching(image, mask, color, spacing, angle_deg):
    hatch_img = Image.new("L", (image.shape[1], image.shape[0]), 0)
    draw = ImageDraw.Draw(hatch_img)
    rows, cols = np.where(mask)
    if rows.size == 0: return
    min_x, min_y, max_x, max_y = np.min(cols), np.min(rows), np.max(cols), np.max(rows)
    diag = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    angle_rad = math.radians(angle_deg)
    for i in range(int(-diag / spacing), int(diag / spacing) + 1):
        line_center_x = (min_x + max_x) / 2 + i * spacing * math.cos(angle_rad + math.pi/2)
        line_center_y = (min_y + max_y) / 2 + i * spacing * math.sin(angle_rad + math.pi/2)
        x1 = line_center_x - diag/2 * math.cos(angle_rad); y1 = line_center_y - diag/2 * math.sin(angle_rad)
        x2 = line_center_x + diag/2 * math.cos(angle_rad); y2 = line_center_y + diag/2 * math.sin(angle_rad)
        draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
    hatch_mask = np.array(hatch_img) > 0
    final_mask = hatch_mask & mask
    image[final_mask] = color

def _draw_tactile_hatching(image, mask, color, spacing, angle_deg):
    _draw_hatching(image, mask, color, spacing, angle_deg)
    _draw_hatching(image, mask, color, spacing, angle_deg + 90)

def draw_window_symbol(image, p1, p2, wall_thickness):
    v = np.array(p2) - np.array(p1)
    if np.linalg.norm(v) < 1: return
    v_perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    frame_offset = v_perp * (wall_thickness / 2)
    p1_f1, p2_f1 = p1 - frame_offset, p2 - frame_offset
    p1_f2, p2_f2 = p1 + frame_offset, p2 + frame_offset
    rr, cc = line(int(p1_f1[0]), int(p1_f1[1]), int(p2_f1[0]), int(p2_f1[1])); image[np.clip(rr,0,image.shape[0]-1), np.clip(cc,0,image.shape[1]-1)] = CONFIG["WALL_COLOR"]
    rr, cc = line(int(p1_f2[0]), int(p1_f2[1]), int(p2_f2[0]), int(p2_f2[1])); image[np.clip(rr,0,image.shape[0]-1), np.clip(cc,0,image.shape[1]-1)] = CONFIG["WALL_COLOR"]
    rr, cc = line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])); image[np.clip(rr,0,image.shape[0]-1), np.clip(cc,0,image.shape[1]-1)] = CONFIG["WINDOW_COLOR"]

def draw_door_symbol(image, p1_center, p2_center, wall_thickness):
    height, width, _ = image.shape
    hinge_center, latch_center = np.array(p1_center), np.array(p2_center)
    v_wall = latch_center - hinge_center
    door_width = np.linalg.norm(v_wall)
    if door_width < 1: return
    v_wall_norm = v_wall / door_width
    v_perp = np.array([-v_wall_norm[1], v_wall_norm[0]])
    center_of_image = np.array([height / 2, width / 2])
    if np.dot(v_perp, center_of_image - hinge_center) > 0: v_perp *= -1
    half_thick = v_perp * (wall_thickness / 2)
    rr, cc = line(int((hinge_center-half_thick)[0]), int((hinge_center-half_thick)[1]), int((hinge_center+half_thick)[0]), int((hinge_center+half_thick)[1])); image[np.clip(rr,0,height-1), np.clip(cc,0,width-1)] = CONFIG["WALL_COLOR"]
    rr, cc = line(int((latch_center-half_thick)[0]), int((latch_center-half_thick)[1]), int((latch_center+half_thick)[0]), int((latch_center+half_thick)[1])); image[np.clip(rr,0,height-1), np.clip(cc,0,width-1)] = CONFIG["WALL_COLOR"]
    hinge_pivot = hinge_center + half_thick
    latch_closed = latch_center + half_thick
    v_closed_leaf = latch_closed - hinge_pivot
    v_open_leaf = np.array([-v_closed_leaf[1], v_closed_leaf[0]])
    open_latch = hinge_pivot + v_open_leaf
    rr_leaf, cc_leaf = line(int(hinge_pivot[0]), int(hinge_pivot[1]), int(open_latch[0]), int(open_latch[1]))
    leaf_mask = np.zeros(image.shape[:2], dtype=bool); leaf_mask[np.clip(rr_leaf,0,height-1), np.clip(cc_leaf,0,width-1)] = True
    dilated_leaf = binary_dilation(leaf_mask, create_disk(CONFIG["DOOR_LEAF_THICKNESS_PIXELS"] / 2)); image[dilated_leaf] = CONFIG["DOOR_COLOR"]
    start_angle = math.atan2((latch_closed - hinge_pivot)[0], (latch_closed - hinge_pivot)[1])
    end_angle = math.atan2((open_latch - hinge_pivot)[0], (open_latch - hinge_pivot)[1])
    if abs(end_angle - start_angle) > math.pi: end_angle += 2 * math.pi * np.sign(start_angle - end_angle)
    last_pt = latch_closed
    for angle in np.linspace(start_angle, end_angle, 30):
        arc_pt = hinge_pivot + np.array([door_width * math.sin(angle), door_width * math.cos(angle)])
        rr, cc = line(int(last_pt[0]), int(last_pt[1]), int(arc_pt[0]), int(arc_pt[1])); image[np.clip(rr,0,height-1), np.clip(cc,0,width-1)] = CONFIG["DOOR_COLOR"]
        last_pt = arc_pt

def generate_output_image(finalized_polygons, exterior_polygon, walls, lines, doors, windows, furniture, image_shape, vectorization_mode):
    output_image = np.full((*image_shape[:2], 3), CONFIG["BACKGROUND_COLOR"], dtype=np.uint8)
    wall_mask = np.zeros(image_shape[:2], dtype=bool)
    all_polys_to_process = finalized_polygons + ([exterior_polygon] if exterior_polygon is not None else [])
    for poly_coords in all_polys_to_process:
        if poly_coords is not None and len(poly_coords) > 1:
            for i in range(len(poly_coords) - 1):
                p1, p2 = poly_coords[i], poly_coords[i+1]
                if not np.allclose(p1, p2):
                    line_seg = LineString([p1, p2]); segment_poly_shape = line_seg.buffer(CONFIG["WALL_THICKNESS_PIXELS"] / 2, cap_style=3)
                    if not segment_poly_shape.is_empty:
                        rr, cc = polygon(np.array(segment_poly_shape.exterior.coords)[:, 0], np.array(segment_poly_shape.exterior.coords)[:, 1], wall_mask.shape)
                        wall_mask[np.clip(rr,0,image_shape[0]-1), np.clip(cc,0,image_shape[1]-1)] = True
    for wall in walls:
        if not wall or not wall.get('points') or len(wall['points']) != 2: continue
        p1_coords, p2_coords = wall['points']; thickness = wall.get('thickness', CONFIG["WALL_THICKNESS_PIXELS"])
        line_seg = LineString([p1_coords, p2_coords]); segment_poly_shape = line_seg.buffer(thickness / 2, cap_style=3)
        if not segment_poly_shape.is_empty:
            rr, cc = polygon(np.array(segment_poly_shape.exterior.coords)[:, 0], np.array(segment_poly_shape.exterior.coords)[:, 1], wall_mask.shape)
            wall_mask[np.clip(rr,0,image_shape[0]-1), np.clip(cc,0,image_shape[1]-1)] = True
            
    cut_walls = [w for w in walls if w and w.get('thickness')] if walls else [{'thickness': CONFIG['WALL_THICKNESS_PIXELS']}]
    for p1, p2 in windows + doors:
        if not p1 or not p2: continue
        cut_thickness = max(w.get('thickness', CONFIG["WALL_THICKNESS_PIXELS"]) for w in cut_walls) if cut_walls else CONFIG["WALL_THICKNESS_PIXELS"]
        v = np.array(p2) - np.array(p1); length = np.linalg.norm(v)
        if length < 1: continue
        v_perp = np.array([-v[1], v[0]]) / length; offset = v_perp * (cut_thickness / 2 + 2)
        cut_poly_coords = np.array([p1 - offset, p2 - offset, p2 + offset, p1 + offset])
        rr, cc = polygon(cut_poly_coords[:, 0], cut_poly_coords[:, 1], wall_mask.shape); wall_mask[np.clip(rr,0,image_shape[0]-1), np.clip(cc,0,image_shape[1]-1)] = False
    
    if vectorization_mode == VectorizationMode.TACTILE:
        _draw_tactile_hatching(output_image, wall_mask, CONFIG["WALL_HATCH_COLOR"], CONFIG["HATCH_SPACING"], CONFIG["HATCH_ANGLE"])
    else:
        _draw_hatching(output_image, wall_mask, CONFIG["WALL_HATCH_COLOR"], CONFIG["HATCH_SPACING"], CONFIG["HATCH_ANGLE"])

    outline = binary_dilation(wall_mask) & ~wall_mask; output_image[outline] = CONFIG["WALL_COLOR"]
    for p1, p2 in lines: 
        if p1 and p2: rr, cc = line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])); output_image[np.clip(rr,0,image_shape[0]-1), np.clip(cc,0,image_shape[1]-1)] = CONFIG["WALL_COLOR"]
    for p1, p2 in windows: 
        if p1 and p2: draw_window_symbol(output_image, p1, p2, CONFIG["WALL_THICKNESS_PIXELS"])
    for p1, p2 in doors: 
        if p1 and p2: draw_door_symbol(output_image, p1, p2, CONFIG["WALL_THICKNESS_PIXELS"])
    return output_image

def generate_segmentation_image(masks, image_shape):
    if not masks: return np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    if not any('color' in m for m in masks):
        for i, m in enumerate(masks): m['color'] = [random.randint(50, 200) for _ in range(3)]
    seg_img = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=False)
    for mask in sorted_masks: seg_img[mask['segmentation']] = mask['color']
    return seg_img

# ==================== NEW/IMPROVED EXPORT HELPER FUNCTIONS ====================
def clean_mask(mask):
    """Cleans up a binary mask using morphological operations."""
    # Remove small noise with an opening operation
    opened_mask = binary_opening(mask, structure=create_disk(2))
    # Fill small holes with a closing operation
    closed_mask = binary_closing(opened_mask, structure=create_disk(2))
    return closed_mask

def simplify_contour(contour):
    """Simplifies a contour using the Ramer-Douglas-Peucker algorithm."""
    return measure.approximate_polygon(contour, tolerance=CONFIG["RDP_TOLERANCE"])

def _transform_coords_for_dxf(p, image_height, scale):
    """Transforms image coordinates (y-down) to DXF coordinates (y-up) and applies scale."""
    return (p[1] * scale, (image_height - p[0]) * scale)

def _add_door_to_dxf(msp, p1, p2, wall_thickness, image_height, scale):
    """Adds a standard door symbol (arc and leaf) to the DXF modelspace."""
    hinge_center, latch_center = np.array(p1), np.array(p2)
    v_wall = latch_center - hinge_center
    door_width = np.linalg.norm(v_wall)
    if door_width < 1: return

    v_perp = np.array([-v_wall[1], v_wall[0]]) / door_width
    
    center_of_drawing = np.array([image_height / 2, (msp.doc.header['$EXTMAX'][0] / (2 * scale)) if msp.doc else 0])
    if np.dot(v_perp, center_of_drawing - hinge_center) > 0:
        v_perp *= -1

    transform = lambda pt: _transform_coords_for_dxf(pt, image_height, scale)

    hinge_pivot = hinge_center + v_perp * (wall_thickness / 2)
    v_closed = latch_center - hinge_pivot
    v_open = np.array([v_closed[1], -v_closed[0]])
    end_of_leaf = hinge_pivot + v_open

    start_angle = math.degrees(math.atan2(v_closed[0], v_closed[1]))
    end_angle = math.degrees(math.atan2(v_open[0], v_open[1]))
    
    msp.add_arc(
        center=transform(hinge_pivot),
        radius=np.linalg.norm(v_closed) * scale,
        start_angle=start_angle,
        end_angle=end_angle,
        dxfattribs={'layer': 'DOORS'}
    )

    msp.add_line(
        start=transform(hinge_pivot),
        end=transform(end_of_leaf),
        dxfattribs={'layer': 'DOORS'}
    )

def _add_window_to_dxf(msp, p1, p2, wall_thickness, image_height, scale):
    """Adds a standard window symbol (frame and glass) to the DXF modelspace."""
    v = np.array(p2) - np.array(p1)
    v_norm = np.linalg.norm(v)
    if v_norm < 1: return
    v_perp = np.array([-v[1], v[0]]) / v_norm
    frame_offset = v_perp * (wall_thickness / 2)
    
    transform = lambda pt: _transform_coords_for_dxf(pt, image_height, scale)

    p1_f1, p2_f1 = p1 - frame_offset, p2 - frame_offset
    p1_f2, p2_f2 = p1 + frame_offset, p2 + frame_offset

    msp.add_line(transform(p1_f1), transform(p2_f1), dxfattribs={'layer': 'WALLS'})
    msp.add_line(transform(p1_f2), transform(p2_f2), dxfattribs={'layer': 'WALLS'})
    msp.add_line(transform(p1), transform(p2), dxfattribs={'layer': 'WINDOWS'})

# ==================== SVG EXPORT HELPERS ====================
def draw_window_symbol_svg(svg_elements, p1, p2, wall_thickness):
    v = np.array(p2) - np.array(p1)
    if np.linalg.norm(v) < 1: return
    v_perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    frame_offset = v_perp * (wall_thickness / 2)
    p1_f1, p2_f1 = p1 - frame_offset, p2 - frame_offset
    p1_f2, p2_f2 = p1 + frame_offset, p2 + frame_offset
    svg_elements.append(f'  <line x1="{p1_f1[1]}" y1="{p1_f1[0]}" x2="{p2_f1[1]}" y2="{p2_f1[0]}" stroke="rgb({",".join(map(str, CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')
    svg_elements.append(f'  <line x1="{p1_f2[1]}" y1="{p1_f2[0]}" x2="{p2_f2[1]}" y2="{p2_f2[0]}" stroke="rgb({",".join(map(str, CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')
    svg_elements.append(f'  <line x1="{p1[1]}" y1="{p1[0]}" x2="{p2[1]}" y2="{p2[0]}" stroke="rgb({",".join(map(str, CONFIG["WINDOW_COLOR"]))})" stroke-width="2"/>')

def draw_door_symbol_svg(svg_elements, p1, p2, wall_thickness, H, W):
    hinge_center, latch_center = np.array(p1), np.array(p2)
    v_wall = latch_center - hinge_center
    door_width = np.linalg.norm(v_wall)
    if door_width < 1: return
    v_perp = np.array([-v_wall[1], v_wall[0]]) / door_width
    if np.dot(v_perp, np.array([H/2, W/2]) - hinge_center) > 0: v_perp *= -1
    
    hinge_pivot = hinge_center + v_perp * (wall_thickness / 2)
    latch_closed = latch_center + v_perp * (wall_thickness / 2)
    
    v_open_leaf = v_perp * door_width
    open_latch = hinge_pivot + v_open_leaf
    
    svg_elements.append(f'  <line x1="{hinge_pivot[1]}" y1="{hinge_pivot[0]}" x2="{open_latch[1]}" y2="{open_latch[0]}" stroke="rgb({",".join(map(str, CONFIG["DOOR_COLOR"]))})" stroke-width="{CONFIG["DOOR_LEAF_THICKNESS_PIXELS"]}"/>')
    svg_elements.append(f'  <path d="M {latch_closed[1]} {latch_closed[0]} A {door_width} {door_width} 0 0 0 {open_latch[1]} {open_latch[0]}" stroke="rgb({",".join(map(str, CONFIG["DOOR_COLOR"]))})" stroke-width="{CONFIG["DOOR_ARC_THICKNESS_PIXELS"]}" fill="none"/>')
    jamb1_p1, jamb1_p2 = hinge_center - v_perp * wall_thickness/2, hinge_center + v_perp * wall_thickness/2
    jamb2_p1, jamb2_p2 = latch_center - v_perp * wall_thickness/2, latch_center + v_perp * wall_thickness/2
    svg_elements.append(f'  <line x1="{jamb1_p1[1]}" y1="{jamb1_p1[0]}" x2="{jamb1_p2[1]}" y2="{jamb1_p2[0]}" stroke="rgb({",".join(map(str, CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')
    svg_elements.append(f'  <line x1="{jamb2_p1[1]}" y1="{jamb2_p1[0]}" x2="{jamb2_p2[1]}" y2="{jamb2_p2[0]}" stroke="rgb({",".join(map(str, CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')


    # Add this code to the end of your ai_core.py file

def _idealize_contour(contour, dominant_y_grid, dominant_x_grid):
    if len(contour) < 4 or dominant_y_grid.size < 2 or dominant_x_grid.size < 2: return contour
    idealized_pts = [[dominant_y_grid[np.argmin(np.abs(dominant_y_grid - pt[0]))], dominant_x_grid[np.argmin(np.abs(dominant_x_grid - pt[1]))]] for pt in contour]
    final_path = []
    if idealized_pts:
        final_path.append(np.array(idealized_pts[0]))
        for i in range(1, len(idealized_pts)):
            next_pt, prev_pt = np.array(idealized_pts[i]), final_path[-1]
            if np.allclose(next_pt, prev_pt): continue
            delta_y, delta_x = abs(next_pt[0] - prev_pt[0]), abs(next_pt[1] - prev_pt[1])
            if delta_y > 1e-3 and delta_x > 1e-3:
                if delta_x > delta_y: final_path.append(np.array([prev_pt[0], next_pt[1]]))
                else: final_path.append(np.array([next_pt[0], prev_pt[1]]))
            if not np.allclose(final_path[-1], next_pt): final_path.append(next_pt)
    unique_path = [final_path[0]] if final_path else []
    for i in range(1, len(final_path)):
        if np.linalg.norm(final_path[i] - unique_path[-1]) > 1.0: unique_path.append(final_path[i])
    return np.array(unique_path)

def _hybrid_simplify_contour(contour):
    if len(contour) < 3: return contour
    new_points = [contour[0]]
    angle_tol = CONFIG["HYBRID_ORTHO_ANGLE_TOLERANCE"]
    i = 0
    while i < len(contour) - 1:
        p1, p2 = new_points[-1], contour[i+1]
        dy, dx = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            i += 1; continue
        angle = math.degrees(math.atan2(dy, dx)) % 360
        if (abs(angle) < angle_tol) or (abs(angle - 180) < angle_tol) or (abs(angle - 360) < angle_tol):
            new_p2 = (p1[0], p2[1]) # Snap to horizontal
        elif (abs(angle - 90) < angle_tol) or (abs(angle - 270) < angle_tol):
            new_p2 = (p2[0], p1[1]) # Snap to vertical
        else:
            new_p2 = tuple(p2) # Keep original
        if tuple(new_p2) != tuple(p1):
             new_points.append(np.array(new_p2))
        i += 1
    return measure.approximate_polygon(np.array(new_points), tolerance=CONFIG["RDP_TOLERANCE"])

def get_final_architectural_contour(contour, vectorization_mode, grid_y=None, grid_x=None):
    if len(contour) < 4: return None
    simplified = process_contour(contour)
    if simplified.size < 3: return None
    
    mode = vectorization_mode
    if mode in [VectorizationMode.MINIMALIST, VectorizationMode.TACTILE]:
        base_map = {VectorizationMode.MINIMALIST: VectorizationMode.IDEALIZE, VectorizationMode.TACTILE: VectorizationMode.TRACE}
        mode = base_map.get(mode, VectorizationMode.TRACE)

    if mode == VectorizationMode.IDEALIZE:
        if grid_y is None or grid_x is None or grid_y.size < 2 or grid_x.size < 2:
             print("Warning: Idealization grid not available, falling back to Hybrid.")
             return _hybrid_simplify_contour(simplified)
        idealized = _idealize_contour(simplified, grid_y, grid_x)
        if idealized.size < 3: return None
        return measure.approximate_polygon(idealized, tolerance=CONFIG["RDP_TOLERANCE"])
    
    elif mode == VectorizationMode.HYBRID:
        return _hybrid_simplify_contour(simplified)
    
    # ... Add other modes like TRACE, SIMPLIFY, RAW if needed, following the pattern ...
    
    else: # Default fallback
        return measure.approximate_polygon(simplified, tolerance=CONFIG["RDP_TOLERANCE"])

def vectorize_floorplan(raw_masks, image_shape, vectorization_mode):
    """
    High-level function to perform the entire vectorization process.
    This is the single source of truth for both desktop and web apps.
    """
    # 1. Filter masks to find rooms
    min_area_override = CONFIG["MINIMALIST_MIN_ROOM_AREA"] if vectorization_mode == VectorizationMode.MINIMALIST else None
    final_room_masks = filter_and_get_rooms(raw_masks, min_area_override=min_area_override)
    
    # 2. Calculate grid if needed
    grid_y, grid_x = np.array([]), np.array([])
    if vectorization_mode in [VectorizationMode.IDEALIZE, VectorizationMode.HYBRID, VectorizationMode.ARCHITECTURAL_CLEAN, VectorizationMode.MINIMALIST]:
        grid_y, grid_x = calculate_global_orthogonal_grid(final_room_masks, image_shape)
        
    # 3. Process each room mask into a finalized polygon
    finalized_polygons = []
    for m in final_room_masks:
        contours = measure.find_contours(m['segmentation'], 0.5)
        if not contours: continue
        largest_contour = max(contours, key=lambda c: len(c))
        
        processed_contour = get_final_architectural_contour(largest_contour, vectorization_mode, grid_y, grid_x)
        
        if processed_contour is not None and len(processed_contour) > 2:
            # Ensure validity
            poly = Polygon(processed_contour)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                finalized_polygons.append(np.array(poly.exterior.coords))

    # 4. Calculate the exterior polygon from the union of all rooms
    exterior_polygon = None
    if finalized_polygons:
        shapely_polygons = [Polygon(p) for p in finalized_polygons if p is not None and len(p) > 2 and Polygon(p).is_valid]
        if shapely_polygons:
            try:
                unified_shape = unary_union([p.buffer(0) for p in shapely_polygons])
                if not unified_shape.is_empty:
                    if isinstance(unified_shape, Polygon):
                        exterior_contour = np.array(unified_shape.exterior.coords)
                    elif isinstance(unified_shape, MultiPolygon):
                        exterior_contour = np.array(max(unified_shape.geoms, key=lambda p: p.area).exterior.coords)
                    else:
                        exterior_contour = None
                    
                    if exterior_contour is not None:
                        exterior_polygon = get_final_architectural_contour(exterior_contour, vectorization_mode, grid_y, grid_x)
            except Exception as e:
                print(f"Warning: Could not unify polygons for exterior: {e}")

    return finalized_polygons, exterior_polygon