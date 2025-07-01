import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, PhotoImage, Menu, messagebox, simpledialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import matplotlib.lines as mlines
from segment_anything import SamPredictor
from PIL import Image, ImageTk
import copy
from collections import deque
import random

# Import all the core logic, Enums, and CONFIG from our new ai_core.py file
import ai_core


# This function uses tkinter, so it belongs in the UI file.
def select_input_file(root):
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select Floorplan Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    root.deiconify()
    return filepath


# ==================== SEGMENTATION EDITOR WINDOW (MODIFIED) ====================
class SegmentationEditor(tk.Toplevel):
    # This nested class must also use the ai_core definition
    class DrawingTool(ai_core.Enum):
        NONE = "none"
        POLYGON = "polygon"
        RECTANGLE = "rectangle"
        CIRCLE = "circle"
        ARC = "arc"

    def __init__(self, parent, raw_masks, image_np, on_close_callback):
        super().__init__(parent)
        self.title("Segmentation Editor")
        self.geometry("1200x900")
        self.transient(parent)
        self.grab_set()

        self.raw_masks = copy.deepcopy(raw_masks)
        self.image_np = image_np
        self.on_close_callback = on_close_callback

        self.selection = set()
        self.is_splitting = False
        self.split_points = []
        
        self.drawing_tool = self.DrawingTool.NONE
        self.draw_points = []
        self.is_dragging_shape = False
        self.preview_artist = None
        self.tool_buttons = {}
        self.ortho_var = tk.BooleanVar(value=False)
        self.hover_close_point = False
        self.last_mouse_pos = None
        
        self.snap_var = tk.BooleanVar(value=True)
        self.snap_cache_dirty = True
        self.snap_points_cache = None
        self.snapped_pos = None
        
        self.alignment_guide_info = None
        self.logical_cursor_pos = None
        
        self.hovered_mask_idx = None
        self.is_zoomed = False

        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.LEFT, anchor=tk.NW)

        ttk.Button(button_frame, text="Merge Selected", command=self.merge_selected).pack(side=tk.LEFT, padx=2)
        self.split_button = ttk.Button(button_frame, text="Split Segment", command=self.toggle_split_mode)
        self.split_button.pack(side=tk.LEFT, padx=2)

        draw_tools_frame = ttk.LabelFrame(button_frame, text="Draw")
        draw_tools_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.tool_buttons[self.DrawingTool.POLYGON] = ttk.Button(draw_tools_frame, text="Polygon", command=lambda: self._set_active_tool(self.DrawingTool.POLYGON))
        self.tool_buttons[self.DrawingTool.POLYGON].pack(side=tk.LEFT, padx=2, pady=2)
        self.tool_buttons[self.DrawingTool.ARC] = ttk.Button(draw_tools_frame, text="Arc Segment", command=lambda: self._set_active_tool(self.DrawingTool.ARC))
        self.tool_buttons[self.DrawingTool.ARC].pack(side=tk.LEFT, padx=2, pady=2)
        
        self.tool_buttons[self.DrawingTool.RECTANGLE] = ttk.Button(draw_tools_frame, text="Rectangle", command=lambda: self._set_active_tool(self.DrawingTool.RECTANGLE))
        self.tool_buttons[self.DrawingTool.RECTANGLE].pack(side=tk.LEFT, padx=2, pady=2)
        self.tool_buttons[self.DrawingTool.CIRCLE] = ttk.Button(draw_tools_frame, text="Circle", command=lambda: self._set_active_tool(self.DrawingTool.CIRCLE))
        self.tool_buttons[self.DrawingTool.CIRCLE].pack(side=tk.LEFT, padx=2, pady=2)
        
        ortho_button = ttk.Checkbutton(draw_tools_frame, text="ORTHO", variable=self.ortho_var)
        ortho_button.pack(side=tk.LEFT, padx=5)

        snap_button = ttk.Checkbutton(draw_tools_frame, text="SNAP", variable=self.snap_var, command=self._on_snap_toggle)
        snap_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=10)

        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(top_frame, text="Tip: Hold Shift to select multiple. Scroll to zoom. Double-click to reset view.", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Apply & Close", command=self.apply_and_close).pack(side=tk.RIGHT, padx=2)
        ttk.Button(top_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=2)
        
        self.fig = Figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().focus_set()
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._update_snap_cache()
        self.redraw()
        
    def on_scroll(self, event):
        if not event.inaxes: return
        scale_factor = 1.1
        if event.button == 'up': scale = 1.0 / scale_factor
        elif event.button == 'down': scale = scale_factor
        else: return
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_xlim = [xdata - (xdata - cur_xlim[0]) * scale, xdata + (cur_xlim[1] - xdata) * scale]
        new_ylim = [ydata - (ydata - cur_ylim[0]) * scale, ydata + (cur_ylim[1] - ydata) * scale]
        self.ax.set_xlim(new_xlim); self.ax.set_ylim(new_ylim)
        self.is_zoomed = True; self.canvas.draw_idle()

    def reset_view(self):
        self.is_zoomed = False; self.redraw()

    def on_key_press(self, event):
        if event.key == 'escape':
            if self.drawing_tool != self.DrawingTool.NONE:
                self._cancel_current_drawing(deactivate_tool=True)

    def redraw(self):
        if self.is_zoomed: xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        if self.preview_artist: self.preview_artist = None

        self.ax.imshow(self.image_np, alpha=0.5)
        seg_overlay = np.zeros((*self.image_np.shape[:2], 4), dtype=np.uint8)
        if self.raw_masks and not any('color' in m for m in self.raw_masks):
            for i, m in enumerate(self.raw_masks): m['color'] = [random.randint(50, 200) for _ in range(3)]
        for i, mask_data in enumerate(self.raw_masks):
            color = mask_data.get('color', [128, 128, 128])
            if i in self.selection: seg_overlay[mask_data['segmentation']] = ai_core.CONFIG['SEGMENT_EDIT_SELECTION_COLOR']
            else: seg_overlay[mask_data['segmentation']] = color + [180]
        self.ax.imshow(seg_overlay)

        if self.hovered_mask_idx is not None and self.hovered_mask_idx != -1 and self.hovered_mask_idx not in self.selection:
            hovered_mask = self.raw_masks[self.hovered_mask_idx]
            contours = ai_core.measure.find_contours(hovered_mask['segmentation'], 0.5)
            for contour in contours: self.ax.plot(contour[:, 1], contour[:, 0], '--', color=ai_core.CONFIG.get("HOVER_COLOR", "#FF8C00"), linewidth=2, zorder=15)
        
        if self.alignment_guide_info:
            p_start, p_end = self.alignment_guide_info
            self.ax.plot([p_start[1], p_end[1]], [p_start[0], p_end[0]], color='g', ls='--', lw=1.0, zorder=25)
            
        if self.snapped_pos:
            y, x = self.snapped_pos
            s = ai_core.CONFIG["HANDLE_SIZE_PIXELS"] * 1.5
            self.ax.add_patch(patches.Rectangle((x - s/2, y - s/2), s, s, fill=False, color='#00FFFF', lw=2, zorder=35))

        if self.drawing_tool == self.DrawingTool.POLYGON and len(self.draw_points) > 0:
            points_np = np.array(self.draw_points)
            self.ax.plot(points_np[:, 1], points_np[:, 0], color='#FF0000', marker='o', markersize=3, lw=1.5, zorder=23)
            if self.logical_cursor_pos:
                y_logic, x_logic = self.logical_cursor_pos
                p_cursor = [y_logic, x_logic]
                p_start, p_last = self.draw_points[0], self.draw_points[-1]
                self.ax.plot([p_start[1], p_cursor[1]], [p_start[0], p_cursor[0]], color='black', lw=1.5, ls='--', zorder=24)
                p_preview = self._apply_ortho_constraint(p_last, p_cursor)
                self.ax.plot([p_last[1], p_preview[1]], [p_last[0], p_preview[0]], color='r', ls='-', lw=1.5, zorder=24)
                self.ax.plot(p_cursor[1], p_cursor[0], 'ro', markersize=5, zorder=25)
            if len(self.draw_points) > 2: self.ax.plot([points_np[-1, 1], points_np[0, 1]], [points_np[-1, 0], points_np[0, 0]], color='#FF0000', ls='--', zorder=22)
            if self.hover_close_point: self.ax.plot(points_np[0, 1], points_np[0, 0], 'go', markersize=12, zorder=30, alpha=0.7)
        
        elif self.is_dragging_shape and len(self.draw_points) == 1 and self.logical_cursor_pos:
             y_logic, x_logic = self.logical_cursor_pos
             p_start = self.draw_points[0]
             self.ax.plot([p_start[1], x_logic], [p_start[0], y_logic], color='black', lw=2, ls='-', zorder=24)
             self.ax.plot(x_logic, y_logic, 'ro', markersize=5, zorder=25)
             if self.drawing_tool == self.DrawingTool.RECTANGLE:
                self.preview_artist = patches.Rectangle((min(p_start[1], x_logic), min(p_start[0], y_logic)), abs(x_logic-p_start[1]), abs(y_logic-p_start[0]), fill=False, color='r', ls='--', lw=1.5)
             elif self.drawing_tool == self.DrawingTool.CIRCLE:
                self.preview_artist = patches.Circle((p_start[1], p_start[0]), np.linalg.norm(np.array(p_start) - np.array([y_logic, x_logic])), fill=False, color='r', ls='--', lw=1.5)
             if self.preview_artist: self.ax.add_patch(self.preview_artist)
        
        elif self.drawing_tool == self.DrawingTool.ARC and self.logical_cursor_pos:
            if len(self.draw_points) == 1:
                p1 = self.draw_points[0]; p2 = self.logical_cursor_pos
                self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r--', lw=1.5)
            elif len(self.draw_points) == 2:
                p1, p2 = self.draw_points[0], self.draw_points[1]
                p_mid = self.logical_cursor_pos
                arc_points = self._calculate_arc_points(p1, p2, p_mid)
                if arc_points:
                    arc_np = np.array(arc_points)
                    self.ax.plot(arc_np[:, 1], arc_np[:, 0], 'r--', lw=1.5)

        if self.is_zoomed: self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.ax.axis('off'); self.canvas.draw_idle()
    
    def on_motion(self, event):
        if not event.inaxes or event.ydata is None or event.xdata is None:
            self.last_mouse_pos = None; self.alignment_guide_info = None; self.redraw()
            return
        
        self.last_mouse_pos = event
        y_cursor, x_cursor = event.ydata, event.xdata
        
        new_hover_idx = -1
        y_int, x_int = int(y_cursor), int(x_cursor)
        if 0 <= y_int < self.image_np.shape[0] and 0 <= x_int < self.image_np.shape[1]:
            sorted_indices = sorted(range(len(self.raw_masks)), key=lambda k: self.raw_masks[k]['area'])
            for i in sorted_indices:
                if self.raw_masks[i]['segmentation'][y_int, x_int]: new_hover_idx = i; break
        self.hovered_mask_idx = new_hover_idx

        y_logic, x_logic = y_cursor, x_cursor
        
        y_snap, x_snap = self._snap_to_targets(y_cursor, x_cursor)
        snapped_to_corner = (y_snap, x_snap) != (y_cursor, x_cursor)
        if snapped_to_corner: y_logic, x_logic = y_snap, x_snap; self.snapped_pos = (y_logic, x_logic)
        else: self.snapped_pos = None
        
        self.alignment_guide_info = None
        if self.drawing_tool == self.DrawingTool.POLYGON and len(self.draw_points) >= 1 and not snapped_to_corner:
            y_start, x_start = self.draw_points[0]
            if abs(x_cursor - x_start) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]:
                x_logic = x_start; self.alignment_guide_info = ((y_start, x_start), (y_cursor, x_logic))
            elif abs(y_cursor - y_start) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]:
                y_logic = y_start; self.alignment_guide_info = ((y_start, x_start), (y_logic, x_cursor))
        
        self.logical_cursor_pos = (y_logic, x_logic)
        
        if self.drawing_tool == self.DrawingTool.POLYGON:
            old_hover_state = self.hover_close_point
            self.hover_close_point = len(self.draw_points) >= 3 and np.linalg.norm(np.array(self.logical_cursor_pos) - np.array(self.draw_points[0])) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]
            if old_hover_state != self.hover_close_point: self.redraw()
        
        self.redraw()

    def on_press(self, event):
        if not event.inaxes or not event.xdata: return
        if event.button == 1 and event.dblclick: self.reset_view(); return

        if self.drawing_tool != self.DrawingTool.NONE:
            y, x = (int(p) for p in (self.logical_cursor_pos or (event.ydata, event.xdata)))
            if event.button == 1:
                if self.drawing_tool == self.DrawingTool.POLYGON:
                    if len(self.draw_points) >= 3 and np.linalg.norm(np.array([y, x]) - self.draw_points[0]) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]:
                        self._finalize_polygon(); return
                    point_to_add = self._apply_ortho_constraint(self.draw_points[-1], [y, x]) if self.ortho_var.get() and self.draw_points else [y, x]
                    self.draw_points.append(point_to_add)
                elif self.drawing_tool in [self.DrawingTool.RECTANGLE, self.DrawingTool.CIRCLE]:
                    if not self.is_dragging_shape: self.is_dragging_shape = True; self.draw_points = [[y, x]]
                elif self.drawing_tool == self.DrawingTool.ARC:
                    self.draw_points.append([y,x])
                    if len(self.draw_points) == 3: self._finalize_arc(); return
            elif event.button == 3:
                if self.drawing_tool == self.DrawingTool.POLYGON and self.draw_points: self._show_polygon_context_menu(event)
                elif self.is_dragging_shape: self._cancel_current_drawing()
            self.logical_cursor_pos = None; self.alignment_guide_info = None; self.redraw()
            return

        y, x = int(event.ydata), int(event.xdata)
        if self.is_splitting: self.split_points.append((x, y)); self.execute_split() if len(self.split_points) == 2 else None; return
        
        clicked_mask_idx = -1
        sorted_indices = sorted(range(len(self.raw_masks)), key=lambda k: self.raw_masks[k]['area'])
        for i in sorted_indices:
            if self.raw_masks[i]['segmentation'][y, x]: clicked_mask_idx = i; break
        
        if clicked_mask_idx != -1:
            if event.key == 'shift': self.selection.symmetric_difference_update({clicked_mask_idx})
            else: self.selection = {clicked_mask_idx}
        elif not event.key == 'shift': self.selection.clear()
        self.redraw()
        
    def on_release(self, event):
        if not event.inaxes or not event.xdata: return
        if self.is_dragging_shape and event.button == 1 and len(self.draw_points) == 1:
            y, x = (int(p) for p in (self.logical_cursor_pos or (event.ydata, event.xdata)))
            self.draw_points.append([y, x])
            if self.drawing_tool == self.DrawingTool.RECTANGLE: self._finalize_rectangle()
            elif self.drawing_tool == self.DrawingTool.CIRCLE: self._finalize_circle()

    def _on_snap_toggle(self):
        if not self.snap_var.get(): self.snapped_pos = None; self.redraw()
        else: self.snap_cache_dirty = True

    def _update_snap_cache(self):
        if not self.snap_var.get(): self.snap_points_cache = None; return
        all_corners = []
        for mask_data in self.raw_masks:
            for contour in ai_core.measure.find_contours(mask_data['segmentation'], 0.5):
                all_corners.extend(ai_core.measure.approximate_polygon(contour, tolerance=ai_core.CONFIG.get("RDP_TOLERANCE", 2.0))[:-1])
        self.snap_points_cache = np.array(all_corners) if all_corners else np.empty((0, 2))
        self.snap_cache_dirty = False
        print(f"Updated snap cache with {len(all_corners)} points.")

    def _snap_to_targets(self, y, x):
        if not self.snap_var.get(): return y, x
        if self.snap_cache_dirty or self.snap_points_cache is None: self._update_snap_cache()
        if self.snap_points_cache is None or self.snap_points_cache.shape[0] == 0: return y, x
        dists_sq = np.sum((self.snap_points_cache - [y, x])**2, axis=1)
        min_idx = np.argmin(dists_sq)
        return self.snap_points_cache[min_idx] if dists_sq[min_idx] < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]**2 else (y, x)
    
    def _set_active_tool(self, tool):
        self.drawing_tool = self.DrawingTool.NONE if self.drawing_tool == tool else tool
        if self.drawing_tool != self.DrawingTool.NONE: self.is_splitting = False; self.split_button.state(['!pressed'])
        self._cancel_current_drawing(deactivate_tool=False)
        for t, btn in self.tool_buttons.items(): btn.state(['pressed'] if t == self.drawing_tool else ['!pressed'])
        self.title(f"Segmentation Editor - Drawing {self.drawing_tool.value.capitalize()}" if self.drawing_tool != self.DrawingTool.NONE else "Segmentation Editor")
        self.canvas.get_tk_widget().focus_set()
        self.redraw()

    def _cancel_current_drawing(self, deactivate_tool=True):
        self.draw_points.clear(); self.is_dragging_shape = False; self.hover_close_point = False
        if self.preview_artist: self.preview_artist.remove(); self.preview_artist = None
        if deactivate_tool:
            self.drawing_tool = self.DrawingTool.NONE
            for btn in self.tool_buttons.values(): btn.state(['!pressed'])
            self.title("Segmentation Editor")
        self.redraw()

    def _apply_ortho_constraint(self, p_last, p_current):
        if not self.ortho_var.get() or not p_last: return p_current
        dx, dy = p_current[1] - p_last[1], p_current[0] - p_last[0]
        return [p_current[0], p_last[1]] if abs(dy) > abs(dx) else [p_last[0], p_current[1]]

    def _show_polygon_context_menu(self, event):
        menu = Menu(self, tearoff=0)
        menu.add_command(label="Finish Shape", command=self._finalize_polygon)
        if self.draw_points: menu.add_command(label="Remove Last Point", command=self._remove_last_poly_point)
        menu.add_separator()
        menu.add_command(label="Cancel Drawing", command=lambda: self._cancel_current_drawing(deactivate_tool=True))
        menu.post(event.x_root, event.y_root)

    def _remove_last_poly_point(self):
        if self.draw_points: self.draw_points.pop(); self.redraw()
        
    def _add_new_segment_from_mask(self, new_mask):
        new_area = np.sum(new_mask)
        if new_area < 50: return
        rows, cols = np.where(new_mask)
        bbox = [np.min(cols), np.min(rows), np.max(cols)-np.min(cols), np.max(rows)-np.min(rows)]
        self.raw_masks.append({'segmentation': new_mask, 'area': int(new_area), 'bbox': [int(c) for c in bbox], 'predicted_iou': 1.0, 'point_coords': [[0,0]], 'stability_score': 1.0, 'crop_box': [0, 0, self.image_np.shape[1], self.image_np.shape[0]], 'color': [random.randint(50, 200) for _ in range(3)]})
        self.snap_cache_dirty = True 

    def _finalize_polygon(self):
        if len(self.draw_points) < 3: messagebox.showwarning("Draw Polygon", "A polygon needs at least 3 points.", parent=self)
        else:
            new_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
            rr, cc = ai_core.polygon(np.array(self.draw_points)[:, 0], np.array(self.draw_points)[:, 1], new_mask.shape)
            new_mask[rr, cc] = True; self._add_new_segment_from_mask(new_mask)
        self._cancel_current_drawing(deactivate_tool=True)

    def _finalize_rectangle(self):
        if len(self.draw_points) != 2: return
        p1, p2 = self.draw_points
        new_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
        rr, cc = ai_core.rectangle((min(p1[0], p2[0]), min(p1[1], p2[1])), extent=(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])), shape=new_mask.shape)
        new_mask[rr, cc] = True; self._add_new_segment_from_mask(new_mask)
        self._cancel_current_drawing(deactivate_tool=True)

    def _finalize_circle(self):
        if len(self.draw_points) != 2: return
        center, edge = self.draw_points
        radius = np.linalg.norm(np.array(center) - np.array(edge))
        new_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
        rr, cc = ai_core.disk((center[0], center[1]), radius, shape=new_mask.shape)
        new_mask[rr, cc] = True; self._add_new_segment_from_mask(new_mask)
        self._cancel_current_drawing(deactivate_tool=True)
    
    def _calculate_arc_points(self, p1_start, p2_end, p3_mid):
        p1, p2, p3 = np.array(p1_start), np.array(p2_end), np.array(p3_mid)
        A = np.vstack((p1, p2, p3)); B = np.ones(3); C = np.sum(A**2, axis=1)
        a = np.linalg.det(np.column_stack((A, B)))
        if abs(a) < 1e-6: return [p1.tolist(), p2.tolist()]
        Dx = np.linalg.det(np.column_stack((C, A[:, 1], B)))
        Dy = -np.linalg.det(np.column_stack((C, A[:, 0], B)))
        center = np.array([Dx, Dy]) / (2 * a)
        radius = np.linalg.norm(p1 - center)
        start_angle = ai_core.math.atan2(p1[0]-center[0], p1[1]-center[1])
        mid_angle = ai_core.math.atan2(p3[0]-center[0], p3[1]-center[1])
        end_angle = ai_core.math.atan2(p2[0]-center[0], p2[1]-center[1])
        
        angles = np.array([start_angle, mid_angle, end_angle])
        sorted_indices = np.argsort(np.unwrap(angles))
        if not np.array_equal(sorted_indices, [0, 1, 2]) and not np.array_equal(sorted_indices, [2, 1, 0]):
            if abs(np.unwrap([start_angle, mid_angle])[1] - mid_angle) > abs(np.unwrap([start_angle, end_angle])[1] - end_angle):
                 start_angle, end_angle = end_angle, start_angle

        points = []
        num_segments = int(radius * abs(np.unwrap([start_angle, end_angle])[1] - start_angle) / 5) + 3
        for angle in np.linspace(start_angle, end_angle, num_segments):
            points.append([center[0] + radius*ai_core.math.sin(angle), center[1] + radius*ai_core.math.cos(angle)])
        return points

    def _finalize_arc(self):
        if len(self.draw_points) != 3: return
        p1, p2, p3 = self.draw_points
        arc_pts = self._calculate_arc_points(p1, p2, p3)
        if not arc_pts or len(arc_pts) < 2: self._cancel_current_drawing(True); return
        
        arc_line = ai_core.LineString([(p[1], p[0]) for p in arc_pts])
        arc_poly = arc_line.buffer(1.5, cap_style=2)
        
        new_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
        if arc_poly and not arc_poly.is_empty:
             rr, cc = ai_core.polygon(np.array(arc_poly.exterior.coords)[:, 1], np.array(arc_poly.exterior.coords)[:, 0], new_mask.shape)
             new_mask[rr, cc] = True; self._add_new_segment_from_mask(new_mask)
        self._cancel_current_drawing(deactivate_tool=True)

    def merge_selected(self):
        if len(self.selection) < 2: messagebox.showinfo("Merge", "Select at least two segments to merge.", parent=self); return
        selected_indices = sorted(list(self.selection), key=lambda i: self.raw_masks[i]['area'], reverse=True)
        target_idx = selected_indices[0]
        for i in selected_indices[1:]: self.raw_masks[target_idx]['segmentation'] |= self.raw_masks[i]['segmentation']
        self.raw_masks[target_idx]['area'] = np.sum(self.raw_masks[target_idx]['segmentation'])
        indices_to_remove = sorted(selected_indices[1:], reverse=True)
        for i in indices_to_remove: del self.raw_masks[i]
        new_selection_idx = target_idx
        for removed_idx in indices_to_remove:
            if new_selection_idx > removed_idx: new_selection_idx -= 1
        self.selection = {new_selection_idx}; self.snap_cache_dirty = True; self.redraw()

    def delete_selected(self):
        if not self.selection: return
        indices_to_remove = sorted(list(self.selection), reverse=True)
        for i in indices_to_remove: del self.raw_masks[i]
        self.selection.clear(); self.snap_cache_dirty = True; self.redraw()

    def toggle_split_mode(self):
        self.is_splitting = not self.is_splitting
        if self.is_splitting:
            if len(self.selection) != 1: messagebox.showinfo("Split", "Please select exactly one segment to split.", parent=self); self.is_splitting = False; return
            self._cancel_current_drawing(deactivate_tool=True)
            self.split_button.state(['pressed']); self.split_points = []
            self.title("Segmentation Editor - SPLITTING (Draw a line)")
        else: self.split_button.state(['!pressed']); self.title("Segmentation Editor")

    def execute_split(self):
        mask_to_split_idx = list(self.selection)[0]
        mask_to_split = self.raw_masks[mask_to_split_idx]['segmentation']
        p1, p2 = np.array(self.split_points[0])[::-1], np.array(self.split_points[1])[::-1]
        vec, vec_len = p2 - p1, np.linalg.norm(p2 - p1)
        if vec_len < 1e-6: messagebox.showwarning("Split", "The line is too short to define a direction.", parent=self); self.selection.clear(); self.toggle_split_mode(); self.redraw(); return
        vec_norm = vec / vec_len
        start_point = p1 - vec_norm * max(self.image_np.shape) * 2
        end_point = p2 + vec_norm * max(self.image_np.shape) * 2
        line_mask = np.zeros_like(mask_to_split, dtype=bool)
        rr, cc, _ = ai_core.line_aa(int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1]))
        valid = (rr >= 0) & (rr < line_mask.shape[0]) & (cc >= 0) & (cc < line_mask.shape[1])
        line_mask[rr[valid], cc[valid]] = True
        line_mask = ai_core.binary_dilation(line_mask, structure=ai_core.create_disk(3))
        remaining_mask = mask_to_split & ~line_mask
        labels, num_features = ai_core.measure.label(remaining_mask, connectivity=2, return_num=True)
        if num_features > 1:
            original_mask_data = self.raw_masks.pop(mask_to_split_idx)
            for i in range(1, num_features + 1):
                new_seg, new_area = (labels == i), np.sum(labels == i)
                if new_area > 50:
                    new_data = original_mask_data.copy()
                    new_data.update({'segmentation': new_seg, 'area': new_area, 'color': [random.randint(50, 200) for _ in range(3)]})
                    self.raw_masks.append(new_data)
            messagebox.showinfo("Split", f"Segment split into {num_features} new pieces.", parent=self)
            self.snap_cache_dirty = True
        else: messagebox.showwarning("Split", "Could not split the segment with the given line.", parent=self)
        self.selection.clear(); self.toggle_split_mode(); self.redraw()

    def apply_and_close(self): self.on_close_callback(self.raw_masks); self.destroy()

# ==================== INTERACTIVE EDITOR ====================
class FloorplanEditor:
    def __init__(self, app, sam_model, fig, ax, original_img, raw_masks, image_shape):
        self.app, self.fig, self.ax = app, fig, ax
        self.sam_model, self.original_img, self.raw_masks, self.image_shape = sam_model, original_img, raw_masks, image_shape
        self.edit_mode, self.view_mode = ai_core.DrawingMode.SELECT, ai_core.ViewMode.VECTOR
        self.finalized_polygons, self.exterior_polygon = [], None
        self.grid_y, self.grid_x = np.array([]), np.array([])
        self.segmentation_image = None
        self.output_image = np.full((*image_shape[:2], 3), ai_core.CONFIG["BACKGROUND_COLOR"], dtype=np.uint8)
        self.walls, self.lines, self.doors, self.windows, self.furniture = [], [], [], [], []
        self.current_drawing, self.drawing_start_pos = None, None
        self.is_dragging_divider, self.snap_enabled, self.ortho_enabled, self.shift_pressed = False, True, True, False
        self.temp_artist, self.slider_handle = None, None
        self.undo_stack, self.redo_stack = deque(maxlen=ai_core.CONFIG["MAX_UNDO_STEPS"]), deque()
        self.selection, self.drag_start_pos, self.control_points = [], None, []
        self.overlay_artists, self.hover_selection = [], None
        self.drag_original_objects, self.active_wall_for_opening = [], None
        self.is_area_selecting = False
        self.area_select_start_pos = None
        self.selection_box_mode = ai_core.SelectionBoxMode.WINDOW
        
        self.drag_mode = ai_core.DragMode.NONE
        self.active_handle = None 
        self.drag_preview_artists = []
        self._snap_points_cache = None
        self._snap_cache_dirty = True
        
        self.vectorization_mode = ai_core.VectorizationMode.IDEALIZE

        self._initialize_sam_predictor(sam_model)
        self.setup_canvas()
        self.reprocess_from_masks(save_state=False)
        self.save_state()
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

# In desktop_app.py, inside the FloorplanEditor class

# In desktop_app.py, inside the FloorplanEditor class

def reprocess_from_masks(self, new_masks=None, save_state=True):
    self.app.update_status("Reprocessing from segmentation...")
    if new_masks is not None:
        self.raw_masks = new_masks
    
    # Keep track of manually drawn items
    manual_walls, manual_lines = self.walls[:], self.lines[:]
    manual_doors, manual_windows, manual_furniture = self.doors[:], self.windows[:], self.furniture[:]

    # === THIS IS THE MAJOR CHANGE ===
    # Call the single, authoritative function from ai_core
    self.finalized_polygons, self.exterior_polygon = ai_core.vectorize_floorplan(
        self.raw_masks,
        self.image_shape,
        self.vectorization_mode
    )
    # ===============================

    # Restore manually drawn items
    self.walls, self.lines = [], manual_lines
    self.walls.extend(manual_walls)
    self.doors, self.windows, self.furniture = manual_doors, manual_windows, manual_furniture

    self.selection, self._snap_cache_dirty = [], True
    
    self.segmentation_image = ai_core.generate_segmentation_image(self.raw_masks, self.image_shape)
    self.output_image = self._generate_full_output_image()
    if save_state:
        self.save_state()
    self.redraw_all()
    self.app.update_status("Reprocessing complete.")

    def set_vectorization_mode(self, mode_string):
        self.vectorization_mode = ai_core.VectorizationMode(mode_string)
        self.app.update_status(f"Vectorization Mode: {self.vectorization_mode.value}")
        self.walls = [] 
        self.reprocess_from_masks()
        self.save_state()

    def _idealize_contour(self, contour, dominant_y_grid, dominant_x_grid):
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

    def _smart_simplify_contour(self, contour, line_tolerance_dot, min_segment_len, rdp_tolerance):
        if len(contour) < 4: return contour
        final_points = [contour[0]]
        i = 0
        while i < len(contour) - 1:
            best_j = -1
            for j in range(i + 2, len(contour)):
                line_vec = contour[j] - contour[i]
                line_len = np.linalg.norm(line_vec)
                if line_len < 1e-6: continue
                line_unit_vec = line_vec / line_len
                is_straight = True
                for k in range(i + 1, j):
                    mid_vec = contour[k] - contour[i]
                    dot_product = np.dot(line_unit_vec, mid_vec / (np.linalg.norm(mid_vec) + 1e-9))
                    if dot_product < line_tolerance_dot: is_straight = False; break
                if is_straight: best_j = j
                else: break
            if best_j != -1 and np.linalg.norm(contour[best_j] - contour[i]) >= min_segment_len:
                final_points.append(contour[best_j]); i = best_j
            else:
                curve_start_index, curve_end_index = i, i + 1
                for j in range(i + 2, len(contour)): curve_end_index = j
                curved_segment = contour[curve_start_index:curve_end_index+1]
                if len(curved_segment) > 2:
                    simplified_curve = ai_core.measure.approximate_polygon(curved_segment, tolerance=rdp_tolerance)
                    final_points.extend(simplified_curve[1:])
                else: final_points.extend(curved_segment[1:])
                i = curve_end_index
        if np.linalg.norm(final_points[0] - final_points[-1]) > 1: final_points.append(final_points[0])
        return np.array(final_points)
    
    def _hybrid_simplify_contour(self, contour):
        if len(contour) < 3: return contour
        
        new_points = [contour[0]]
        angle_tol = ai_core.CONFIG["HYBRID_ORTHO_ANGLE_TOLERANCE"]
        
        i = 0
        while i < len(contour) - 1:
            p1 = new_points[-1]
            p2 = contour[i+1]
            
            dy, dx = p2[0] - p1[0], p2[1] - p1[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                i += 1
                continue
            
            angle = ai_core.math.degrees(ai_core.math.atan2(dy, dx)) % 360

            if (abs(angle) < angle_tol) or (abs(angle - 180) < angle_tol) or (abs(angle - 360) < angle_tol):
                new_p2 = (p1[0], p2[1]) # Snap to horizontal
            elif (abs(angle - 90) < angle_tol) or (abs(angle - 270) < angle_tol):
                new_p2 = (p2[0], p1[1]) # Snap to vertical
            else:
                new_p2 = tuple(p2) # Keep original
            
            if tuple(new_p2) != tuple(p1):
                 new_points.append(np.array(new_p2))
            i += 1
            
        return ai_core.measure.approximate_polygon(np.array(new_points), tolerance=ai_core.CONFIG["RDP_TOLERANCE"])

    def _get_final_architectural_contour(self, contour):
        if len(contour) < 4: return None
        simplified = ai_core.process_contour(contour)
        if simplified.size < 3: return None

        mode = self.vectorization_mode
        if mode in [ai_core.VectorizationMode.MINIMALIST, ai_core.VectorizationMode.TACTILE]:
            base_map = {ai_core.VectorizationMode.MINIMALIST: ai_core.VectorizationMode.IDEALIZE, ai_core.VectorizationMode.TACTILE: ai_core.VectorizationMode.TRACE}
            mode = base_map.get(self.vectorization_mode, ai_core.VectorizationMode.TRACE)
            self.app.update_status(f"Processing geometry using {mode.value} rules...")

        if mode == ai_core.VectorizationMode.IDEALIZE:
            if self.grid_y.size < 2 or self.grid_x.size < 2:
                 print("Warning: Idealization grid not available, falling back to Hybrid.")
                 return self._hybrid_simplify_contour(simplified)
            idealized = self._idealize_contour(simplified, self.grid_y, self.grid_x)
            if idealized.size < 3: return None
            return ai_core.measure.approximate_polygon(idealized, tolerance=ai_core.CONFIG["RDP_TOLERANCE"])
        
        elif mode == ai_core.VectorizationMode.HYBRID:
            return self._hybrid_simplify_contour(simplified)
            
        elif mode == ai_core.VectorizationMode.ARCHITECTURAL_CLEAN:
            hybrid_points = self._hybrid_simplify_contour(simplified)
            return ai_core.measure.approximate_polygon(hybrid_points, tolerance=ai_core.CONFIG["RDP_TOLERANCE"])
            
        elif mode == ai_core.VectorizationMode.TRACE:
            return self._smart_simplify_contour(simplified, ai_core.CONFIG["STRAIGHT_LINE_TOLERANCE"], ai_core.CONFIG["MIN_LINE_SEGMENT_LENGTH"], ai_core.CONFIG["RDP_TOLERANCE"])
        
        elif mode == ai_core.VectorizationMode.VECTOR_TRACE_PLUS:
            return self._smart_simplify_contour(simplified, ai_core.CONFIG["TRACE_PLUS_STRAIGHT_LINE_TOLERANCE"], ai_core.CONFIG["MIN_LINE_SEGMENT_LENGTH"], ai_core.CONFIG["TRACE_PLUS_RDP_TOLERANCE"])

        elif mode == ai_core.VectorizationMode.SIMPLIFY:
            return ai_core.measure.approximate_polygon(simplified, tolerance=ai_core.CONFIG["RDP_TOLERANCE"])

        elif mode == ai_core.VectorizationMode.RAW:
            return contour

        return None

    def _get_finalized_polygons(self, room_masks):
        polygons = []
        for m in room_masks:
            contours = ai_core.measure.find_contours(m['segmentation'], 0.5)
            if not contours: continue
            largest_contour = max(contours, key=lambda c: len(c))
            processed_contour = self._get_final_architectural_contour(largest_contour)
            if processed_contour is not None and len(processed_contour) > 3:
                polygons.append(processed_contour)
        return polygons

    def _get_exterior_polygon(self, finalized_polygons):
        if not finalized_polygons: return None
        shapely_polygons = [ai_core.Polygon(p) for p in finalized_polygons if p is not None and len(p) > 2 and ai_core.Polygon(p).is_valid]
        if not shapely_polygons: return None
        try:
            buffered_polygons = [p.buffer(0) for p in shapely_polygons]
            unified_shape = ai_core.unary_union(buffered_polygons)
        except Exception as e:
            print(f"Warning: Could not unify polygons for exterior: {e}"); return None
        if unified_shape.is_empty: return None
        if isinstance(unified_shape, ai_core.Polygon): exterior_contour = np.array(unified_shape.exterior.coords)
        elif isinstance(unified_shape, ai_core.MultiPolygon): exterior_contour = np.array(max(unified_shape.geoms, key=lambda p: p.area).exterior.coords)
        else: return None
        return self._get_final_architectural_contour(exterior_contour)
    
    def _perform_area_selection(self, p1, p2, selection_mode, add_to_selection=False):
        y1, x1 = p1; y2, x2 = p2
        selection_box = ai_core.box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        newly_selected = []
        object_sources = {'wall': self.walls, 'line': self.lines, 'door': self.doors, 'window': self.windows}
        
        for obj_type, obj_list in object_sources.items():
            for i, obj in enumerate(obj_list):
                if not obj: continue
                points = obj['points'] if isinstance(obj, dict) else obj
                if not points or len(points) < 2: continue
                obj_line = ai_core.LineString([(p[1], p[0]) for p in points])
                if (selection_mode == ai_core.SelectionBoxMode.WINDOW and selection_box.contains(obj_line)) or \
                   (selection_mode == ai_core.SelectionBoxMode.CROSSING and selection_box.intersects(obj_line)):
                    newly_selected.append({'type': obj_type, 'index': i})

        all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
        for poly_idx, poly in enumerate(all_polys):
            if poly is None: continue
            for seg_idx in range(len(poly) - 1):
                p1_seg, p2_seg = poly[seg_idx], poly[seg_idx + 1]
                obj_line = ai_core.LineString([(p1_seg[1], p1_seg[0]), (p2_seg[1], p2_seg[0])])
                if (selection_mode == ai_core.SelectionBoxMode.WINDOW and selection_box.contains(obj_line)) or \
                   (selection_mode == ai_core.SelectionBoxMode.CROSSING and selection_box.intersects(obj_line)):
                    newly_selected.append({'type': 'generated_wall', 'poly_index': poly_idx, 'seg_index': seg_idx})
        
        if add_to_selection:
            current_sel_set = {tuple(sorted(d.items())) for d in self.selection}
            for item in newly_selected:
                if tuple(sorted(item.items())) not in current_sel_set:
                    self.selection.append(item)
        else:
            self.selection = newly_selected
            
        self.app.update_status(f"Selected {len(self.selection)} items.")
        self.redraw_all()

    def _find_closest_object(self, pos):
        min_dist, found_object = ai_core.CONFIG["SNAP_DISTANCE_PIXELS"], None
        for i, wall in enumerate(self.walls):
            if not wall or not wall.get('points') or len(wall['points']) != 2: continue
            p1, p2 = wall['points']; dist = ai_core.Point(pos[::-1]).distance(ai_core.LineString([p1[::-1], p2[::-1]]))
            if dist < min_dist: min_dist, found_object = dist, {'type': 'wall', 'index': i}
        
        all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
        for poly_idx, poly in enumerate(all_polys):
            if poly is None: continue
            for seg_idx in range(len(poly) - 1):
                p1, p2 = poly[seg_idx], poly[seg_idx + 1]; dist = ai_core.Point(pos[::-1]).distance(ai_core.LineString([p1[::-1], p2[::-1]]))
                if dist < min_dist: min_dist, found_object = dist, {'type': 'generated_wall', 'poly_index': poly_idx, 'seg_index': seg_idx}
        
        for i, segment in enumerate(self.lines):
            if not segment or len(segment) != 2: continue
            p1, p2 = segment; dist = ai_core.Point(pos[::-1]).distance(ai_core.LineString([p1[::-1], p2[::-1]]))
            if dist < min_dist: min_dist, found_object = dist, {'type': 'line', 'index': i}

        for obj_type, obj_list in [('door', self.doors), ('window', self.windows)]:
             for i, segment in enumerate(obj_list):
                if not segment or len(segment) != 2: continue
                p1, p2 = segment; dist = ai_core.Point(pos[::-1]).distance(ai_core.LineString([p1[::-1], p2[::-1]]))
                if dist < min_dist: min_dist, found_object = dist, {'type': obj_type, 'index': i}
        return found_object

    def _get_segment_at_pos(self, pos, segment_list, is_wall=False):
        for i, seg in enumerate(segment_list):
            points = seg['points'] if is_wall else seg
            if not points or len(points) != 2: continue
            p1, p2 = np.array(points[0]), np.array(points[1])
            line = ai_core.LineString([p1, p2]); point = ai_core.Point(pos)
            if point.distance(line) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]:
                proj_point = line.interpolate(line.project(point))
                return i, list(proj_point.coords)[0]
        return None, None

    def _get_generated_segment_at_pos(self, pos):
        all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
        for poly_idx, poly in enumerate(all_polys):
            if poly is None: continue
            for seg_idx in range(len(poly) - 1):
                p1, p2 = np.array(poly[seg_idx]), np.array(poly[seg_idx + 1])
                line = ai_core.LineString([p1, p2]); point = ai_core.Point(pos)
                if point.distance(line) < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]:
                    proj_point = line.interpolate(line.project(point))
                    return poly_idx, seg_idx, list(proj_point.coords)[0]
        return None, None, None

    def _calculate_arc_points(self, p1_start, p2_end, p3_mid):
        p1, p2, p3 = np.array(p1_start), np.array(p2_end), np.array(p3_mid)
        A = np.vstack((p1, p2, p3)); B = np.ones(3); C = np.sum(A**2, axis=1)
        a = np.linalg.det(np.column_stack((A, B)))
        if abs(a) < 1e-6: return [p1.tolist(), p2.tolist()]
        Dx = np.linalg.det(np.column_stack((C, A[:, 1], B)))
        Dy = -np.linalg.det(np.column_stack((C, A[:, 0], B)))
        center = np.array([Dx, Dy]) / (2 * a)
        radius = np.linalg.norm(p1 - center)
        start_angle, mid_angle, end_angle = (ai_core.math.atan2(p[0]-center[0], p[1]-center[1]) for p in [p1,p3,p2])
        if (end_angle - start_angle) % (2*ai_core.math.pi) > (mid_angle - start_angle) % (2*ai_core.math.pi):
            start_angle, end_angle = end_angle, start_angle
        points = []
        for angle in np.linspace(start_angle, end_angle, ai_core.CONFIG["ARC_WALL_SEGMENTS"] + 1):
            points.append([center[0] + radius*ai_core.math.sin(angle), center[1] + radius*ai_core.math.cos(angle)])
        return points

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.xdata < self.split_pos: return
        if abs(event.xdata - self.split_pos) < 30 and event.button == 1: self.is_dragging_divider = True; return
        pos = np.array([event.ydata, event.xdata])
        
        if event.button == 3: # Right-click context menu
            self._show_wall_context_menu(event)
            return

        if event.button == 1 and self.edit_mode == ai_core.DrawingMode.SELECT:
            self.save_state()
            handle_clicked = self._find_handle_at_pos(pos)
            if handle_clicked:
                self.drag_mode = ai_core.DragMode.STRETCH
                self.active_handle = handle_clicked
                self.drag_start_pos = pos
                sel_item = self.selection[self.active_handle['sel_index']]
                if sel_item['type'] == 'generated_wall':
                    new_wall_index = self._convert_and_find_new_wall(sel_item)
                    if new_wall_index is not None:
                        self.selection = [{'type': 'wall', 'index': new_wall_index}]
                        self.active_handle['sel_index'] = 0 
                        self.app.update_status("Generated wall converted to manual walls for editing.")
                        self.redraw_all()
                    else: 
                        self.drag_mode = ai_core.DragMode.NONE; return
                return

            clicked_object = self._find_closest_object(pos)
            if clicked_object:
                if self.shift_pressed:
                    if clicked_object in self.selection: self.selection.remove(clicked_object)
                    else: self.selection.append(clicked_object)
                else:
                    if clicked_object not in self.selection: self.selection = [clicked_object]
                
                self.drag_mode = ai_core.DragMode.MOVE
                self.drag_start_pos = pos
                self.drag_original_objects = [copy.deepcopy(self._get_object_from_selection(s)) for s in self.selection]
                self.app.update_status(f"Selected {len(self.selection)} item(s). Drag to move or stretch handles.")
            else:
                if not self.shift_pressed: self.selection = []
                self.is_area_selecting = True
                self.area_select_start_pos = pos
            self.update_overlay(draw_now=True)
            return
        
        snap_pos = np.array(self._snap_to_targets(event.xdata, event.ydata)[::-1])
        if self.edit_mode in [ai_core.DrawingMode.WALL, ai_core.DrawingMode.LINE, ai_core.DrawingMode.RECTANGLE, ai_core.DrawingMode.CIRCLE, ai_core.DrawingMode.ARC_WALL]:
            self.save_state(); self.drawing_start_pos = snap_pos
            if self.edit_mode == ai_core.DrawingMode.ARC_WALL:
                self.control_points.append(snap_pos.tolist())
                if len(self.control_points)==1: self.app.update_status("Arc Wall: Set end point.")
                elif len(self.control_points)==2: self.app.update_status("Arc Wall: Define curve.")
                elif len(self.control_points)==3:
                    arc_points = self._calculate_arc_points(*self.control_points)
                    for i in range(len(arc_points)-1): self.walls.append({'points': arc_points[i:i+2], 'thickness': ai_core.CONFIG['WALL_THICKNESS_PIXELS']})
                    self.control_points = []; self._snap_cache_dirty = True; self.redraw_all()
                    self.app.update_status("Arc wall created.")
        elif self.edit_mode in [ai_core.DrawingMode.DOOR, ai_core.DrawingMode.WINDOW]:
            self.save_state(); wall_obj = self._find_closest_object(snap_pos)
            if wall_obj and wall_obj['type'] in ['wall', 'generated_wall']:
                self.active_wall_for_opening = self._get_object_from_selection(wall_obj)
                p1, p2 = self.active_wall_for_opening['points']; line_seg = ai_core.LineString([p1, p2])
                self.drawing_start_pos = np.array(line_seg.interpolate(line_seg.project(ai_core.Point(snap_pos))).coords[0])
            else: self.app.update_status("Start by clicking on a wall."); self.drawing_start_pos = None
        elif self.edit_mode == ai_core.DrawingMode.SPLIT_WALL:
            self.save_state()
            wall_idx, split_point = self._get_segment_at_pos(pos, self.walls, is_wall=True)
            if wall_idx is not None:
                wall = self.walls.pop(wall_idx); p1, p2, th = *wall['points'], wall['thickness']
                self.walls.extend([{'points': [p1, split_point], 'thickness': th}, {'points': [split_point, p2], 'thickness': th}])
                self._snap_cache_dirty = True; self.redraw_all()
                self.app.update_status(f"Manual Wall {wall_idx} split.")
            else:
                poly_idx, seg_idx, split_point = self._get_generated_segment_at_pos(pos)
                if poly_idx is not None:
                    self._convert_generated_poly_to_walls(poly_idx, split_at_seg=(seg_idx, split_point))
                    self._snap_cache_dirty = True; self.redraw_all()
                    self.app.update_status(f"Generated wall split.")
                else: self.app.update_status("No wall found.")

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None: self._clear_temp_artist(); return
        if self.is_dragging_divider: self.update_divider_position(event.xdata); return
        if event.xdata < self.split_pos: self._clear_temp_artist(); return
        
        if self.is_area_selecting and self.area_select_start_pos is not None:
            start_x = self.area_select_start_pos[1]
            end_x = event.xdata
            self.selection_box_mode = ai_core.SelectionBoxMode.CROSSING if end_x < start_x else ai_core.SelectionBoxMode.WINDOW
            self.update_overlay(draw_now=True, area_select_end_pos=np.array([event.ydata, event.xdata]))
            return

        x, y = self._snap_to_targets(event.xdata, event.ydata)
        pos = np.array([y, x])
        
        if self.edit_mode == ai_core.DrawingMode.SELECT and self.drag_start_pos is not None:
            self._update_drag_previews(pos)
            return

        if self.edit_mode == ai_core.DrawingMode.SPLIT_WALL:
            hover_obj = self._find_closest_object(pos)
            self.hover_selection = hover_obj if hover_obj and hover_obj['type'] in ['wall', 'generated_wall'] else None
            self.update_overlay(draw_now=True)
            
        if self.drawing_start_pos is not None:
            end_pos = pos
            if self.active_wall_for_opening:
                p1, p2 = self.active_wall_for_opening['points']; line_seg = ai_core.LineString([p1, p2])
                end_pos = np.array(line_seg.interpolate(line_seg.project(ai_core.Point(pos))).coords[0])
            elif self.ortho_enabled and not self.shift_pressed and self.edit_mode in [ai_core.DrawingMode.WALL, ai_core.DrawingMode.LINE, ai_core.DrawingMode.DOOR, ai_core.DrawingMode.WINDOW]:
                 x_draw, y_draw = self._apply_ortho_constraint((self.drawing_start_pos[1], self.drawing_start_pos[0]), (x, y))
                 end_pos = np.array([y_draw, x_draw])
            self.current_drawing = [self.drawing_start_pos.tolist(), end_pos.tolist()]
        
        self.update_overlay(draw_now=True, temp_pos=pos.tolist())

    def on_release(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        if self.is_dragging_divider: self.is_dragging_divider = False; return
        if event.xdata < self.split_pos: return

        if self.edit_mode == ai_core.DrawingMode.SELECT and self.drag_mode != ai_core.DragMode.NONE:
            final_pos = np.array(self._snap_to_targets(event.xdata, event.ydata)[::-1])
            self._finalize_drag(final_pos)
            self.drag_mode = ai_core.DragMode.NONE
            self.drag_start_pos = None
            self.active_handle = None
            self.drag_original_objects = []
            self._clear_drag_previews()
            self._snap_cache_dirty = True
            self.redraw_all()
            return

        if self.is_area_selecting:
            self.is_area_selecting = False
            end_pos = np.array([event.ydata, event.xdata])
            if np.linalg.norm(end_pos - self.area_select_start_pos) > 5:
                self._perform_area_selection(self.area_select_start_pos, end_pos, self.selection_box_mode, add_to_selection=self.shift_pressed)
            self.area_select_start_pos = None
            self.update_overlay(draw_now=True)
            return
            
        if self.drawing_start_pos is not None and self.current_drawing is not None:
            start_pos, end_pos = np.array(self.current_drawing[0]), np.array(self.current_drawing[1])
            if np.linalg.norm(start_pos - end_pos) > 5:
                mode_actions = {
                    ai_core.DrawingMode.WALL: lambda: self.walls.append({'points': self.current_drawing, 'thickness': ai_core.CONFIG['WALL_THICKNESS_PIXELS']}),
                    ai_core.DrawingMode.LINE: lambda: self.lines.append(self.current_drawing),
                    ai_core.DrawingMode.DOOR: lambda: self.doors.append(self.current_drawing),
                    ai_core.DrawingMode.WINDOW: lambda: self.windows.append(self.current_drawing),
                    ai_core.DrawingMode.RECTANGLE: lambda: self.walls.extend([{'points':[[start_pos[0],start_pos[1]],[start_pos[0],end_pos[1]]],'thickness':ai_core.CONFIG['WALL_THICKNESS_PIXELS']},{'points':[[start_pos[0],end_pos[1]],[end_pos[0],end_pos[1]]],'thickness':ai_core.CONFIG['WALL_THICKNESS_PIXELS']},{'points':[[end_pos[0],end_pos[1]],[end_pos[0],start_pos[1]]],'thickness':ai_core.CONFIG['WALL_THICKNESS_PIXELS']},{'points':[[end_pos[0],start_pos[1]],[start_pos[0],start_pos[1]]],'thickness':ai_core.CONFIG['WALL_THICKNESS_PIXELS']}]),
                    ai_core.DrawingMode.CIRCLE: lambda: self.walls.extend([{'points':[ [start_pos[0]+np.linalg.norm(end_pos-start_pos)*ai_core.math.sin(2*ai_core.math.pi*i/32), start_pos[1]+np.linalg.norm(end_pos-start_pos)*ai_core.math.cos(2*ai_core.math.pi*i/32)], [start_pos[0]+np.linalg.norm(end_pos-start_pos)*ai_core.math.sin(2*ai_core.math.pi*(i+1)/32), start_pos[1]+np.linalg.norm(end_pos-start_pos)*ai_core.math.cos(2*ai_core.math.pi*(i+1)/32)] ],'thickness':ai_core.CONFIG['WALL_THICKNESS_PIXELS']} for i in range(32)])
                }
                if self.edit_mode in mode_actions:
                    mode_actions[self.edit_mode]()
                    self._snap_cache_dirty = True
        self.drawing_start_pos, self.current_drawing = None, None
        self.active_wall_for_opening = None; self.redraw_all()

    def on_key_press(self, event):
        if event.key == 'escape':
            self.drawing_start_pos, self.current_drawing, self.control_points = None, None, []
            self.selection = []; self.hover_selection = None
            self.drag_original_objects = []; self.active_wall_for_opening = None
            self.drag_mode = ai_core.DragMode.NONE; self._clear_drag_previews()
            self.app.update_status("Action cancelled."); self.redraw_all()
        elif event.key == 'ctrl+z': self.undo()
        elif event.key == 'ctrl+y': self.redo()
        elif event.key == 'shift': self.shift_pressed = True
        elif event.key == 'delete' and self.selection:
            self.save_state()
            deletions = {'wall': [], 'line': [], 'door': [], 'window': [], 'generated_wall': []}
            for item in self.selection: deletions[item['type']].append(item)
            if deletions['generated_wall']:
                converted_polys = {}
                for item in deletions['generated_wall']:
                    if item['poly_index'] not in converted_polys: converted_polys[item['poly_index']] = []
                    converted_polys[item['poly_index']].append(item['seg_index'])
                for poly_idx, seg_indices in sorted(converted_polys.items(), reverse=True):
                    self._convert_generated_poly_to_walls(poly_idx, delete_seg_indices=sorted(seg_indices, reverse=True))
            for obj_type in ['wall', 'line', 'door', 'window']:
                indices_to_delete = sorted([item['index'] for item in deletions[obj_type]], reverse=True)
                obj_list = getattr(self, obj_type + 's')
                for index in indices_to_delete: obj_list.pop(index)
            self.app.update_status(f"Deleted {len(self.selection)} item(s).")
            self.selection = []; self._snap_cache_dirty = True; self.redraw_all()

    def on_key_release(self, event):
        if event.key == 'shift': self.shift_pressed = False

    def redraw_all(self):
        if self.view_mode == ai_core.ViewMode.VECTOR:
            self.output_image = self._generate_full_output_image()
            self.right_img.set_data(self.output_image)
        elif self.view_mode == ai_core.ViewMode.SEGMENTATION:
            if self.segmentation_image is None:
                 self.segmentation_image = ai_core.generate_segmentation_image(self.raw_masks, self.image_shape)
            self.right_img.set_data(self.segmentation_image)
        elif self.view_mode == ai_core.ViewMode.ORIGINAL:
            self.right_img.set_data(self.original_img)
        self.update_overlay(draw_now=True)

    def _clear_overlay_artists(self):
        for artist in self.overlay_artists: artist.remove()
        self.overlay_artists.clear()

    def update_overlay(self, draw_now=False, temp_pos=None, area_select_end_pos=None):
        self._clear_overlay_artists()
        if self.view_mode != ai_core.ViewMode.VECTOR:
            if draw_now: self.fig.canvas.draw_idle()
            return

        if self.is_area_selecting and self.area_select_start_pos is not None and area_select_end_pos is not None:
            y1, x1 = self.area_select_start_pos; y2, x2 = area_select_end_pos
            width, height = abs(x2 - x1), abs(y2 - y1)
            
            if self.selection_box_mode == ai_core.SelectionBoxMode.WINDOW:
                style = {'fill': True, 'facecolor': ai_core.CONFIG["WINDOW_SELECTION_FILL_COLOR"], 'alpha': 0.2, 'ls': '-', 'lw': 1, 'edgecolor': ai_core.CONFIG["WINDOW_SELECTION_EDGE_COLOR"], 'zorder': 25}
            else:
                style = {'fill': True, 'facecolor': ai_core.CONFIG["CROSSING_SELECTION_FILL_COLOR"], 'alpha': 0.2, 'ls': '--', 'lw': 1.5, 'edgecolor': ai_core.CONFIG["CROSSING_SELECTION_EDGE_COLOR"], 'zorder': 25}
            
            rect = patches.Rectangle((min(x1, x2), min(y1, y2)), width, height, **style)
            self.ax.add_patch(rect); self.overlay_artists.append(rect)

        for sel_idx, sel_item in enumerate(self.selection):
            obj = self._get_object_from_selection(sel_item)
            if obj and obj.get('points') and len(obj['points']) > 1:
                p1, p2 = obj['points']
                line = mlines.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color=ai_core.CONFIG["SELECTION_COLOR"], lw=3, zorder=20)
                self.ax.add_line(line); self.overlay_artists.append(line)
            
                if len(self.selection) == 1 and sel_item['type'] in ['wall', 'line', 'generated_wall']:
                    s = ai_core.CONFIG["HANDLE_SIZE_PIXELS"]
                    h1 = patches.Rectangle((p1[1] - s/2, p1[0] - s/2), s, s, color=ai_core.CONFIG["HANDLE_COLOR"], zorder=22)
                    h2 = patches.Rectangle((p2[1] - s/2, p2[0] - s/2), s, s, color=ai_core.CONFIG["HANDLE_COLOR"], zorder=22)
                    self.ax.add_patch(h1); self.overlay_artists.append(h1)
                    self.ax.add_patch(h2); self.overlay_artists.append(h2)

        if self.hover_selection:
            obj = self._get_object_from_selection(self.hover_selection)
            if obj and obj.get('points') and len(obj['points']) > 1:
                p1, p2 = obj['points']
                line = mlines.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color=ai_core.CONFIG["HOVER_COLOR"], ls='--', lw=2, zorder=21)
                self.ax.add_line(line); self.overlay_artists.append(line)

        if self.current_drawing:
            p1, p2 = self.current_drawing
            if self.edit_mode in [ai_core.DrawingMode.RECTANGLE, ai_core.DrawingMode.CIRCLE]:
                if self.edit_mode == ai_core.DrawingMode.RECTANGLE:
                    y1,x1 = p1; y2,x2 = p2
                    preview = patches.Rectangle((min(x1,x2), min(y1,y2)), abs(x2-x1), abs(y2-y1), fill=False, color=ai_core.CONFIG["PREVIEW_COLOR"], ls='--', lw=1.5, zorder=20)
                else:
                    radius = np.linalg.norm(np.array(p2) - np.array(p1))
                    preview = patches.Circle((p1[1], p1[0]), radius, fill=False, color=ai_core.CONFIG["PREVIEW_COLOR"], ls='--', lw=1.5, zorder=20)
                self.ax.add_patch(preview); self.overlay_artists.append(preview)
            else:
                preview = mlines.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color=ai_core.CONFIG["PREVIEW_COLOR"], ls='--', lw=1.5, zorder=20)
                self.ax.add_line(preview); self.overlay_artists.append(preview)

        if draw_now: self.fig.canvas.draw_idle()
        
    def _show_wall_context_menu(self, event):
        clicked_object = self._find_closest_object(np.array([event.ydata, event.xdata]))
        if not clicked_object or clicked_object['type'] not in ['wall', 'generated_wall']: return
        
        if clicked_object not in self.selection:
            self.selection = [clicked_object]
            self.redraw_all()

        menu = Menu(self.app.root, tearoff=0)
        menu.add_command(label="Set Thickness...", command=lambda: self._set_wall_thickness(clicked_object))
        menu.add_command(label="Delete", command=lambda: self.on_key_press(type('obj', (object,), {'key': 'delete'})))
        widget = self.fig.canvas.get_tk_widget(); menu.post(widget.winfo_rootx() + int(event.x), widget.winfo_rooty() + int(event.y))

    def _set_wall_thickness(self, target_obj):
        walls_in_selection = [s for s in self.selection if s['type'] == 'wall']
        if walls_in_selection:
            current_thickness = self.walls[walls_in_selection[0]['index']].get('thickness', ai_core.CONFIG['WALL_THICKNESS_PIXELS'])
            new_thickness = simpledialog.askfloat("Set Group Thickness", f"Enter new thickness for {len(walls_in_selection)} walls:", parent=self.app.root, initialvalue=current_thickness, minvalue=1)
            if new_thickness is not None:
                self.save_state()
                for sel_item in walls_in_selection: self.walls[sel_item['index']]['thickness'] = new_thickness
                self.redraw_all()
        elif target_obj['type'] == 'wall':
            current_thickness = self.walls[target_obj['index']].get('thickness', ai_core.CONFIG['WALL_THICKNESS_PIXELS'])
            new_thickness = simpledialog.askfloat("Set Wall Thickness", "Enter new thickness in pixels:", parent=self.app.root, initialvalue=current_thickness, minvalue=1)
            if new_thickness is not None:
                self.save_state()
                self.walls[target_obj['index']]['thickness'] = new_thickness
                self.redraw_all()

    def _get_object_from_selection(self, selection_dict, get_ref=False):
        if not selection_dict: return None
        obj_type = selection_dict['type']
        try:
            if obj_type == 'wall':
                obj = self.walls[selection_dict['index']]
                return obj if get_ref else copy.deepcopy(obj)
            elif obj_type == 'generated_wall':
                all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
                if selection_dict['poly_index'] >= len(all_polys): return None
                poly = all_polys[selection_dict['poly_index']]
                if poly is None or selection_dict['seg_index'] >= len(poly) - 1: return None
                p1, p2 = poly[selection_dict['seg_index']], poly[selection_dict['seg_index'] + 1]
                return {'points': [p1.tolist(), p2.tolist()], 'thickness': ai_core.CONFIG['WALL_THICKNESS_PIXELS'], 'type': 'generated_wall'}
            elif obj_type in ['line', 'door', 'window']:
                obj_list = getattr(self, obj_type + 's')
                obj_points = obj_list[selection_dict['index']]
                wrapped_obj = {'points': obj_points, 'type': obj_type}
                if get_ref:
                    return obj_list[selection_dict['index']]
                return copy.deepcopy(wrapped_obj)

        except (IndexError, AttributeError): return None
        return None

    def _convert_generated_poly_to_walls(self, poly_idx, split_at_seg=None, delete_seg_indices=None):
        all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
        if poly_idx >= len(all_polys): return
        
        is_exterior = (self.exterior_polygon is not None) and (poly_idx == len(self.finalized_polygons))
        
        if is_exterior:
            poly_to_edit = self.exterior_polygon
            self.exterior_polygon = None
        else:
            poly_to_edit = self.finalized_polygons.pop(poly_idx)

        delete_indices_set = set(delete_seg_indices) if delete_seg_indices else set()
        for i in range(len(poly_to_edit) - 1):
            if i in delete_indices_set: continue
            p1, p2, thickness = poly_to_edit[i].tolist(), poly_to_edit[i+1].tolist(), ai_core.CONFIG['WALL_THICKNESS_PIXELS']
            if split_at_seg and i == split_at_seg[0]:
                split_point = split_at_seg[1]
                self.walls.extend([{'points': [p1, split_point], 'thickness': thickness}, {'points': [split_point, p2], 'thickness': thickness}])
            else: self.walls.append({'points': [p1, p2], 'thickness': thickness})
        self._snap_cache_dirty = True

    def _initialize_sam_predictor(self, sam_model): self.sam_predictor = SamPredictor(sam_model); self.sam_predictor.set_image(self.original_img)
    def _generate_full_output_image(self): return ai_core.generate_output_image(self.finalized_polygons, self.exterior_polygon, self.walls, self.lines, self.doors, self.windows, self.furniture, self.image_shape, self.vectorization_mode)
    def setup_canvas(self): self.ax.clear(); self.ax.axis('off'); self.split_pos = self.image_shape[1] // 2; self.left_img = self.ax.imshow(self.original_img); self.right_img = self.ax.imshow(self.output_image); self.divider = self.ax.axvline(self.split_pos, color='red', lw=1.5); self._create_slider_handle(); self.update_clip_paths(); self.fig.canvas.draw_idle()
    def _create_slider_handle(self): y_center = self.image_shape[0] // 2; r = 25; c = patches.Circle((self.split_pos, y_center), r, fc='k', alpha=0.4, ec='w', lw=1.5, zorder=10); a_s, a_o = r*0.4, r*0.3; a_l = patches.Polygon([(self.split_pos-a_o-a_s, y_center), (self.split_pos-a_o, y_center+a_s), (self.split_pos-a_o, y_center-a_s)], closed=True, fc='w', alpha=0.8, zorder=11); a_r = patches.Polygon([(self.split_pos+a_o+a_s, y_center), (self.split_pos+a_o, y_center+a_s), (self.split_pos+a_o, y_center-a_s)], closed=True, fc='w', alpha=0.8, zorder=11); self.slider_handle = {'c': c, 'al': a_l, 'ar': a_r}; self.ax.add_patch(c); self.ax.add_patch(a_l); self.ax.add_patch(a_r)
    def _update_slider_handle_position(self):
        if not self.slider_handle: return
        x, y = self.split_pos, self.image_shape[0]//2; r = 25; a_s, a_o = r*0.4, r*0.3
        self.slider_handle['c'].center = (x, y); self.slider_handle['al'].set_xy(np.array([(x-a_o-a_s, y), (x-a_o, y+a_s), (x-a_o, y-a_s)])); self.slider_handle['ar'].set_xy(np.array([(x+a_o+a_s, y), (x+a_o, y+a_s), (x+a_o, y-a_s)]))
    def update_clip_paths(self): w, h = self.image_shape[1], self.image_shape[0]; self.left_img.set_clip_path(patches.Rectangle((0,0), self.split_pos, h, transform=self.ax.transData)); self.right_img.set_clip_path(patches.Rectangle((self.split_pos,0), w-self.split_pos, h, transform=self.ax.transData)); self._update_slider_handle_position()
    def update_divider_position(self, val): self.split_pos = int(np.clip(val, 0, self.image_shape[1])); self.divider.set_xdata([self.split_pos, self.split_pos]); self.update_clip_paths(); self.fig.canvas.draw_idle()
    def set_edit_mode(self, mode): self.edit_mode = mode; self.app.update_status(f"Mode: {mode.name.replace('_', ' ').title()}"); self.drawing_start_pos, self.current_drawing, self.selection, self.drag_start_pos, self.control_points = None, None, [], None, []; self._clear_temp_artist(); self.app.update_button_states(mode); self.redraw_all()
    def toggle_snap(self): self.snap_enabled = not self.snap_enabled; self.app.update_snap_button_state(self.snap_enabled); self.app.update_status(f"Snap {'On' if self.snap_enabled else 'Off'}")
    def toggle_ortho(self): self.ortho_enabled = not self.ortho_enabled; self.app.update_ortho_button_state(self.ortho_enabled); self.app.update_status(f"Ortho {'On' if self.ortho_enabled else 'Off'}")
    def save_state(self): self.undo_stack.append(copy.deepcopy({'raw_masks': self.raw_masks, 'walls':self.walls, 'lines':self.lines, 'doors':self.doors, 'windows':self.windows, 'furniture':self.furniture, 'finalized_polygons': self.finalized_polygons, 'exterior_polygon': self.exterior_polygon})); self.redo_stack.clear()
    def undo(self):
        if len(self.undo_stack) > 1: self.redo_stack.append(self.undo_stack.pop()); self._restore_state(self.undo_stack[-1]); self.app.update_status("Undo successful")
        else: self.app.update_status("Nothing more to undo")
    def redo(self):
        if self.redo_stack: state = self.redo_stack.pop(); self.undo_stack.append(state); self._restore_state(state); self.app.update_status("Redo successful")
        else: self.app.update_status("Nothing to redo")
    
    def _restore_state(self, state):
        s = copy.deepcopy(state)
        self.raw_masks = s['raw_masks']
        self.finalized_polygons = s.get('finalized_polygons', [])
        self.exterior_polygon = s.get('exterior_polygon', None)
        self.walls = s['walls']
        self.lines = s.get('lines', [])
        self.doors = s['doors']
        self.windows = s['windows']
        self.furniture = s['furniture']
        
        self.selection = []
        self._snap_cache_dirty = True
        self.redraw_all()

    def _update_snap_cache(self):
        points = []
        all_polys = self.finalized_polygons + ([self.exterior_polygon] if self.exterior_polygon is not None else [])
        for poly in all_polys:
            if poly is not None: points.extend(poly[:, ::-1])
        for wall in self.walls:
            if wall and wall.get('points'): points.extend([p[::-1] for p in wall['points']])
        for line_seg in self.lines:
            if line_seg and len(line_seg)==2: points.extend([p[::-1] for p in line_seg])
        for obj_list in [self.doors, self.windows]:
            for p1, p2 in obj_list:
                if p1 and p2: points.extend([p1[::-1], p2[::-1]])
        self._snap_points_cache = np.array(points) if points else np.empty((0, 2))
        self._snap_cache_dirty = False

    def _snap_to_targets(self, x, y):
        if not self.snap_enabled: return x, y
        if self._snap_cache_dirty or self._snap_points_cache is None:
            self._update_snap_cache()
        
        if self._snap_points_cache.shape[0] == 0: return x, y
        
        dists = np.sum((self._snap_points_cache - [x, y])**2, axis=1)
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < ai_core.CONFIG["SNAP_DISTANCE_PIXELS"]**2:
            best_x, best_y = self._snap_points_cache[min_idx]
            self._update_temp_artist('snap_point', (best_x, best_y))
            return best_x, best_y
        
        self._clear_temp_artist(); return x, y

    def _find_handle_at_pos(self, pos):
        if self.edit_mode != ai_core.DrawingMode.SELECT or len(self.selection) != 1:
            return None
        
        sel_item = self.selection[0]
        obj = self._get_object_from_selection(sel_item)
        if obj and obj.get('points') and len(obj['points']) == 2:
            p1, p2 = np.array(obj['points'][0]), np.array(obj['points'][1])
            if np.linalg.norm(pos - p1) < ai_core.CONFIG['HANDLE_SIZE_PIXELS']:
                return {'sel_index': 0, 'point_index': 0}
            if np.linalg.norm(pos - p2) < ai_core.CONFIG['HANDLE_SIZE_PIXELS']:
                return {'sel_index': 0, 'point_index': 1}
        return None

    def _convert_and_find_new_wall(self, sel_item_to_convert):
        original_segment = self._get_object_from_selection(sel_item_to_convert)
        if not original_segment: return None
        p1_orig, p2_orig = [tuple(p) for p in original_segment['points']]

        self._convert_generated_poly_to_walls(sel_item_to_convert['poly_index'])

        for i in range(len(self.walls) - 1, -1, -1):
            wall_item = self.walls[i]
            if not wall_item or not wall_item.get('points') or len(wall_item['points']) != 2: continue
            p1_new, p2_new = [tuple(p) for p in wall_item['points']]
            if (p1_new == p1_orig and p2_new == p2_orig) or \
               (p1_new == p2_orig and p2_new == p1_orig):
                return i
        return None

    def _update_drag_previews(self, current_pos):
        self._clear_drag_previews()
        delta = current_pos - self.drag_start_pos
        
        if self.drag_mode == ai_core.DragMode.STRETCH and self.active_handle:
            sel_item = self.selection[self.active_handle['sel_index']]
            obj = self._get_object_from_selection(sel_item) 
            if obj and obj.get('points'):
                other_point_idx = 1 - self.active_handle['point_index']
                p_moving = current_pos
                p_static = obj['points'][other_point_idx]
                
                if self.ortho_enabled and not self.shift_pressed:
                    x_s, y_s = p_static[1], p_static[0]
                    x_c, y_c = p_moving[1], p_moving[0]
                    dx, dy = abs(x_c - x_s), abs(y_c - y_s)
                    if dx > dy: p_moving = [y_s, x_c]
                    else: p_moving = [y_c, x_s]

                p1, p2 = ([0,0], [0,0])
                p1 = p_moving if self.active_handle['point_index'] == 0 else p_static
                p2 = p_moving if self.active_handle['point_index'] == 1 else p_static

                line = mlines.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color=ai_core.CONFIG["PREVIEW_COLOR"], lw=2.5, zorder=25)
                self.ax.add_line(line); self.drag_preview_artists.append(line)

        elif self.drag_mode == ai_core.DragMode.MOVE:
            for i, original_obj in enumerate(self.drag_original_objects):
                if original_obj and original_obj.get('points'):
                    p1_orig, p2_orig = original_obj['points']
                    new_points = [(np.array(p1_orig) + delta).tolist(), (np.array(p2_orig) + delta).tolist()]
                    p1, p2 = new_points
                    line = mlines.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color=ai_core.CONFIG["PREVIEW_COLOR"], lw=2.5, zorder=25)
                    self.ax.add_line(line); self.drag_preview_artists.append(line)
        
        self.fig.canvas.draw_idle()

    def _finalize_drag(self, final_pos):
        delta = final_pos - self.drag_start_pos

        if self.drag_mode == ai_core.DragMode.STRETCH and self.active_handle:
            sel_item = self.selection[self.active_handle['sel_index']]
            obj_ref = self._get_object_from_selection(sel_item, get_ref=True)
            if obj_ref:
                points_list = obj_ref['points'] if isinstance(obj_ref, dict) else obj_ref
                
                other_point_idx = 1 - self.active_handle['point_index']
                p_moving = final_pos.tolist()
                p_static = points_list[other_point_idx]
                
                if self.ortho_enabled and not self.shift_pressed:
                    x_s, y_s = p_static[1], p_static[0]
                    x_c, y_c = p_moving[1], p_moving[0]
                    dx, dy = abs(x_c - x_s), abs(y_c - y_s)
                    if dx > dy: p_moving = [y_s, x_c]
                    else: p_moving = [y_c, x_s]
                
                points_list[self.active_handle['point_index']] = p_moving
                self.app.update_status("Object stretched.")

        elif self.drag_mode == ai_core.DragMode.MOVE:
            for i, sel_item in enumerate(self.selection):
                if i >= len(self.drag_original_objects): continue
                original_obj = self.drag_original_objects[i]
                if not original_obj or 'points' not in original_obj: continue
                
                if sel_item['type'] == 'generated_wall':
                    new_wall_index = self._convert_and_find_new_wall(sel_item)
                    if new_wall_index is not None:
                         self.selection = [] 
                         self.app.update_status("Generated wall converted. Select parts to move.")
                         return
                    continue

                obj_ref = self._get_object_from_selection(sel_item, get_ref=True)
                if obj_ref:
                    points_list = obj_ref['points'] if isinstance(obj_ref, dict) else obj_ref
                    p1_orig, p2_orig = original_obj['points']
                    points_list[0] = (np.array(p1_orig) + delta).tolist()
                    points_list[1] = (np.array(p2_orig) + delta).tolist()
            self.app.update_status(f"Moved {len(self.selection)} item(s).")
            
    def _clear_drag_previews(self):
        for artist in self.drag_preview_artists: artist.remove()
        self.drag_preview_artists = []
        
    def _apply_ortho_constraint(self, start_point, current_point):
        if start_point is None: return current_point
        dx, dy = current_point[0] - start_point[0], current_point[1] - start_point[1]
        return (current_point[0], start_point[1]) if abs(dx) > abs(dy) else (start_point[0], current_point[1])
    def _update_temp_artist(self, artist_type, data):
        self._clear_temp_artist()
        if artist_type == 'snap_point' and self.view_mode == ai_core.ViewMode.VECTOR:
            x, y = data; self.temp_artist = self.ax.plot(x, y, marker='+', color='#FF4500', markersize=16, markeredgewidth=2.5, zorder=20)[0]
            self.fig.canvas.draw_idle()
    def _clear_temp_artist(self):
        if self.temp_artist: self.temp_artist.remove(); self.temp_artist = None; self.fig.canvas.draw_idle()
    def set_view_mode(self, view_mode):
        self.view_mode = view_mode
        self.app.update_view_buttons(view_mode)
        self.app.update_status(f"View: {view_mode.value}")
        self.redraw_all()

# ==================== MAIN APPLICATION (GUI) ====================
class ArchitecturalVectorizerApp:
    def __init__(self, root, sam_model, image_np, raw_masks):
        self.root = root
        self.toolbar_buttons, self.view_buttons = {}, {}
        self.editor, self.base_filename = None, "floorplan"
        self.root.title("Architectural Vectorizer (Professional Edition)")
        style = ttk.Style(self.root); style.theme_use('clam'); style.configure('Toolbutton', padding=3)
        self.load_icons()
        top_bar = ttk.Frame(self.root); top_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        canvas_frame = ttk.Frame(self.root); canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._create_canvas(canvas_frame)
        self.editor = FloorplanEditor(self, sam_model, self.fig, self.ax, image_np, raw_masks, image_np.shape)
        self._create_controls(top_bar)
        self.editor.set_edit_mode(ai_core.DrawingMode.SELECT)

    def load_icons(self):
        self.icons = {}
        icon_names = ["select", "wall", "line", "door", "window", "rectangle", "circle", "arc_wall", "split_wall", "snap", "ortho", "undo", "redo", "segment_edit"]
        for name in icon_names:
            path = os.path.join(ai_core.CONFIG["icon_dir"], f"{name}.png")
            try:
                if os.path.exists(path): self.icons[name] = PhotoImage(file=path)
                else: self.icons[name] = name.replace('_', ' ').title()
            except tk.TclError:
                print(f"Warning: Could not load icon {path}. Using text fallback."); self.icons[name] = name.replace('_', ' ').title()

    def _create_canvas(self, parent_frame):
        self.fig = Figure(figsize=(16, 10), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_controls(self, top_bar):
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export As...", command=self.export_drawing, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-e>", lambda e: self.export_drawing())
        
        toolbar_frame = ttk.Frame(top_bar)
        toolbar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        modes = [("select",ai_core.DrawingMode.SELECT), ("wall",ai_core.DrawingMode.WALL), ("line",ai_core.DrawingMode.LINE), ("arc_wall",ai_core.DrawingMode.ARC_WALL), ("door",ai_core.DrawingMode.DOOR), ("window",ai_core.DrawingMode.WINDOW), ("split_wall",ai_core.DrawingMode.SPLIT_WALL), ("rectangle",ai_core.DrawingMode.RECTANGLE), ("circle",ai_core.DrawingMode.CIRCLE)]
        for name, mode in modes: self._create_toolbar_button(toolbar_frame, name, mode=mode)
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y, pady=4)
        self._create_toolbar_button(toolbar_frame, "segment_edit", command=self.open_segmentation_editor)
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y, pady=4)
        
        settings_frame = ttk.LabelFrame(toolbar_frame, text="Settings")
        settings_frame.pack(side=tk.LEFT, padx=5, fill='y')

        self.snap_button = self._create_toolbar_button(settings_frame, "snap", command=self.editor.toggle_snap)
        self.ortho_button = self._create_toolbar_button(settings_frame, "ortho", command=self.editor.toggle_ortho)
        
        ttk.Label(settings_frame, text="Vectorization:").pack(padx=5, pady=(5,0))
        self.vectorization_mode_var = tk.StringVar(value=self.editor.vectorization_mode.value)
        self.vectorization_mode_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.vectorization_mode_var,
            values=[e.value for e in ai_core.VectorizationMode],
            state="readonly",
            width=35
        )
        self.vectorization_mode_combo.pack(padx=5, pady=2)
        self.vectorization_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.editor.set_vectorization_mode(self.vectorization_mode_var.get()))
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y, pady=4)

        self._create_toolbar_button(toolbar_frame, "undo", command=self.editor.undo)
        self._create_toolbar_button(toolbar_frame, "redo", command=self.editor.redo)
        view_frame = ttk.LabelFrame(top_bar, text="View")
        view_frame.pack(side=tk.LEFT, padx=10, pady=2, fill=tk.Y)
        for view_mode in ai_core.ViewMode:
            btn = ttk.Button(view_frame, text=view_mode.value, command=lambda vm=view_mode: self.editor.set_view_mode(vm))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self.view_buttons[view_mode] = btn

        export_button = ttk.Button(view_frame, text="Export", command=self.export_drawing)
        export_button.pack(side=tk.LEFT, padx=(5, 2), pady=2)

        self.update_snap_button_state(self.editor.snap_enabled)
        self.update_ortho_button_state(self.editor.ortho_enabled)
        self.update_view_buttons(self.editor.view_mode)

    def _create_toolbar_button(self, parent, icon_name, mode=None, command=None):
        if command is None and mode: command = lambda m=mode: self.editor.set_edit_mode(m)
        icon = self.icons.get(icon_name)
        btn_args = {'style': 'Toolbutton', 'command': command}
        if isinstance(icon, str): btn_args.update({'text': icon})
        else: btn_args.update({'image': icon, 'compound': tk.LEFT})
        btn = ttk.Button(parent, **btn_args); btn.pack(side=tk.LEFT, padx=1, pady=1)
        if mode: self.toolbar_buttons[mode] = btn
        return btn

    def update_button_states(self, active_mode):
        for mode, button in self.toolbar_buttons.items(): button.state(['pressed'] if mode == active_mode else ['!pressed'])
    def update_snap_button_state(self, is_on): self.snap_button.state(['pressed' if is_on else '!pressed'])
    def update_ortho_button_state(self, is_on): self.ortho_button.state(['pressed' if is_on else '!pressed'])
    
    def update_view_buttons(self, active_view):
        for view_mode, button in self.view_buttons.items(): button.state(['pressed'] if view_mode == active_view else ['!pressed'])
        
    def open_segmentation_editor(self): self.update_status("Opening segmentation editor..."); editor_window = SegmentationEditor(self.root, self.editor.raw_masks, self.editor.original_img, self.on_segmentation_close)
    def on_segmentation_close(self, edited_masks): self.update_status("Applying changes..."); self.editor.reprocess_from_masks(new_masks=edited_masks)
    def update_status(self, message): self.status_bar.config(text=message); self.root.update_idletasks()
    
    def export_drawing(self):
        """Opens a save dialog and exports the drawing to the chosen format."""
        if not self.editor:
            messagebox.showerror("Export Error", "Editor not initialized.", parent=self.root)
            return

        file_types = [
            ('DXF Drawing', '*.dxf'),
            ('SVG Vector Image', '*.svg'),
            ('PDF Document', '*.pdf'),
            ('JPEG Image', '*.jpg'),
            ('FML XML File', '*.fml'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Export Drawing",
            initialfile=f"{self.base_filename}_vectorized",
            defaultextension=".dxf",
            filetypes=file_types,
            parent=self.root
        )

        if not filepath:
            self.update_status("Export cancelled.")
            return

        self.update_status(f"Exporting to {filepath}...")
        
        _, extension = os.path.splitext(filepath)
        extension = extension.lower()

        try:
            if extension == '.dxf':
                self._export_as_dxf(filepath)
            elif extension == '.svg':
                self._export_as_svg(filepath)
            elif extension == '.pdf':
                self._export_as_pdf(filepath)
            elif extension in ['.jpg', '.jpeg']:
                self._export_as_jpeg(filepath)
            elif extension == '.fml':
                self._export_as_fml(filepath)
            else:
                messagebox.showwarning("Export Warning", f"Unsupported file extension '{extension}'. Please choose a supported type.", parent=self.root)
                self.update_status(f"Export failed: unsupported format '{extension}'.")
                return
            
            self.update_status(f"Successfully exported to {filepath}")
            messagebox.showinfo("Export Successful", f"Drawing saved to:\n{filepath}", parent=self.root)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"An error occurred during export:\n\n{e}", parent=self.root)

    def _export_as_jpeg(self, filepath):
        """Exports the current vector view as a JPEG image."""
        output_image_pil = Image.fromarray(self.editor.output_image)
        output_image_pil.save(filepath, 'JPEG', quality=95)

    def _export_as_pdf(self, filepath):
        """Exports the current vector view as a raster image embedded in a PDF."""
        output_image_pil = Image.fromarray(self.editor.output_image).convert("RGB")
        output_image_pil.save(filepath, 'PDF', resolution=100.0)

    def _export_as_fml(self, filepath):
        """Exports the drawing data to a custom FML (Floorplan Markup Language) XML file."""
        root = ai_core.ET.Element("floorplan", {
            "width": str(self.editor.image_shape[1]),
            "height": str(self.editor.image_shape[0]),
            "scale_pixels_to_meters": str(ai_core.CONFIG['dxf_scale'])
        })

        def add_points_to_element(parent, points):
            if not points: return
            for p in points:
                if p is not None:
                    ai_core.ET.SubElement(parent, "point", {"y": str(p[0]), "x": str(p[1])})

        all_polys = self.editor.finalized_polygons + ([self.editor.exterior_polygon] if self.editor.exterior_polygon is not None else [])
        gen_walls_node = ai_core.ET.SubElement(root, "generated_walls")
        for poly in all_polys:
            if poly is not None:
                poly_node = ai_core.ET.SubElement(gen_walls_node, "polygon")
                add_points_to_element(poly_node, poly)
        
        walls_node = ai_core.ET.SubElement(root, "manual_walls")
        for wall in self.editor.walls:
            if wall and wall.get('points'):
                wall_node = ai_core.ET.SubElement(walls_node, "wall", {"thickness": str(wall.get('thickness', ai_core.CONFIG['WALL_THICKNESS_PIXELS']))})
                add_points_to_element(wall_node, wall['points'])

        for name, collection in [("lines", self.editor.lines), ("doors", self.editor.doors), ("windows", self.editor.windows)]:
            node = ai_core.ET.SubElement(root, name)
            for item in collection:
                if item and len(item) == 2:
                    item_node = ai_core.ET.SubElement(node, name[:-1])
                    add_points_to_element(item_node, item)

        xml_str = ai_core.ET.tostring(root, 'utf-8')
        pretty_xml_str = ai_core.minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(pretty_xml_str)

    def _export_as_dxf(self, filepath):
        """Exports the drawing to a robust DXF file with hatched walls."""
        doc = ai_core.ezdxf.new()
        msp = doc.modelspace()
        
        H, W = self.editor.image_shape[:2]
        S = ai_core.CONFIG['dxf_scale']
        doc.header['$INSUNITS'] = 6  # 6 = Meters

        # Define layers, adding one for hatches
        layers = {
            "WALLS": ai_core.const.CYAN,
            "WALL_HATCH": 8,  # A standard grey color for hatches
            "DOORS": ai_core.const.RED,
            "WINDOWS": ai_core.const.GREEN,
            "LINES": ai_core.const.YELLOW
        }
        for name, color in layers.items():
            doc.layers.add(name=name, color=color)

        # --- GEOMETRY GENERATION (Mirrors the visualizer) ---

        # 1. Collect all wall centerlines (from auto-detection and manual edits)
        all_wall_geoms = []
        
        # Add auto-detected walls
        all_polys = self.editor.finalized_polygons + ([self.editor.exterior_polygon] if self.editor.exterior_polygon is not None else [])
        for poly in all_polys:
            if poly is not None and len(poly) > 1:
                for i in range(len(poly) - 1):
                    p1, p2 = poly[i], poly[i+1]
                    line = ai_core.LineString([p1, p2])
                    all_wall_geoms.append({'line': line, 'thickness': ai_core.CONFIG['WALL_THICKNESS_PIXELS']})
        
        # Add manually drawn walls
        for wall in self.editor.walls:
            if wall and wall.get('points') and len(wall['points']) == 2:
                p1, p2 = wall['points']
                thickness = wall.get('thickness', ai_core.CONFIG['WALL_THICKNESS_PIXELS'])
                line = ai_core.LineString([p1, p2])
                all_wall_geoms.append({'line': line, 'thickness': thickness})

        # 2. Buffer each line by its thickness and unify into a single wall shape
        if not all_wall_geoms:
            print("No wall geometry to export.")
            # Still export other elements even if there are no walls
        else:
            wall_polygons = [geom['line'].buffer(geom['thickness'] / 2, cap_style=2, join_style=2) for geom in all_wall_geoms]
            wall_shape = ai_core.unary_union(wall_polygons)
            
            # 3. Cut out openings (doors and windows)
            for opening in self.editor.windows + self.editor.doors:
                if not opening or len(opening) != 2: continue
                p1_open, p2_open = opening
                opening_line = ai_core.LineString([p1_open, p2_open])
                # Buffer slightly larger to ensure a clean cut
                cutout_poly = opening_line.buffer(ai_core.CONFIG['WALL_THICKNESS_PIXELS'] / 2 + 2, cap_style=3)
                if wall_shape.is_valid:
                    wall_shape = wall_shape.difference(cutout_poly)

            # 4. Add the final wall geometry to the DXF modelspace
            if wall_shape and not wall_shape.is_empty:
                geoms = wall_shape.geoms if hasattr(wall_shape, 'geoms') else [wall_shape]
                for p in geoms:
                    if p.is_empty or not isinstance(p, ai_core.Polygon): continue
                    
                    # Add the hatch pattern for the wall fill
                    hatch = msp.add_hatch(dxfattribs={"layer": "WALL_HATCH"})
                    hatch.set_pattern_fill(
                        "ANSI31",  # Standard diagonal hatch pattern
                        scale=ai_core.CONFIG['HATCH_SPACING'] * S * 4,  # Scale the pattern
                        angle=ai_core.CONFIG['HATCH_ANGLE']
                    )
                    
                    # Add the exterior outline as a boundary for the hatch and as a visible line
                    exterior_points = [ai_core._transform_coords_for_dxf(point, H, S) for point in p.exterior.coords]
                    msp.add_lwpolyline(exterior_points, close=True, dxfattribs={"layer": "WALLS"})
                    hatch.paths.add_polyline_path(exterior_points, is_closed=True)

                    # Add any interior holes as boundaries
                    for interior in p.interiors:
                        interior_points = [ai_core._transform_coords_for_dxf(point, H, S) for point in interior.coords]
                        msp.add_lwpolyline(interior_points, close=True, dxfattribs={"layer": "WALLS"})
                        hatch.paths.add_polyline_path(interior_points, is_closed=True)

        # --- Process other manually drawn elements (doors, windows, lines) ---
        for door in self.editor.doors:
            if door and len(door) == 2:
                ai_core._add_door_to_dxf(msp, door[0], door[1], ai_core.CONFIG['WALL_THICKNESS_PIXELS'], H, S)

        for window in self.editor.windows:
            if window and len(window) == 2:
                ai_core._add_window_to_dxf(msp, window[0], window[1], ai_core.CONFIG['WALL_THICKNESS_PIXELS'], H, S)

        for line_seg in self.editor.lines:
            if line_seg and len(line_seg) == 2:
                p1_transformed = ai_core._transform_coords_for_dxf(line_seg[0], H, S)
                p2_transformed = ai_core._transform_coords_for_dxf(line_seg[1], H, S)
                msp.add_line(p1_transformed, p2_transformed, dxfattribs={"layer": "LINES"})
        
        try:
            doc.saveas(filepath)
        except IOError:
            messagebox.showerror("Export Error", f"Could not write to file:\n{filepath}\n\nPlease check permissions.", parent=self.root)

    def _export_as_svg(self, filepath):
        """Exports the drawing to an SVG file, recreating the vector view."""
        W, H = self.editor.image_shape[1], self.editor.image_shape[0]
        
        svg_elements = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
        svg_elements.append(f'  <rect width="100%" height="100%" fill="rgb({",".join(map(str, ai_core.CONFIG["BACKGROUND_COLOR"]))})"/>')

        all_wall_lines = []
        all_polys = self.editor.finalized_polygons + ([self.editor.exterior_polygon] if self.editor.exterior_polygon is not None else [])
        for poly in all_polys:
            if poly is not None and len(poly) > 1:
                for i in range(len(poly) - 1):
                    p1, p2 = poly[i], poly[i+1]
                    all_wall_lines.append(ai_core.LineString([(p1[1], p1[0]), (p2[1], p2[0])]))

        for wall in self.editor.walls:
            if not wall or not wall.get('points') or len(wall['points']) != 2: continue
            p1, p2 = wall['points']
            all_wall_lines.append(ai_core.LineString([(p1[1], p1[0]), (p2[1], p2[0])]))

        if all_wall_lines:
            wall_shape = ai_core.unary_union([line.buffer(ai_core.CONFIG['WALL_THICKNESS_PIXELS'] / 2, cap_style=2, join_style=2) for line in all_wall_lines])
            
            for opening in self.editor.windows + self.editor.doors:
                if not opening or len(opening) != 2: continue
                p1_open, p2_open = opening
                opening_line = ai_core.LineString([(p1_open[1], p1_open[0]), (p2_open[1], p2_open[0])])
                cutout_poly = opening_line.buffer(ai_core.CONFIG['WALL_THICKNESS_PIXELS'] / 2 + 2, cap_style=3)
                wall_shape = wall_shape.difference(cutout_poly)
            
            if wall_shape and not wall_shape.is_empty:
                path_data = []
                geoms = wall_shape.geoms if hasattr(wall_shape, 'geoms') else [wall_shape]
                for p in geoms:
                    if not p.is_empty and isinstance(p, ai_core.Polygon):
                        path_data.append(f"M {' '.join(f'{x},{y}' for x,y in p.exterior.coords)} Z")
                        for interior in p.interiors:
                           path_data.append(f"M {' '.join(f'{x},{y}' for x,y in interior.coords)} Z")
                
                hatch_id, wall_id = "hatch_pattern", "wall_shape"
                svg_elements.append('<defs>')
                svg_elements.append(f'<pattern id="{hatch_id}" patternUnits="userSpaceOnUse" width="{ai_core.CONFIG["HATCH_SPACING"]*2}" height="{ai_core.CONFIG["HATCH_SPACING"]*2}" patternTransform="rotate({ai_core.CONFIG["HATCH_ANGLE"]})">')
                svg_elements.append(f'  <path d="M 0,0 v {ai_core.CONFIG["HATCH_SPACING"]*2}" style="stroke:rgb({",".join(map(str, ai_core.CONFIG["WALL_HATCH_COLOR"]))}); stroke-width:1"/>')
                svg_elements.append('</pattern>')
                svg_elements.append(f'<clipPath id="{wall_id}"><path d="{" ".join(path_data)}" fill-rule="evenodd" /></clipPath>')
                svg_elements.append('</defs>')
                svg_elements.append(f'<rect x="0" y="0" width="100%" height="100%" fill="url(#{hatch_id})" clip-path="url(#{wall_id})"/>')
                svg_elements.append(f'<path d="{" ".join(path_data)}" fill="none" stroke="rgb({",".join(map(str, ai_core.CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')
        
        for segment in self.editor.windows:
            if not segment or len(segment) != 2: continue
            p1, p2 = segment
            ai_core.draw_window_symbol_svg(svg_elements, p1, p2, ai_core.CONFIG['WALL_THICKNESS_PIXELS'])
        for segment in self.editor.doors:
            if not segment or len(segment) != 2: continue
            p1, p2 = segment
            ai_core.draw_door_symbol_svg(svg_elements, p1, p2, ai_core.CONFIG['WALL_THICKNESS_PIXELS'], H, W)
        for segment in self.editor.lines:
            if not segment or len(segment) != 2: continue
            p1, p2 = segment
            svg_elements.append(f'  <line x1="{p1[1]}" y1="{p1[0]}" x2="{p2[1]}" y2="{p2[0]}" stroke="rgb({",".join(map(str, ai_core.CONFIG["WALL_COLOR"]))})" stroke-width="1"/>')

        svg_elements.append('</svg>')
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("\n".join(svg_elements))

# ==================== MAIN EXECUTION ====================
def main():
    root = tk.Tk()
    root.withdraw()
    try:
        # Get the directory where the script is located
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: # __file__ is not defined in interactive environments
            script_dir = os.getcwd()
            
        if "weights_path" in ai_core.CONFIG and not os.path.isabs(ai_core.CONFIG["weights_path"]):
             ai_core.CONFIG["weights_path"] = os.path.join(script_dir, os.path.basename(ai_core.CONFIG["weights_path"]))
        if "icon_dir" in ai_core.CONFIG and not os.path.isabs(ai_core.CONFIG["icon_dir"]):
             ai_core.CONFIG["icon_dir"] = os.path.join(script_dir, "icons")
        
        sam_model = ai_core.load_model_secure()
        image_path = select_input_file(root)
        if not image_path:
            root.destroy()
            return
        os.makedirs(ai_core.CONFIG["output_dir"], exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image_np = ai_core.process_image(image)
        print("🤖 Generating initial segmentation... (this may take a moment)")
        mask_generator = ai_core.SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=ai_core.CONFIG['points_per_side'],
            pred_iou_thresh=ai_core.CONFIG['pred_iou_thresh'],
            stability_score_thresh=ai_core.CONFIG['stability_score_thresh'],
            min_mask_region_area=ai_core.CONFIG['min_mask_region_area']
        )
        raw_masks = mask_generator.generate(image_np)
        print(f"✅ Found {len(raw_masks)} initial segments.")
        root.deiconify()
        app = ArchitecturalVectorizerApp(root, sam_model, image_np, raw_masks)
        app.base_filename = os.path.splitext(os.path.basename(image_path))[0]
        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        if not root.winfo_viewable():
            root.deiconify()
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n\n{e}")
    finally:
        if root.winfo_exists():
            root.destroy()

if __name__ == "__main__":
    main()