# app.py

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import ai_core 

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- LOAD THE AI MODEL ONCE ---
print("⚙️ Loading SAM model for the web server...")
SAM_MODEL = ai_core.load_model_secure()
MASK_GENERATOR = ai_core.SamAutomaticMaskGenerator(
    model=SAM_MODEL,
    points_per_side=ai_core.CONFIG['points_per_side'],
    pred_iou_thresh=ai_core.CONFIG['pred_iou_thresh'],
    stability_score_thresh=ai_core.CONFIG['stability_score_thresh'],
    min_mask_region_area=ai_core.CONFIG['min_mask_region_area']
)
print("✅ SAM Model is ready to serve requests.")


# --- API ENDPOINTS ---

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/api/segment', methods=['POST'])
def segment_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image_np = ai_core.process_image(image)

        print(f"🤖 Segmenting image: {filename}...")
        raw_masks = MASK_GENERATOR.generate(image_np)
        print(f"✅ Segmentation complete. Found {len(raw_masks)} masks.")
        
        # Prepare data for front-end
        for mask in raw_masks:
            segmentation_np = np.array(mask['segmentation'])
            contours = ai_core.measure.find_contours(segmentation_np, 0.5)
            mask['contour'] = max(contours, key=len)[:, [1, 0]].ravel().tolist() if contours else []
            mask['segmentation'] = mask['segmentation'].tolist()

        # Also send back image shape for the vectorize step
        return jsonify({'masks': raw_masks, 'image_shape': image_np.shape})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/vectorize', methods=['POST'])
def vectorize_masks():
    data = request.json
    raw_masks = data.get('masks')
    image_shape = data.get('image_shape')
    vectorization_mode_str = data.get('mode', 'Idealize (Orthogonal)')

    if not raw_masks or not image_shape:
        return jsonify({'error': 'Missing masks or image_shape'}), 400

    try:
        for mask in raw_masks:
            mask['segmentation'] = np.array(mask['segmentation'], dtype=bool)

        print(f"📐 Vectorizing {len(raw_masks)} masks with mode: {vectorization_mode_str}...")
        
        vectorization_mode = ai_core.VectorizationMode(vectorization_mode_str)
        final_room_masks = ai_core.filter_and_get_rooms(raw_masks)
        
        # --- Faithful Recreation of Desktop App Logic ---
        
        grid_y, grid_x = ai_core.calculate_global_orthogonal_grid(final_room_masks, image_shape)
        
        all_finalized_polygons = []
        for m in final_room_masks:
            contours = ai_core.measure.find_contours(m['segmentation'], 0.5)
            if not contours: continue
            
            largest_contour = max(contours, key=len)
            simplified = ai_core.process_contour(largest_contour)
            if simplified.size < 3: continue
            
            # This is the _idealize_contour logic ported directly
            if grid_y.size > 1 and grid_x.size > 1:
                idealized_pts_raw = [[grid_y[np.argmin(np.abs(grid_y - pt[0]))], 
                                      grid_x[np.argmin(np.abs(grid_x - pt[1]))]] 
                                     for pt in simplified]
                
                final_path = []
                if idealized_pts_raw:
                    final_path.append(np.array(idealized_pts_raw[0]))
                    for i in range(1, len(idealized_pts_raw)):
                        next_pt, prev_pt = np.array(idealized_pts_raw[i]), final_path[-1]
                        if np.allclose(next_pt, prev_pt): continue
                        
                        delta_y, delta_x = abs(next_pt[0] - prev_pt[0]), abs(next_pt[1] - prev_pt[1])
                        if delta_y > 1e-3 and delta_x > 1e-3:
                            if delta_x > delta_y:
                                final_path.append(np.array([prev_pt[0], next_pt[1]]))
                            else:
                                final_path.append(np.array([next_pt[0], prev_pt[1]]))
                        
                        if not np.allclose(final_path[-1], next_pt):
                            final_path.append(next_pt)

                unique_path = [final_path[0]] if final_path else []
                for i in range(1, len(final_path)):
                    if np.linalg.norm(final_path[i] - unique_path[-1]) > 1.0:
                        unique_path.append(final_path[i])
                
                processed_contour = np.array(unique_path)
            else:
                processed_contour = ai_core.measure.approximate_polygon(simplified, tolerance=ai_core.CONFIG["RDP_TOLERANCE"])

            if processed_contour is not None and len(processed_contour) > 2:
                poly = ai_core.Polygon(processed_contour)
                if not poly.is_valid: poly = poly.buffer(0)
                if not poly.is_empty:
                    all_finalized_polygons.append(np.array(poly.exterior.coords))
        
        final_polygons_list = [poly.tolist() for poly in all_finalized_polygons if poly is not None]

        print("✅ Vectorization complete with full idealization.")
        
        return jsonify({'finalized_polygons': final_polygons_list})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')