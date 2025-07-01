# app.py

import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import ai_core 

# --- FLASK APP INITIALIZATION ---
# The 'static_folder' points to the 'static' directory where your CSS/JS for the template will be.
# The 'template_folder' points to where index.html is.
app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- LOAD THE AI MODEL ONCE AT STARTUP ---
print("⚙️ Loading SAM model for the web server...")
try:
    SAM_MODEL = ai_core.load_model_secure()
    MASK_GENERATOR = ai_core.SamAutomaticMaskGenerator(
        model=SAM_MODEL,
        points_per_side=ai_core.CONFIG.get('points_per_side', 32),
        pred_iou_thresh=ai_core.CONFIG.get('pred_iou_thresh', 0.88),
        stability_score_thresh=ai_core.CONFIG.get('stability_score_thresh', 0.95),
        min_mask_region_area=ai_core.CONFIG.get('min_mask_region_area', 100)
    )
    print("✅ SAM Model is ready to serve requests.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load the AI model. The server cannot start.")
    print(f"   Error details: {e}")
    # Exit if the model can't be loaded, as the app is useless without it.
    exit()


# --- WEB & API ENDPOINTS ---

@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    Receives an uploaded image, saves it, runs segmentation, 
    and returns the masks and contours to the client.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image for the model
        image = Image.open(filepath).convert("RGB")
        image_np = ai_core.process_image(image)

        print(f"🤖 Segmenting image: {filename}...")
        raw_masks = MASK_GENERATOR.generate(image_np)
        print(f"✅ Segmentation complete. Found {len(raw_masks)} masks.")
        
        # Prepare data for JSON response
        for mask in raw_masks:
            # The contour is useful for drawing the mask outlines on the frontend canvas
            segmentation_np = np.array(mask['segmentation'])
            contours = ai_core.measure.find_contours(segmentation_np, 0.5)
            # Get the largest contour and flatten it for easier processing in JS
            mask['contour'] = max(contours, key=len)[:, [1, 0]].ravel().tolist() if contours else []
            # Convert the boolean mask to a list of lists for JSON
            mask['segmentation'] = mask['segmentation'].tolist()

        return jsonify({
            'masks': raw_masks,
            'image_shape': image_np.shape
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred during segmentation: {str(e)}'}), 500


@app.route('/api/vectorize', methods=['POST'])
def vectorize_masks():
    """
    Receives segmentation masks and vectorization settings, then calls the
    centralized 'vectorize_floorplan' function from ai_core to get the final geometry.
    """
    data = request.json
    raw_masks_json = data.get('masks')
    image_shape = data.get('image_shape')
    vectorization_mode_str = data.get('mode', 'Idealize (Orthogonal)')

    if not raw_masks_json or not image_shape:
        return jsonify({'error': 'Missing required data: masks or image_shape'}), 400

    try:
        # Reconstruct the raw_masks list with numpy arrays from the JSON data
        raw_masks = raw_masks_json
        for mask in raw_masks:
            mask['segmentation'] = np.array(mask['segmentation'], dtype=bool)

        print(f"📐 Vectorizing {len(raw_masks)} masks with mode: '{vectorization_mode_str}'...")
        
        # Convert the string mode from the client into the proper Enum
        vectorization_mode = ai_core.VectorizationMode(vectorization_mode_str)

        # === THIS IS THE CORRECTED LOGIC ===
        # Call the single, authoritative function from ai_core to do all the work.
        finalized_polygons, exterior_polygon = ai_core.vectorize_floorplan(
            raw_masks,
            tuple(image_shape),  # Ensure shape is a tuple
            vectorization_mode
        )
        # ==================================
        
        # Convert numpy arrays to simple lists for JSON serialization
        final_polygons_list = [poly.tolist() for poly in finalized_polygons if poly is not None]
        exterior_polygon_list = exterior_polygon.tolist() if exterior_polygon is not None else None

        print(f"✅ Vectorization complete. Returning {len(final_polygons_list)} room polygons and an exterior outline.")
        
        return jsonify({
            'finalized_polygons': final_polygons_list,
            'exterior_polygon': exterior_polygon_list
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred during vectorization: {str(e)}'}), 500


# To run this server, use a production-ready WSGI server like Gunicorn.
# Example: gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
# For development, you can use the Flask development server:
if __name__ == '__main__':
    # Use debug=False if deploying, or debug=True for development ONLY.
    # host='0.0.0.0' makes it accessible from other devices on the same network.
    app.run(debug=True, host='0.0.0.0', port=5000)