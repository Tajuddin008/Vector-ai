# app.py

import os
# Add render_template to this line
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Import our new logic file
import ai_core 

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)

# It's good practice to have a temporary upload folder
# This will be created inside your Vector-AI-Project folder
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


# --- DEFINE OUR API ENDPOINTS ---


@app.route("/")
def hello_world():
    # Instead of showing text, now we will show our HTML page
    return render_template('index.html')

# ---- NEW API ENDPOINT FOR IMAGE SEGMENTATION ----
@app.route('/api/segment', methods=['POST'])
def segment_image():
    # 1. Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # 2. Save the uploaded file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 3. Use your ai_core functions to process the image
        image = Image.open(filepath).convert("RGB")
        image_np = ai_core.process_image(image)

        # 4. Run the AI segmentation
        print(f"🤖 Segmenting image: {filename}...")
        raw_masks = MASK_GENERATOR.generate(image_np)
        print(f"✅ Segmentation complete. Found {len(raw_masks)} masks.")

        # 5. Prepare the data for sending back as JSON
        # We can't send numpy arrays directly, so we convert the segmentation mask to a simple list.
        for mask in raw_masks:
            if 'segmentation' in mask:
                mask['segmentation'] = mask['segmentation'].tolist()

        return jsonify(raw_masks)

    except Exception as e:
        # If anything goes wrong, send back an error message
        return jsonify({'error': str(e)}), 500


# This line allows us to run the server directly for testing
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')