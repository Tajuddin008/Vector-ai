# app.py - Our New Web Server

from flask import Flask, request, jsonify
import ai_core  # Import our new logic file

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD THE AI MODEL ONCE ---
# This is very important! We load the model when the server starts,
# so we don't have to reload it for every user request.
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


# --- DEFINE OUR API ENDPOINTS (THE WEB URLS) ---

@app.route("/")
def hello_world():
    # This is a simple test to make sure our server is running.
    return "<p>Hello, the Vector-AI server is running!</p>"

# We will add more endpoints here later, like '/api/segment'

# This line allows us to run the server directly for testing
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')