// static/main.js

window.onload = function() {
    
    // --- Global Variables ---
    let stage; // The main canvas stage from Konva
    let imageLayer; // Layer for the background image
    let maskLayer; // Layer for the raw, "wobbly" outlines
    let vectorLayer; // Layer for the clean, vectorized lines

    // We store data between API calls
    let rawMasksData = []; 
    let originalImageDimensions = { width: 0, height: 0 };

    // Get references to our HTML elements
    const uploadButton = document.getElementById('uploadButton');
    const vectorizeButton = document.getElementById('vectorizeButton');
    const imageFileInput = document.getElementById('imageFile');
    const statusElem = document.getElementById('status');
    
    // --- Event Listeners ---
    uploadButton.addEventListener('click', uploadAndProcessImage);
    vectorizeButton.addEventListener('click', vectorizeSegmentations);

    // --- Core Functions ---

    // Initializes or resets the Konva stage
    function initializeStage(width, height) {
        if (stage) {
            stage.destroy();
        }
        stage = new Konva.Stage({
            container: 'container',
            width: width,
            height: height,
        });

        imageLayer = new Konva.Layer();
        maskLayer = new Konva.Layer();
        vectorLayer = new Konva.Layer();

        stage.add(imageLayer, maskLayer, vectorLayer);
    }

    // Handles the initial image upload and call to /api/segment
async function uploadAndProcessImage() {
    if (imageFileInput.files.length === 0) {
        alert('Please select an image file first!');
        return;
    }
    const file = imageFileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    statusElem.textContent = 'Uploading and processing... This can take a moment.';
    vectorizeButton.style.display = 'none';
    uploadButton.disabled = true;

    try {
        const response = await fetch('/api/segment', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json(); // result is now an object: {masks: [], image_shape: []}

        if (response.ok) {
            statusElem.textContent = `Success! Received ${result.masks.length} masks. Drawing...`;
            rawMasksData = result.masks; // Store only the masks array
            const imageURL = URL.createObjectURL(file);
            drawDataOnCanvas(imageURL, result.masks); 
        } else {
            statusElem.textContent = `An error occurred: ${result.error}`;
            console.error(result);
        }
    } catch (error) {
        statusElem.textContent = 'A network error occurred. Check the console.';
        console.error(error);
    } finally {
        uploadButton.disabled = false;
    }
}
    
    // Handles the call to our new /api/vectorize endpoint
    async function vectorizeSegmentations() {
        if (rawMasksData.length === 0) {
            alert("Please segment an image first.");
            return;
        }

        statusElem.textContent = 'Vectorizing outlines...';
        vectorizeButton.disabled = true;

        try {
            const response = await fetch('/api/vectorize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    masks: rawMasksData,
                    image_shape: [originalImageDimensions.height, originalImageDimensions.width],
                    mode: 'Idealize (Orthogonal)'
                }),
            });

            const result = await response.json();

            if (response.ok) {
                statusElem.textContent = 'Vectorization complete.';
                maskLayer.hide(); // Hide the old "wobbly" outlines
                drawVectorizedOutlines(result); // Draw the new clean vectors
            } else {
                statusElem.textContent = `Vectorization failed: ${result.error}`;
            }
        } catch (error) {
            statusElem.textContent = `Network error during vectorization: ${error}`;
        } finally {
            vectorizeButton.disabled = false;
        }
    }

    // This function draws the initial "wobbly" outlines from the segmentation
    function drawDataOnCanvas(imageURL, masks) {
        const imageObj = new Image();
        imageObj.onload = function() {
            // Store dimensions for the vectorize step
            originalImageDimensions.width = this.width;
            originalImageDimensions.height = this.height;

            // Reset canvas to the new image size
            initializeStage(this.width, this.height);
            imageLayer.add(new Konva.Image({ image: imageObj, width: this.width, height: this.height }));
            
            maskLayer.destroyChildren();
            vectorLayer.destroyChildren();
            maskLayer.show();

            masks.forEach(maskData => {
                if (!maskData.contour || maskData.contour.length === 0) return;
                
                const outline = new Konva.Line({
                    points: maskData.contour,
                    stroke: '#' + Math.floor(Math.random()*16777215).toString(16),
                    strokeWidth: 2,
                    closed: true,
                    opacity: 0.7
                });
                maskLayer.add(outline);
            });

            statusElem.textContent = 'Segmentation complete. Ready to vectorize.';
            vectorizeButton.style.display = 'inline-block'; // Show the vectorize button
        };
        imageObj.src = imageURL;
    }

    // This function draws the final, clean, vectorized lines
    function drawVectorizedOutlines(vectorData) {
        vectorLayer.destroyChildren(); // Clear any previous vectors

        const drawPolygon = (points, color, width) => {
            if (!points || points.length === 0) return;
            const flatPoints = points.flat();
            const line = new Konva.Line({
                points: flatPoints,
                stroke: color,
                strokeWidth: width,
                closed: true,
                lineCap: 'round',
                lineJoin: 'round',
            });
            vectorLayer.add(line);
        };

        if (vectorData.finalized_polygons) {
            vectorData.finalized_polygons.forEach(polygon => {
                drawPolygon(polygon, '#000000', 2); // Black, 2px thick
            });
        }
        if (vectorData.exterior_polygon) {
            drawPolygon(vectorData.exterior_polygon, '#0000FF', 3); // Blue, 3px thick
        }
    }
    
    // Initialize with a placeholder stage when the page first loads
    initializeStage(600, 400); 
};