# Vector-AI Floorplan Converter

An AI-powered application to convert floorplan images into vector drawings (DXF, SVG). This project uses the Segment Anything Model (SAM) to identify rooms and walls.

## Features

*   Automatic room and wall segmentation from image files.
*   Interactive editor to refine walls, add doors, and add windows.
*   Export to multiple formats, including DXF for CAD software.

## How to Run

1.  Make sure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python app.py
    ```