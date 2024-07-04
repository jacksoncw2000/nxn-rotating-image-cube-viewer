# nxn-rotating-image-cube-viewer

A viewer for displaying a scalable number of rotating image cubes.

## Setup Instructions

1. Ensure you have Python 3.7 or later installed on your system.

2. Clone this repository or download the source code.

3. Open a terminal/command prompt and navigate to the project directory.

4. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

5. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

6. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Preparing Your Images

1. Create a main folder for your project.

2. Inside this folder, create subfolders named `source_images_1`, `source_images_2`, etc., up to the number of cubes you want to display (e.g., for a 2x2 grid, create 4 subfolders).

3. Place your PNG images in these subfolders. Each subfolder should contain at least 2 images.

## Running the Program

1. Open the `nxn_rotating_image_cube_viewer.py` file in a text editor.

2. Adjust the constants at the top of the file as needed:
   - `GRID_SIZE`: Number of cubes in each row/column
   - `VIEW_DISTANCE`: Distance of the camera from the cubes
   - `CUBE_SPACING`: Space between cubes
   - `SIDE_DISPLAY_TIME`: How long to display each side before rotating
   - `ROTATION_SPEED`: Speed of cube rotation

3. Save your changes.

4. Run the program:
   ```
   python nxn_rotating_image_cube_viewer.py
   ```

5. When prompted, select the main folder containing your `source_images_X` subfolders.

6. The program will display the rotating image cubes. Close the window to exit.

## Troubleshooting

- If you encounter any "module not found" errors, ensure you've activated your virtual environment and installed all requirements.
- If no images are displayed, check that your folder structure is correct and that you have at least 2 PNG images in each `source_images_X` subfolder.
- For performance issues, try reducing the `GRID_SIZE` or increasing the `VIEW_DISTANCE`.

## Recommended Inputs
* 1x1 Cube
    * GRID_SIZE = 1
    * VIEW_DISTANCE = 10
    * CUBE_SPACING = N/A
    * SIDE_DISPLAY_TIME = 0
    * ROTATION_SPEED = 1.0
* 2x2 Cubes with space
    * GRID_SIZE = 2
    * VIEW_DISTANCE = 16
    * CUBE_SPACING = 1.5
    * SIDE_DISPLAY_TIME = 0
    * ROTATION_SPEED = 1.0
* 3x3 Cubes with no space
    * GRID_SIZE = 3
    * VIEW_DISTANCE = 16
    * CUBE_SPACING = 2
    * SIDE_DISPLAY_TIME = 0
    * ROTATION_SPEED = 1.0
    