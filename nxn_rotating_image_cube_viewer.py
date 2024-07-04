import os
import time
import glob
import random
import math
import tkinter as tk
from tkinter import filedialog
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Constants and configuration
GRID_SIZE = 1  # Change this to 1, 2, 3, 4, etc. for different grid sizes
VIEW_DISTANCE = 10  # The higher the number, the farther out the camera view will be from the cubes
CUBE_SPACING = 2  # The higher the number, the more space between cubes
SIDE_DISPLAY_TIME = 0  # How long to stay on each side (in seconds)
ROTATION_SPEED = 1.0  # Adjust this value to control the speed of rotation (degrees per second)

def select_folder():
    """
    Desc:
        Opens a file dialog to allow the user to select a folder.
        The last selected folder is saved and opened the next time.
    Args:
        None
    Returns:
        str: The path of the selected folder.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Define the absolute path to the last_folder.txt file
    last_folder_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_folder.txt")
    #print(f"[INFO] last_folder.txt is being saved to: {last_folder_file}")

    # Read the last selected folder from a file
    last_folder_path = ""
    if os.path.exists(last_folder_file):
        with open(last_folder_file, "r") as file:
            last_folder_path = file.read().strip()

    # Open the directory chooser with the last selected folder
    folder_path = filedialog.askdirectory(initialdir=last_folder_path)

    # Save the selected folder path to a file
    with open(last_folder_file, "w") as file:
        file.write(folder_path)

    return folder_path

# Ensure directory exists
def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Pygame and OpenGL initialization
def init_pygame_opengl():
    pygame.init()
    infoObject = pygame.display.Info()
    display = (infoObject.current_w, infoObject.current_h)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glViewport(0, 0, display[0], display[1])
    gluPerspective(30, (display[0] / display[1]), 0.1, 50.0)
    glEnable(GL_DEPTH_TEST)
    camera_z = -(VIEW_DISTANCE + GRID_SIZE * 2.5)
    glTranslatef(0.0, 0.0, camera_z)

# Texture loading and management
def load_texture(img_path):
    texture_surface = pygame.image.load(img_path).convert_alpha()
    texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
    width, height = texture_surface.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id

def rotate_image_180(img_path):
    image = pygame.image.load(img_path).convert_alpha()
    return pygame.transform.rotate(image, 180)

def load_rotated_texture(rotated_image):
    texture_data = pygame.image.tostring(rotated_image, "RGBA", 1)
    width, height = rotated_image.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id

# Preload textures
def preload_textures(img_files):
    preloaded_textures = {}
    for img_path in img_files:
        texture_id = load_texture(img_path)
        preloaded_textures[img_path] = texture_id
    print(f"Preloaded {len(preloaded_textures)} textures")
    return preloaded_textures

# Cube creation and drawing
def create_cube_with_images(cube_index, front_side_image_path, adjacent_sides_image_path, texture_ids_list, preloaded_textures):
    texture_ids_list[cube_index][0] = preloaded_textures[front_side_image_path]
    for i in range(1, 6):
        if i == 3:  # If it's the bottom face, rotate the image 180 degrees
            rotated_image = rotate_image_180(adjacent_sides_image_path)
            texture_ids_list[cube_index][i] = load_rotated_texture(rotated_image)
        else:
            texture_ids_list[cube_index][i] = preloaded_textures[adjacent_sides_image_path]

def draw_cube(cube_index, texture_ids_list):
    glEnable(GL_TEXTURE_2D)
    for i, texture_id in enumerate(texture_ids_list[cube_index]):
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_QUADS)
        
        # Adjust the cube size to 1.5 for better fit
        if i == 0:  # Front face
            glTexCoord2f(0.0, 0.0); glVertex3f(-1.5, -1.5, 1.5)
            glTexCoord2f(1.0, 0.0); glVertex3f(1.5, -1.5, 1.5)
            glTexCoord2f(1.0, 1.0); glVertex3f(1.5, 1.5, 1.5)
            glTexCoord2f(0.0, 1.0); glVertex3f(-1.5, 1.5, 1.5)
        elif i == 1:  # Back face
            glTexCoord2f(1.0, 0.0); glVertex3f(-1.5, -1.5, -1.5)
            glTexCoord2f(1.0, 1.0); glVertex3f(-1.5, 1.5, -1.5)
            glTexCoord2f(0.0, 1.0); glVertex3f(1.5, 1.5, -1.5)
            glTexCoord2f(0.0, 0.0); glVertex3f(1.5, -1.5, -1.5)
        elif i == 2:  # Top face
            glTexCoord2f(0.0, 1.0); glVertex3f(-1.5, 1.5, -1.5)
            glTexCoord2f(0.0, 0.0); glVertex3f(-1.5, 1.5, 1.5)
            glTexCoord2f(1.0, 0.0); glVertex3f(1.5, 1.5, 1.5)
            glTexCoord2f(1.0, 1.0); glVertex3f(1.5, 1.5, -1.5)
        elif i == 3:  # Bottom face
            glTexCoord2f(1.0, 1.0); glVertex3f(-1.5, -1.5, -1.5)
            glTexCoord2f(0.0, 1.0); glVertex3f(1.5, -1.5, -1.5)
            glTexCoord2f(0.0, 0.0); glVertex3f(1.5, -1.5, 1.5)
            glTexCoord2f(1.0, 0.0); glVertex3f(-1.5, -1.5, 1.5)
        elif i == 4:  # Right face
            glTexCoord2f(1.0, 0.0); glVertex3f(1.5, -1.5, -1.5)
            glTexCoord2f(1.0, 1.0); glVertex3f(1.5, 1.5, -1.5)
            glTexCoord2f(0.0, 1.0); glVertex3f(1.5, 1.5, 1.5)
            glTexCoord2f(0.0, 0.0); glVertex3f(1.5, -1.5, 1.5)
        elif i == 5:  # Left face
            glTexCoord2f(0.0, 0.0); glVertex3f(-1.5, -1.5, -1.5)
            glTexCoord2f(1.0, 0.0); glVertex3f(-1.5, -1.5, 1.5)
            glTexCoord2f(1.0, 1.0); glVertex3f(-1.5, 1.5, 1.5)
            glTexCoord2f(0.0, 1.0); glVertex3f(-1.5, 1.5, -1.5)

        glEnd()
    glDisable(GL_TEXTURE_2D)

# Rotation calculations
def calculate_next_rotation():
    rotation_sequence = [
        (0, 1, 0, 90),   # Rotate 90 degrees around the Y-axis (to the right)
        (0, -1, 0, 90),  # Rotate 90 degrees around the Y-axis (to the left)
        (1, 0, 0, 90),   # Rotate 90 degrees around the X-axis (downward)
        (-1, 0, 0, 90)   # Rotate 90 degrees around the X-axis (upward)
    ]
    return random.choice(rotation_sequence)

def ease_in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def main():
    global GRID_SIZE

    base_input_folder = select_folder()
    print(f"Selected folder: {base_input_folder}")
    
    # Define directories for N*N input folders
    source_images_directories = [
        os.path.join(base_input_folder, f'source_images_{i+1}') for i in range(GRID_SIZE * GRID_SIZE)
    ]
    
    # Ensure all directories exist
    for directory in source_images_directories:
        ensure_directory_exists(directory)
    
    # Get image paths for each cube
    img_files_list = [sorted(glob.glob(f'{dir}/*.png')) for dir in source_images_directories]
    
    # Check if any images were found and if each folder has at least 2 images
    if not any(img_files_list):
        print(f"Error: No PNG images found in the selected directories.")
        print(f"Searched directories:")
        for directory in source_images_directories:
            print(f"  - {directory}")
        print("Please ensure that your images are in the correct directories and are in PNG format.")
        return

    valid_cubes = []
    for i, img_files in enumerate(img_files_list):
        if len(img_files) < 2:
            print(f"Warning: Cube {i+1} (source_images_{i+1}) has less than 2 images. It will be skipped.")
        else:
            valid_cubes.append(i)

    if not valid_cubes:
        print("Error: No valid cubes to display. Each cube needs at least 2 images.")
        return

    # Adjust GRID_SIZE if necessary
    effective_grid_size = int(math.sqrt(len(valid_cubes)))
    if effective_grid_size != GRID_SIZE:
        print(f"Adjusting grid size from {GRID_SIZE}x{GRID_SIZE} to {effective_grid_size}x{effective_grid_size} due to insufficient valid cubes.")
        GRID_SIZE = effective_grid_size

    # Print debugging information
    print(f"Found images:")
    for i in valid_cubes:
        print(f"  Cube {i+1}: {len(img_files_list[i])} images")
    
    init_pygame_opengl()
    
    # Calculate spacing based on grid size
    spacing = (3 / (GRID_SIZE - 1) if GRID_SIZE > 1 else 0) * CUBE_SPACING
    
    # Generate cube positions
    cube_positions = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = (col - (GRID_SIZE - 1) / 2) * spacing
            y = ((GRID_SIZE - 1) / 2 - row) * spacing
            cube_positions.append((x, y, 0))
    
    # Preload textures
    preloaded_textures = {}
    for i in valid_cubes:
        preloaded_textures.update(preload_textures(img_files_list[i]))
    
    # Lists to hold texture IDs for each cube
    texture_ids_list = [[None] * 6 for _ in valid_cubes]
    
    # Create rotation information for each cube
    cube_rotations = [
        {'current_axis': None, 'start_angle': 0, 'end_angle': 90, 'start_time': 0} for _ in valid_cubes
    ]
    
    img_counts = [0] * len(valid_cubes)
    clock = pygame.time.Clock()
    start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        for idx, cube_index in enumerate(valid_cubes):
            if img_counts[idx] >= len(img_files_list[cube_index]) - 1:
                img_counts[idx] = 0
            
            create_cube_with_images(
                idx,
                img_files_list[cube_index][img_counts[idx]],
                img_files_list[cube_index][img_counts[idx] + 1],
                texture_ids_list,
                preloaded_textures
            )
            
            rotation = calculate_next_rotation()
            cube_rotations[idx]['current_axis'] = rotation[:3]
            cube_rotations[idx]['start_angle'] = 0
            cube_rotations[idx]['end_angle'] = rotation[3]
            cube_rotations[idx]['start_time'] = current_time
        
        print(f"\nDisplaying images at {elapsed_time:.2f}s:")
        print(f"  Displaying for {SIDE_DISPLAY_TIME:.2f}s")
        
        # Display current sides for SIDE_DISPLAY_TIME seconds
        side_display_start = time.time()
        while time.time() - side_display_start < SIDE_DISPLAY_TIME:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for idx, cube_index in enumerate(valid_cubes):
                glPushMatrix()
                glTranslatef(*cube_positions[idx])
                draw_cube(idx, texture_ids_list)
                glPopMatrix()
            
            pygame.display.flip()
            clock.tick(60)
        
        rotation_start = time.time()
        print(f"  Rotation started at {rotation_start - start_time:.2f}s")
        
        # Perform rotation
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            current_time = time.time()
            rotation_elapsed_time = current_time - rotation_start
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            all_rotations_complete = True
            for idx, cube_index in enumerate(valid_cubes):
                rotation = cube_rotations[idx]
                progress = min(rotation_elapsed_time / ROTATION_SPEED, 1.0)
                eased_progress = ease_in_out_cubic(progress)
                current_angle = rotation['start_angle'] + (rotation['end_angle'] - rotation['start_angle']) * eased_progress
                
                glPushMatrix()
                glTranslatef(*cube_positions[idx])
                glRotatef(current_angle, *rotation['current_axis'])
                draw_cube(idx, texture_ids_list)
                glPopMatrix()
                
                if progress < 1.0:
                    all_rotations_complete = False
            
            pygame.display.flip()
            clock.tick(60)
            
            if all_rotations_complete:
                break
        
        rotation_end = time.time()
        print(f"  Rotation completed at {rotation_end - start_time:.2f}s")
        print(f"  Rotation duration: {rotation_end - rotation_start:.2f}s")
        
        for i in range(len(valid_cubes)):
            img_counts[i] += 1
    
    pygame.quit()
    quit()

if __name__ == "__main__":
    main()