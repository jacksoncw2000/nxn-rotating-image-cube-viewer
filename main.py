import os
import time
import datetime
import glob
import random
import math
from tqdm import tqdm
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from moviepy.editor import ImageSequenceClip

# Constants and configuration
GRID_SIZE = 9  # Change this to 1, 2, 3, 4, etc. for different grid sizes
VIEW_DISTANCE = 22  # The higher the number, the farther out the camera view will be from the cubes
CUBE_SPACING = 8  # The higher the number, the more space between cubes
SIDE_DISPLAY_TIME = 0.5  # How long to stay on each side (in seconds)
ROTATION_SPEED = 0.5  # Adjust this value to control the speed of rotation (degrees per second)
FPS = 30  # Frames per second for the output video
VIDEO_RESOLUTION = (1000, 1000)

# Ensure directory exists
def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Pygame and OpenGL initialization
def init_pygame_opengl():
    pygame.init()
    display = VIDEO_RESOLUTION
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glViewport(0, 0, display[0], display[1])
    gluPerspective(30, (display[0] / display[1]), 0.1, 50.0)
    glEnable(GL_DEPTH_TEST)
    camera_z = -(VIEW_DISTANCE + GRID_SIZE * 2.5)
    glTranslatef(0.0, 0.0, camera_z)

# Texture loading and management
def load_texture(img_path):
    max_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
    #print(f"Maximum texture size supported: {max_size}x{max_size}")
    
    texture_surface = pygame.image.load(img_path).convert_alpha()
    texture_data = pygame.image.tobytes(texture_surface, "RGBA", 1)
    width, height = texture_surface.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id, (width, height)

def rotate_image_180(img_path):
    image = pygame.image.load(img_path).convert_alpha()
    return pygame.transform.rotate(image, 180)

def load_rotated_texture(rotated_image):
    texture_data = pygame.image.tobytes(rotated_image, "RGBA", 1)
    width, height = rotated_image.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id, (width, height)

# Preload textures
def preload_textures(img_files):
    preloaded_textures = {}
    for img_path in img_files:
        texture_id, dimensions = load_texture(img_path)
        preloaded_textures[img_path] = (texture_id, dimensions)
    #print(f"Preloaded {len(preloaded_textures)} textures")
    return preloaded_textures

def delete_texture(texture_id):
    glDeleteTextures(1, [texture_id])

def check_texture_size(texture_id):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
    return width, height
 
# Cube creation and drawing
def create_cube_with_images(cube_index, front_side_image_path, adjacent_sides_image_path, texture_ids_list, preloaded_textures):
    texture_ids_list[cube_index][0] = preloaded_textures[front_side_image_path][0]
    actual_size = check_texture_size(texture_ids_list[cube_index][0])
    #print(f"Cube {cube_index + 1} front face:")
    #print(f"  Reported resolution: {preloaded_textures[front_side_image_path][1][0]}x{preloaded_textures[front_side_image_path][1][1]}")
    #print(f"  Actual GL texture size: {actual_size[0]}x{actual_size[1]}")
    
    for i in range(1, 6):
        if i == 3:  # If it's the bottom face, rotate the image 180 degrees
            rotated_image = rotate_image_180(adjacent_sides_image_path)
            texture_id, dimensions = load_rotated_texture(rotated_image)
            texture_ids_list[cube_index][i] = texture_id
            actual_size = check_texture_size(texture_id)
            #print(f"Cube {cube_index + 1} bottom face:")
            #print(f"  Reported resolution: {dimensions[0]}x{dimensions[1]}")
            #print(f"  Actual GL texture size: {actual_size[0]}x{actual_size[1]}")
        else:
            texture_ids_list[cube_index][i] = preloaded_textures[adjacent_sides_image_path][0]
    
    actual_size = check_texture_size(preloaded_textures[adjacent_sides_image_path][0])
    #print(f"Cube {cube_index + 1} other faces:")
    #print(f"  Reported resolution: {preloaded_textures[adjacent_sides_image_path][1][0]}x{preloaded_textures[adjacent_sides_image_path][1][1]}")
    #print(f"  Actual GL texture size: {actual_size[0]}x{actual_size[1]}")

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

def render_frame(valid_cubes, cube_positions, texture_ids_list, cube_rotations):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    for idx, cube_index in enumerate(valid_cubes):
        rotation = cube_rotations[idx]
        
        glPushMatrix()
        glTranslatef(*cube_positions[idx])
        if rotation['is_rotating']:
            eased_progress = ease_in_out_cubic(rotation['rotation_progress'])
            current_angle = rotation['start_angle'] + (rotation['end_angle'] - rotation['start_angle']) * eased_progress
            glRotatef(current_angle, *rotation['current_axis'])
        draw_cube(idx, texture_ids_list)
        glPopMatrix()
    
    pygame.display.flip()
    
    # Capture the frame using the VIDEO_RESOLUTION
    frame_data = glReadPixels(0, 0, VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], GL_RGB, GL_UNSIGNED_BYTE)
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0], 3))
    return np.flipud(frame)  # Flip the frame vertically

def main():
    global GRID_SIZE

    # Get the selected folder from the select_folder.py script
    selected_folder = os.popen('python3 select_folder.py').read().strip()
    base_input_folder = selected_folder
    print(f"Selected folder: {base_input_folder}")
    
    print(f"Video resolution set to: {VIDEO_RESOLUTION}")

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
    
    # Preload textures for all cubes
    preloaded_textures = {}
    print("Preloading textures...")
    for cube_files in tqdm(img_files_list, desc="Preloading textures", unit="cube"):
        preloaded_textures.update(preload_textures(cube_files))
    
    texture_ids_list = [[None] * 6 for _ in valid_cubes]
    
    # Create rotation information for each cube
    cube_rotations = [
        {'current_axis': None, 'start_angle': 0, 'end_angle': 90, 'rotation_progress': 0, 'is_rotating': False, 'needs_texture_update': False} 
        for _ in valid_cubes
    ]

    img_counts = [0] * len(valid_cubes)
    frames = []
    
    # Calculate total frames
    max_images = max(len(img_files_list[i]) for i in valid_cubes)
    total_frames = int(max_images * (SIDE_DISPLAY_TIME + ROTATION_SPEED) * FPS)

    # Initialize cubes with first images
    print("Initializing cubes...")
    for idx, cube_index in enumerate(tqdm(valid_cubes, desc="Initializing cubes", unit="cube")):
        front_image = img_files_list[cube_index][0]
        adjacent_image = img_files_list[cube_index][1] if len(img_files_list[cube_index]) > 1 else front_image
        create_cube_with_images(idx, front_image, adjacent_image, texture_ids_list, preloaded_textures)

    current_time = 0
    next_rotation_time = SIDE_DISPLAY_TIME

    print("Rendering frames...")
    for frame in tqdm(range(total_frames), desc="Rendering frames", unit="frame"):
        current_time += 1 / FPS

        if current_time >= next_rotation_time:
            for idx, cube_index in enumerate(valid_cubes):
                if img_counts[idx] >= len(img_files_list[cube_index]) - 1:
                    continue  # Skip updating if we've shown all images

                # Start rotation
                rotation = calculate_next_rotation()
                cube_rotations[idx]['current_axis'] = rotation[:3]
                cube_rotations[idx]['start_angle'] = 0
                cube_rotations[idx]['end_angle'] = rotation[3]
                cube_rotations[idx]['rotation_progress'] = 0
                cube_rotations[idx]['is_rotating'] = True
                cube_rotations[idx]['needs_texture_update'] = True

                # Increment img_counts before updating textures
                img_counts[idx] += 1

            next_rotation_time = current_time + SIDE_DISPLAY_TIME + ROTATION_SPEED

        # Update rotations and textures
        for idx, rotation in enumerate(cube_rotations):
            if rotation['is_rotating']:
                rotation['rotation_progress'] += 1 / (ROTATION_SPEED * FPS)
                if rotation['rotation_progress'] >= 1:
                    rotation['is_rotating'] = False
                    rotation['rotation_progress'] = 1
                    
                    if rotation['needs_texture_update']:
                        # Update textures after rotation is complete
                        cube_index = valid_cubes[idx]
                        current_image = img_files_list[cube_index][img_counts[idx]]
                        next_image = img_files_list[cube_index][img_counts[idx] + 1] if img_counts[idx] + 1 < len(img_files_list[cube_index]) else current_image
                        create_cube_with_images(idx, current_image, next_image, texture_ids_list, preloaded_textures)
                        rotation['needs_texture_update'] = False

        # Render and capture the frame
        frame_image = render_frame(valid_cubes, cube_positions, texture_ids_list, cube_rotations)
        frames.append(frame_image)
    
    # Generate the current date and time string
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create and save the video
    output_path = os.path.join(base_input_folder, f"{current_datetime}_rotating_cubes.mp4")
    
    print("Creating video...")
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(output_path, codec="libx264", logger='bar')
    
    print(f"Video saved to: {output_path}")
    
    pygame.quit()

if __name__ == "__main__":
    main()
