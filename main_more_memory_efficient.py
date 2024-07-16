import os
import time
import datetime
import subprocess
import glob
import random
import math
from tqdm import tqdm
import traceback
import logging
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from moviepy.editor import ImageSequenceClip


# Constants and configuration
GRID_SIZE = 10  # Change this to 1, 2, 3, 4, etc. for different grid sizes
VIEW_DISTANCE = 25  # The higher the number, the farther out the camera view will be from the cubes
CUBE_SPACING = 9  # The higher the number, the more space between cubes
SIDE_DISPLAY_TIME = 0.5  # How long to stay on each side (in seconds)
ROTATION_SPEED = 0.75 # Adjust this value to control the speed of rotation (degrees per second)
FPS = 60  # Frames per second for the output video
VIDEO_RESOLUTION = (2000, 2000)
CAMERA_POSITION_MODIFIER = 1.31  # the higher the number, the farther away the camera position? How does this differ from VIEW_DISTANCE?


logging.basicConfig(filename='cube_renderer.log', level=logging.DEBUG)

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def init_pygame_opengl():
    pygame.init()
    display = VIDEO_RESOLUTION
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glViewport(0, 0, display[0], display[1])
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glEnable(GL_DEPTH_TEST)
    camera_z = -(VIEW_DISTANCE + GRID_SIZE * CAMERA_POSITION_MODIFIER)
    glTranslatef(0.0, 0.0, camera_z)

def load_texture(img_path):
    texture_surface = pygame.image.load(img_path).convert_alpha()
    texture_data = pygame.image.tobytes(texture_surface, "RGBA", 1)
    width, height = texture_surface.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    del texture_surface  # Release the surface immediately
    return texture_id

def rotate_image_180(img_path):
    image = pygame.image.load(img_path).convert_alpha()
    rotated = pygame.transform.rotate(image, 180)
    del image  # Release the original image
    return rotated

def load_rotated_texture(rotated_image):
    texture_data = pygame.image.tobytes(rotated_image, "RGBA", 1)
    width, height = rotated_image.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    del rotated_image  # Release the rotated image
    return texture_id

def create_cube_with_images(cube_index, front_side_image_path, adjacent_sides_image_path, texture_ids_list):
    texture_ids_list[cube_index][0] = load_texture(front_side_image_path)
    
    for i in range(1, 6):
        if i == 3:  # Bottom face
            rotated_image = rotate_image_180(adjacent_sides_image_path)
            texture_ids_list[cube_index][i] = load_rotated_texture(rotated_image)
        else:
            texture_ids_list[cube_index][i] = load_texture(adjacent_sides_image_path)

def draw_cube(cube_index, texture_ids_list):
    glEnable(GL_TEXTURE_2D)
    for i, texture_id in enumerate(texture_ids_list[cube_index]):
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_QUADS)
        
        # Vertex coordinates
        vertices = [
            [(-1.5, -1.5, 1.5), (1.5, -1.5, 1.5), (1.5, 1.5, 1.5), (-1.5, 1.5, 1.5)],  # Front
            [(-1.5, -1.5, -1.5), (-1.5, 1.5, -1.5), (1.5, 1.5, -1.5), (1.5, -1.5, -1.5)],  # Back
            [(-1.5, 1.5, -1.5), (-1.5, 1.5, 1.5), (1.5, 1.5, 1.5), (1.5, 1.5, -1.5)],  # Top
            [(-1.5, -1.5, -1.5), (1.5, -1.5, -1.5), (1.5, -1.5, 1.5), (-1.5, -1.5, 1.5)],  # Bottom
            [(1.5, -1.5, -1.5), (1.5, 1.5, -1.5), (1.5, 1.5, 1.5), (1.5, -1.5, 1.5)],  # Right
            [(-1.5, -1.5, -1.5), (-1.5, -1.5, 1.5), (-1.5, 1.5, 1.5), (-1.5, 1.5, -1.5)]  # Left
        ]
        
        # Texture coordinates
        tex_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        
        for vertex, tex_coord in zip(vertices[i], tex_coords):
            glTexCoord2f(*tex_coord)
            glVertex3f(*vertex)

        glEnd()
    glDisable(GL_TEXTURE_2D)

def calculate_next_rotation():
    rotation_sequence = [
        (0, 1, 0, 90),   # Y-axis (right)
        (0, -1, 0, 90),  # Y-axis (left)
        (1, 0, 0, 90),   # X-axis (down)
        (-1, 0, 0, 90)   # X-axis (up)
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
    
    frame_data = glReadPixels(0, 0, VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], GL_RGB, GL_UNSIGNED_BYTE)
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0], 3))
    return np.flipud(frame)

def main():
    global GRID_SIZE

    selected_folder = os.popen('python3 select_folder.py').read().strip()
    base_input_folder = selected_folder
    print(f"Selected folder: {base_input_folder}")
    print(f"Video resolution set to: {VIDEO_RESOLUTION}")

    source_images_directories = [
        os.path.join(base_input_folder, f'source_images_{i+1}') for i in range(GRID_SIZE * GRID_SIZE)
    ]
    
    for directory in source_images_directories:
        ensure_directory_exists(directory)
    
    img_files_list = [sorted(glob.glob(f'{dir}/*.png')) for dir in source_images_directories]
    
    if not any(img_files_list):
        print(f"Error: No PNG images found in the selected directories.")
        print(f"Searched directories:")
        for directory in source_images_directories:
            print(f"  - {directory}")
        print("Please ensure that your images are in the correct directories and are in PNG format.")
        return

    valid_cubes = [i for i, img_files in enumerate(img_files_list) if len(img_files) >= 2]

    if not valid_cubes:
        print("Error: No valid cubes to display. Each cube needs at least 2 images.")
        return

    effective_grid_size = int(math.sqrt(len(valid_cubes)))
    if effective_grid_size != GRID_SIZE:
        print(f"Adjusting grid size from {GRID_SIZE}x{GRID_SIZE} to {effective_grid_size}x{effective_grid_size} due to insufficient valid cubes.")
        GRID_SIZE = effective_grid_size

    print(f"Found images:")
    for i in valid_cubes:
        print(f"  Cube {i+1}: {len(img_files_list[i])} images")
    
    init_pygame_opengl()
    
    spacing = (3 / (GRID_SIZE - 1) if GRID_SIZE > 1 else 0) * CUBE_SPACING
    
    cube_positions = [
        ((col - (GRID_SIZE - 1) / 2) * spacing, ((GRID_SIZE - 1) / 2 - row) * spacing, 0)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]
    
    texture_ids_list = [[None] * 6 for _ in valid_cubes]
    
    cube_rotations = [
        {'current_axis': None, 'start_angle': 0, 'end_angle': 90, 'rotation_progress': 0, 'is_rotating': False, 'needs_texture_update': False} 
        for _ in valid_cubes
    ]

    img_counts = [0] * len(valid_cubes)
    
    max_images = max(len(img_files_list[i]) for i in valid_cubes)
    total_frames = int(max_images * (SIDE_DISPLAY_TIME + ROTATION_SPEED) * FPS)

    print("Initializing cubes...")
    for idx, cube_index in enumerate(tqdm(valid_cubes, desc="Initializing cubes", unit="cube")):
        front_image = img_files_list[cube_index][0]
        adjacent_image = img_files_list[cube_index][1] if len(img_files_list[cube_index]) > 1 else front_image
        create_cube_with_images(idx, front_image, adjacent_image, texture_ids_list)

    current_time = 0
    next_rotation_time = SIDE_DISPLAY_TIME

    print("Rendering frames...")
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_path = os.path.join(base_input_folder, f"{current_datetime}_rotating_cubes.mp4")
    
    # Create a temporary directory for frames and video segments
    temp_dir = os.path.join(base_input_folder, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    batch_size = 50  # Adjust this value based on your system's capabilities
    video_segments = []

    try:
        for batch_start in range(0, total_frames, batch_size):
            try:
                batch_end = min(batch_start + batch_size, total_frames)
                
                for frame in tqdm(range(batch_start, batch_end), desc=f"Rendering batch {batch_start//batch_size + 1}", unit="frame"):
                    try:
                        current_time += 1 / FPS

                        if current_time >= next_rotation_time:
                            for idx, cube_index in enumerate(valid_cubes):
                                if img_counts[idx] >= len(img_files_list[cube_index]) - 1:
                                    continue

                                rotation = calculate_next_rotation()
                                cube_rotations[idx].update({
                                    'current_axis': rotation[:3],
                                    'start_angle': 0,
                                    'end_angle': rotation[3],
                                    'rotation_progress': 0,
                                    'is_rotating': True,
                                    'needs_texture_update': True
                                })

                                img_counts[idx] += 1

                            next_rotation_time = current_time + SIDE_DISPLAY_TIME + ROTATION_SPEED

                        for idx, rotation in enumerate(cube_rotations):
                            if rotation['is_rotating']:
                                rotation['rotation_progress'] += 1 / (ROTATION_SPEED * FPS)
                                if rotation['rotation_progress'] >= 1:
                                    rotation['is_rotating'] = False
                                    rotation['rotation_progress'] = 1
                                    
                                    if rotation['needs_texture_update']:
                                        cube_index = valid_cubes[idx]
                                        current_image = img_files_list[cube_index][img_counts[idx]]
                                        next_image = img_files_list[cube_index][img_counts[idx] + 1] if img_counts[idx] + 1 < len(img_files_list[cube_index]) else current_image
                                        create_cube_with_images(idx, current_image, next_image, texture_ids_list)
                                        rotation['needs_texture_update'] = False

                        frame_image = render_frame(valid_cubes, cube_positions, texture_ids_list, cube_rotations)
                        
                        # Save frame to disk
                        frame_path = os.path.join(temp_dir, f"frame_{frame:05d}.png")
                        pygame.image.save(pygame.surfarray.make_surface(frame_image), frame_path)
                        
                    except Exception as e:
                        logging.error(f"Error rendering frame {frame}: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise

                # Create video segment from the batch of frames
                segment_path = os.path.join(temp_dir, f"segment_{batch_start//batch_size:05d}.mp4")
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(FPS),
                    "-start_number", str(batch_start),
                    "-i", os.path.join(temp_dir, "frame_%05d.png").replace("\\", "/"),
                    "-frames:v", str(batch_end - batch_start),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    segment_path.replace("\\", "/")
                ]
                result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"Error creating video segment. FFmpeg command: {' '.join(ffmpeg_command)}")
                    logging.error(f"FFmpeg stderr output: {result.stderr}")
                    raise Exception("FFmpeg command failed")
                
                video_segments.append(segment_path)

                # Remove frame images to free up disk space
                for frame in range(batch_start, batch_end):
                    try:
                        os.remove(os.path.join(temp_dir, f"frame_{frame:05d}.png"))
                    except OSError as e:
                        logging.warning(f"Error removing frame {frame}: {e}")

            except Exception as e:
                logging.error(f"Error processing batch starting at frame {batch_start}: {str(e)}")
                logging.error(traceback.format_exc())
                raise

        print("Combining video segments...")
        # Create a file listing all video segments
        segment_list_path = os.path.join(temp_dir, "segment_list.txt")
        with open(segment_list_path, 'w') as f:
            for segment in video_segments:
                f.write(f"file '{os.path.basename(segment)}'\n")

        # Combine all segments into the final video
        concat_command = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", segment_list_path.replace("\\", "/"),
            "-c", "copy",
            final_output_path.replace("\\", "/")
        ]
        result = subprocess.run(concat_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error combining video segments. FFmpeg command: {' '.join(concat_command)}")
            print(f"FFmpeg stderr output: {result.stderr}")
            raise Exception("FFmpeg command failed")
        
        print(f"Video saved to: {final_output_path}")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"An error occurred. Please check the log file 'cube_renderer.log' for details.")
    
    finally:
        # Clean up
        for texture_list in texture_ids_list:
            for texture_id in texture_list:
                glDeleteTextures(1, [texture_id])
        
        pygame.quit()

        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()