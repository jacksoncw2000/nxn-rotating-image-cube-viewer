import os
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Define the absolute path to the last_folder.txt file
    last_folder_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_folder.txt")
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

if __name__ == "__main__":
    folder_path = select_folder()
    print(folder_path)
