import os
from shutil import rmtree
import numpy as np

def create_directory_if_not_defined(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def delete_files_in_directory(dir,recursive=False):
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and recursive: rmtree(file_path)
        except Exception as e:
            print(e)

def setup_clean_directory(dir):
    create_directory_if_not_defined(dir)
    delete_files_in_directory(dir,recursive=True)

def get_files_of_type(path, type='jpg'):
    return np.array([x for x in sorted(os.listdir(path)) if x.lower().endswith(type.lower())])

def get_subdirectories(path):
    return os.walk(path).__next__()[1]
