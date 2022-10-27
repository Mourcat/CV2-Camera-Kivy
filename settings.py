from pathlib import Path
import os
import cv2


BASE_DIR = Path.cwd()
IMG_DIR = BASE_DIR.joinpath('imgs')

cascade_filepath_face = os.path.dirname(cv2.__file__)+ "/data/haarcascade_frontalface_alt2.xml"

cascade_filepath_eyes = os.path.dirname(cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"