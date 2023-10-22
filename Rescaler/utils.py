import os
import cv2
import pathlib
import argparse
import numpy as np
from time import time
from tkinter import Tk, filedialog

def sys_clear():
    if os.name == 'posix':
        os.system('clear')
    elif os.name == 'nt':
        os.system('cls')

def get_args():
    parser = argparse.ArgumentParser(description="Image Rescaling Application")
    parser.add_argument("--mode", required=False, choices=["downscale", "upscale"], help="Specify the mode (downscale or upscale)")
    parser.add_argument("--factor", required=False, type=float, help="Scaling factor for resizing")
    parser.add_argument("--method", required=False, choices=["bilinear", "bicubic", "lanczos"], help="Resampling method")
    parser.add_argument("--root", required=False, help="Root directory containing the images")
    args = parser.parse_args()
    return vars(args)

def print_args(args):
    sys_clear()
    print("Image Rescaling Application")
    print(f"Mode   : {args['mode']}")
    print(f"Factor : {args['factor']}")
    print(f"Method : {args['method']}")
    print(f"Root   : {args['root']}")
    print()

def choose_mode(args):
    modes_dict = {
        '1': "downscale",
        '2': "upscale"
    }
    while True:
        print_args(args)
        print("Choose a scaling mode:")
        print("1 - Downscale")
        print("2 - Upscale")
        mode = input()
        if mode in modes_dict:
            args['mode'] = modes_dict[mode]
            break

def provide_factor(args):
    print_args(args)
    factor = input("Provide a scaling factor: ")
    args['factor'] = float(factor)

def print_methods(mode):
    if mode == "downscale":
        print("1 - Bilinear")
        print("2 - Bicubic")
        print("3 - Lanczos")

def choose_method(args):
    methods_dict = {
        "downscale": {
            '1': "bilinear",
            '2': "bicubic",
            '3': "lanczos",
        },
        "upscale": ""
    }
    mode = args['mode']
    while True:
        print_args(args)
        print("Choose a scaling method:")
        print_methods(mode)
        method = input()
        if method in methods_dict[mode]:
            args['method'] = methods_dict[mode][method]
            break

def provide_root(args):
    print_args(args)
    print("Provide a root directory.")
    root = filedialog.askdirectory(
        initialdir=os.getcwd(),
        title="Select Root Directory",
        mustexist=True
    )
    args['root'] = root

def get_img_paths(root):
    allowed_extensions = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.webp')
    img_paths = []
    for extension in allowed_extensions:
        for path in pathlib.Path(root).glob(f"**/{extension}"):
            img_paths.append(str(path))

    img_paths = list(set(img_paths))
    print(f"{len(img_paths)} images found.")

    return img_paths

def build_relpath(path, old_root, new_root):
    old_relpath = os.path.relpath(path, old_root)
    old_relpath_parts = old_relpath.split(os.path.sep)
    new_relpath = new_root
    for dir in old_relpath_parts[:-1]:
        new_relpath = os.path.join(new_relpath, dir)
        if not os.path.exists(new_relpath):
            os.mkdir(new_relpath)
    new_relpath = os.path.join(new_relpath, old_relpath_parts[-1])
    return new_relpath

def read_img(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img

def rescale_img(img, scale_factor, method):
    new_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=method)
    return new_img
