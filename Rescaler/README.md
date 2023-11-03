# Image Rescaling Application

## Available modes
- [x] Downscale
- [x] Upscale

## Available methods
- [x] Bilinear
- [x] Bicubic
- [x] Lanczos
- [x] GFPGAN
- [x] Real-ESRGAN

## Authenticating with Replica
To utilize the GFPGAN and Real-ESRGAN upscaling, authentication with Replica's API is required. Instructions for exporting your Replica token can be found [here](https://replicate.com/docs/get-started/python#authenticate).

## Usage

### Option 1: Running with Python
Ensure you have the required packages installed: Tkinter (Tk), OpenCV, Replicate, and Requests.
Run the Python script with the following command:
```
python rescaler.py
```

### Option 2: Executable
Avoid installing any packages by running the standalone executable directly:
```
.\rescaler.exe
```

## Optional Flags
The application supports the following optional flags for specifying parameters before execution:
```
options:
  -h, --help            show this help message and exit
  --mode {downscale,upscale}
                        Specify the mode (downscale or upscale)
  --factor FACTOR       Scaling factor for resizing
  --method {bilinear,bicubic,lanczos,gfpgan,real-esrgan}
                        Resampling method
  --root ROOT           Root directory containing the images
```
For any unspecified parameters, the program will prompt the user for input during execution.
