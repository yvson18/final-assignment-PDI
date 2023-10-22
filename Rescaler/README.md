# Image Rescaling Application

## Available modes
- [x] Downsample
- [ ] Upsample

## Available methods
- [x] Bilinear
- [x] Bicubic
- [x] Lanczos
- [ ] GFPGAN
- [ ] PULSE

## Usage
Install the Tkinter (Tk) and OpenCV packages and run the python file:
```
python rescaler.py
```

Or, avoid installing any packages by running the executable directly:
```
.\rescaler.exe
```

The application supports the following optional flags for specifying parameters before execution:
```
options:
  -h, --help            show this help message and exit
  --mode {downscale,upscale}
                        Specify the mode (downscale or upscale)
  --factor FACTOR       Scaling factor for resizing
  --method {bilinear,bicubic,lanczos}
                        Resampling method
  --root ROOT           Root directory containing the images
```
For any unspecified parameters, the program will prompt the user for input during execution.
