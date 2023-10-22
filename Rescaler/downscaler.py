from utils import os, cv2, time, read_img, rescale_img, build_relpath

def can_downscale(img, factor):
    (h, w) = img.shape[:2]
    return h >= factor and w >= factor

def downscale_imgs(img_paths, old_root, factor, method):
    print("Starting Downscale...")

    start_t = time()
    new_root = ".\\downscaler_out"
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    new_root = os.path.join(new_root, method)
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    methods_dict = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    scale_factor = 1.0/factor
    skip = []

    for img_path in img_paths:
        img = read_img(img_path)
        if can_downscale(img, factor):
            new_img = rescale_img(img, scale_factor, methods_dict[method])
            new_relpath = build_relpath(img_path, old_root, new_root)
            cv2.imwrite(new_relpath, new_img)
        else:
            skip.append(img_path)
    end_t = time()

    print("Downscale finished.")
    print("Elapsed time: {:.6f} seconds".format(end_t - start_t))
    if skip:
        print(f"Could't Downscale:")
        for f in skip:
            print(f)
