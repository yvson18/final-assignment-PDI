from utils import *

def can_downscale(img, factor):
    (h, w) = img.shape[:2]
    return h >= factor and w >= factor

def downscale_imgs(img_paths, old_root, factor, method):
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
    failed_downscale = []

    print("Starting Downscale...")
    start_t = time()
    for img_path in img_paths:
        img = read_img(img_path)
        if can_downscale(img, factor):
            new_img = cv_rescale_img(img, scale_factor, methods_dict[method])
            new_relpath = build_relpath(img_path, old_root, new_root)
            cv2.imwrite(new_relpath, new_img)
        else:
            failed_downscale.append(img_path)
    end_t = time()
    print("Downscale finished.")
    print("Elapsed time: {:.6f} seconds".format(end_t - start_t))
    if failed_downscale:
        print(f"Failed to Downscale:")
        for f in failed_downscale:
            print(f)
    input()
