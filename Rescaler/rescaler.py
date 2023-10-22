from utils import *
from downscaler import *

if __name__ == "__main__":
    args = get_args()

    if args['root'] is None:
        provide_root(args)
    if args['mode'] is None:
        choose_mode(args)
    if args['factor'] is None:
        provide_factor(args)
    if args['method'] is None:
        choose_method(args)

    print_args(args)
    mode, factor, method, root = args.values()

    img_paths = get_img_paths(root)

    if mode == "downscale":
        downscale_imgs(img_paths, root, factor, method)
