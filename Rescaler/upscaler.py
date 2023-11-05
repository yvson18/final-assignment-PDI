from utils import *

def cv_upscale_imgs(img_paths, old_root, new_root, factor, method):
    cv_methods_dict = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    scale_factor = factor

    print("Starting Upscale...")
    start_t = time()
    for img_path in img_paths:
        img = read_img(img_path)
        new_img = cv_rescale_img(img, scale_factor, cv_methods_dict[method])
        new_relpath = build_relpath(img_path, old_root, new_root)
        cv2.imwrite(new_relpath, new_img)
    end_t = time()
    print("Upscale finished.")
    print("Elapsed time: {:.6f} seconds".format(end_t - start_t))
    input()

def start_model_predicts(img_paths, model_version, version_input, scale_input, method):
    scale_factor = scale_input
    if method == "real-esrgan":
        scale_factor /= 2
    predict_ids_paths = {}
    for img_path in img_paths:
        predict = replicate.predictions.create(
            version=model_version,
            input={"img": open(img_path, "rb"), "version": version_input, "scale": scale_factor}
        )
        predict_ids_paths[predict.id] = img_path
    return predict_ids_paths

def wait_model_predicts(client, predict_ids_paths, old_root, new_root):
    succeeded_path_url = []
    failed_paths = []
    while predict_ids_paths:
        finished_predict = None
        failed_predict = None
        for id, _ in predict_ids_paths.items():
            prediction = client.predictions.get(id)
            if prediction.status == "succeeded":
                finished_predict = prediction
                break
            elif prediction.status == "failed":
                failed_predict = prediction
                break
        if finished_predict:
            img_path = predict_ids_paths[finished_predict.id]
            img_url = finished_predict.output
            new_relpath = build_relpath(img_path, old_root, new_root)
            download_img_from_url(img_url, new_relpath)
            succeeded_path_url.append((img_path, img_url))
            del predict_ids_paths[finished_predict.id]
        elif failed_predict:
            img_path = predict_ids_paths[failed_predict.id]
            failed_paths.append(img_path)
            del predict_ids_paths[failed_predict.id]
    return succeeded_path_url, failed_paths

def ia_upscale_imgs(img_paths, old_root, new_root, factor, method):
    ia_models_dict = {
        "gfpgan": {
            "model": "tencentarc/gfpgan",
            "model_version": "9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
            "version": "v1.4"
        },
        "real-esrgan": {
            "model": "xinntao/realesrgan",
            "model_version": "1b976a4d456ed9e4d1a846597b7614e79eadad3032e9124fa63859db0fd59b56",
            "version": "General - v3"
        }
    }

    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    model = replicate.models.get(ia_models_dict[method]["model"])
    model_version = model.versions.get(ia_models_dict[method]["model_version"])
    v = ia_models_dict[method]["version"]

    imgs_to_predict = img_paths
    while True:
        print("Starting Upscale...")
        start_t = time()
        predict_ids_paths = start_model_predicts(imgs_to_predict, model_version, v, factor, method)
        succeeded_path_url, failed_paths = wait_model_predicts(client, predict_ids_paths, old_root, new_root)
        end_t = time()
        print("Upscale finished.")
        print("Elapsed time: {:.6f} seconds".format(end_t - start_t))
        input()
        if failed_paths:
            print(f"Failed to Upscale:")
            for f in failed_paths:
                print(f)
            c = input("Would you like to try again? (y|Y): ")
            if c in ["y", "Y"]:
                imgs_to_predict = failed_paths
                continue
        break

def upscale_imgs(img_paths, root, factor, method):
    new_root = ".\\upscaler_out"
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    new_root = os.path.join(new_root, method)
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    if method in ["bilinear", "bicubic", "lanczos"]:
        cv_upscale_imgs(img_paths, root, new_root, factor, method)
    if method in ["gfpgan", "real-esrgan"]:
        ia_upscale_imgs(img_paths, root, new_root, factor, method)
