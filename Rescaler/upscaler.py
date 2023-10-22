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

def start_gfpgan_predicts(img_paths, model_version, version_input, scale_input):
    predict_ids_paths = {}
    scale_input = 2*scale_input # requiered due to API bug
    for img_path in img_paths:
        predict = replicate.predictions.create(
            version=model_version,
            input={"img": open(img_path, "rb"), "version": version_input, "scale": scale_input}
        )
        predict_ids_paths[predict.id] = img_path
    return predict_ids_paths

def gfpgan_upscale_imgs(img_paths, old_root, new_root, factor):
    print("Starting Upscale...")
    start_t = time()

    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    model = replicate.models.get("tencentarc/gfpgan")
    model_version = model.versions.get("9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3")
    v = "v1.4"

    predict_ids_paths = start_gfpgan_predicts(img_paths, model_version, v, factor)

    failed_upscale = []
    finished_predict = None
    failed_predict = None
    while True:
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
            new_relpath = build_relpath(img_path, old_root, new_root)
            save_url_img(finished_predict.output, new_relpath)
            del predict_ids_paths[finished_predict.id]
            finished_predict = None
        elif failed_predict:
            img_path = predict_ids_paths[failed_predict.id]
            failed_upscale.append(img_path)
            del predict_ids_paths[failed_predict.id]
            failed_predict = None
        if not predict_ids_paths:
            break
        sleep(2)

    end_t = time()
    print("Upscale finished.")
    print("Elapsed time: {:.6f} seconds".format(end_t - start_t))
    if failed_upscale:
        print(f"Failed to Upscale:")
        for f in failed_upscale:
            print(f)

def upscale_imgs(img_paths, root, factor, method):
    new_root = ".\\upscaler_out"
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    new_root = os.path.join(new_root, method)
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    if method in ["bilinear", "bicubic", "lanczos"]:
        cv_upscale_imgs(img_paths, root, new_root, factor, method)
    if method == "gfpgan":
        gfpgan_upscale_imgs(img_paths, root, new_root, factor)
