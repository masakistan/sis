from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from glob import glob
import tensorflow as tf
import os, sys

if __name__ == '__main__':
    model_name = sys.argv[1]
    fe = FeatureExtractor(model_name)


    #for img_path in sorted(Path("./static/img").glob("*.jpg")):
    items = glob("./static/img/**/*.jpg", recursive=True)
    for idx, img_path in enumerate(sorted(items)):
        img_path_str = img_path
        img_path = Path(img_path)
        print("{}/{} - {}".format(idx, len(items), img_path))
        img_path_no_ext = os.path.splitext(img_path)[0]
        feature_path = "{}_{}.npy".format(img_path_no_ext, model_name)

        if os.path.exists(feature_path):
            continue

        with tf.device("/gpu:0"):
            feature = fe.extract(img=Image.open(img_path))
        
        #feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
