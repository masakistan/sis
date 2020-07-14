import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from glob import glob
import os, sys

app = Flask(__name__)

# Read image features
model_name = sys.argv[1]
fe = FeatureExtractor(model_name)
features = []
img_paths = []

#for feature_path in Path("./static/feature").glob("*.npy"):
for feature_path in sorted(glob("./static/img/**/*_{}.npy".format(model_name), recursive=True)):
    features.append(np.load(feature_path))
    #img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    fp_no_ext = os.path.splitext(feature_path)[0]
    fp_no_ext = fp_no_ext[:len(fp_no_ext) - len(model_name) - 1]
    img_paths.append("{}.jpg".format(fp_no_ext))
print("loaded {} indexed images".format(len(img_paths)))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
