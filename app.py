import os
from flask import Flask, request, jsonify
#  To load the image from url
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
# To cartoonify
from model.model import cartoonify
# Start making Flask object
app = Flask(__name__)


@app.route('/apiv1', methods=['POST'])
def convert_image():

    # Load image
    try:
        url = request.args['url']
    except:
        return jsonify('No url specified')
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    # # Convert image to CV2 to use it on model
    # open_cv_image = np.array(img) 
    # open_cv_image = open_cv_image[:, :, ::-1].copy() 

    # Cartoonify
    out = cartoonify(img)
    print('output', out)
    # Prepare answer
    answer = {'img_bits': list(out.flatten().tolist()),
              'dim': out.shape}
    return jsonify(answer)




if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))    