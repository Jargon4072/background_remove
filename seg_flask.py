# @Author: Dwivedi Chandan
# @Date:   2019-07-29T12:59:18+05:30
# @Email:  chandandwivedi795@gmail.com
# @Last modified by:   Dwivedi Chandan
# @Last modified time: 2019-07-29T18:00:55+05:30



import os
from io import BytesIO
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sys
import datetime
from flask import Flask, request, Response
import jsonpickle
import binascii
import io as StringIO
import base64


class DeepLabModel(object):
  """Class to load model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained  model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()
    image=Image.fromarray(image)
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    print("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map

def drawSegment(baseImg, matImg):
  width, height = baseImg.size
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = matImg[y,x]
                (r,g,b) = baseImg.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  #img.save(outputFilePath)
  return img


#inputFilePath = sys.argv[1]
#outputFilePath = sys.argv[2]

#if inputFilePath is None or outputFilePath is None:
 # print("Bad parameters. Please specify input file path and output file path")
 # exit()

#modelType = "mobile_net_model"
#if len(sys.argv) > 3 and sys.argv[3] == "1":
#  modelType = "xception_model"
modelType = "xception_model"
MODEL = DeepLabModel(modelType)
print('model loaded successfully : ' + modelType)

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def run_visualization(file):
  """Inferences DeepLab model and visualizes result."""
  #try:
  	#print("Trying to open : " + sys.argv[1])
  	# f = open(sys.argv[1])
  	#jpeg_str = open(file, "rb").read()
  	#orignal_im = Image.open(BytesIO(jpeg_str))
    #if(file):
        #print("Image recived! processing..........")
 # except IOError:
    #print('Cannot retrieve image. Please check your file ')
    #return

  print('running on image ...' % file)
  resized_im, seg_map = MODEL.run(file)

  # vis_segmentation(resized_im, seg_map)
  '''height,width=resized_im.size
  blank_image = np.zeros((height,width,3), np.uint8)
  blank_image[:,0:width] = (255,255,255)
  plt.imshow(blank_image)
  plt.show()
  blank_img=Image.fromarray(blank_image)
  res=drawSegment(blank_img, seg_map)'''
  res=drawSegment(resized_im, seg_map)
  return res
# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
   ''' req = request
    # convert string of image data to uint8
    imgdata = base64.b64decode(req.data)
    #imgdata=r.data
    nparr = np.fromstring(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image=img.copy()'''
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    final_img=run_visualization(image)
    img_encoded=image_to_byte_array(final_img)
    return Response(response=img_encoded, status=200,mimetype="image/jpeg")

#run_visualization(inputFilePath)

# start flask app
app.run(host="0.0.0.0", port=5000)