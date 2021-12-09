import os
import glob
import numpy as np
from PIL import Image
import cv2
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression


# Flask utils
from flask import Flask, redirect,url_for,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

#model path
proto = 'model/face_mask_detection.prototxt'
model = 'model/face_mask_detection.caffemodel'

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0 , 0))


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]









def inference(net, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, chinese=False):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    cnt=0
    res=''
    tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
    for idx in keep_idxs:
        cnt+=1
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        res = id2class[class_id]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)           
            cv2.putText(image, "%d  %s: %.2f" % (cnt,id2class[class_id], conf), (xmin + 2, ymin - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,colors[class_id])
    return image,cnt,res




def model_predict(img_path):
    Net = cv2.dnn.readNet(model, proto)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result,cnt,res = inference(Net, img, target_shape=(260, 260))
    cv2.imwrite('static/uploads/res.jpg',result[:,:,::-1])

    if cnt==1:

        


        return res
    
    return 'Try Again'
    


@app.route('/',methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        image = Image.open(f)
        rgb_im = image.convert('RGB')
        rgb_im.save('static/uploads/read.jpg')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'static','uploads/read.jpg')
        result = model_predict(file_path)
      

        return result

    return None


    

if __name__ == '__main__':

    #new
    #app.run(host="192.168.29.186", port=800, debug=False,ssl_context=('cert.pem', 'key.pem'))

    app.run(debug=True)
