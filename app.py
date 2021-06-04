import os
import glob
import numpy as np
from PIL import Image
import cv2
import keras
import mediapipe as mp

# Flask utils
from flask import Flask, redirect,url_for,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

path = 'model/mask.h5'
proto = 'model/deploy.prototxt'
resNet = 'model/res10_300x300_ssd_iter_140000.caffemodel'

model = keras.models.load_model(path)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}


def check(X,Y,img):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                X.append(cx)
                Y.append(cy)




def model_predict(img,model):
    net  = cv2.dnn.readNetFromCaffe(proto, resNet)

    image = cv2.imread(img)
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    count = 0
    X=[]
    Y=[]

    res='Try Again and capture a bright photo'

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            count+=1


            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            check(X,Y,image)
            crop = image[startY:endY,startX:endX]
            crop = cv2.resize(crop,(224,224))
            crop = np.reshape(crop,[1,224,224,3])/255.0
            result = model.predict(crop)
            

            

            if np.size(X)==0 and np.size(Y)==0:
               
                res = mask_label[result.argmax()]

                
            else:
                res =  'Remove your hand from face.'
                              

            cv2.rectangle(image, (startX, startY), (endX, endY),dist_label[result.argmax()], 2)
            cv2.putText(image,mask_label[result.argmax()], (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2) 
    
    if count>1:
        return "Try Again"
        
    cv2.imwrite('static/uploads/res.jpg',image)
    return res


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
        result = model_predict(file_path,model)
      

        return result

    return None


    

if __name__ == '__main__':

    
    #app.run(host="192.168.29.186", port=800, debug=False,ssl_context=('cert.pem', 'key.pem'))

    app.run()
