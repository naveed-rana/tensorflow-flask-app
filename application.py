from flask import Flask, render_template, redirect, request, url_for, jsonify, session, json
from flask_cors import CORS, cross_origin
from PIL import Image
import cv2
import numpy
import os
import csv
#models

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getDatabaseLink(avg_color):
    Links_list = []
    data_total = []
    files = [
        '1 - Foundation Findr output _ links - FENTY BEAUTY.csv',
        '2 - Foundation Findr output _ links - KAT VON D.csv',
        '3 - Foundation Findr output _ links - Mini KAT VON D.csv',
        '4 - Foundation Findr output _ links - Marc Jacobs Shameless.csv',
        '5 - Foundation Findr output _ links - Marc Jacobs Re(marc)able Full Cover Foundation.csv',
        '6 - Foundation Findr output _ links - MAKE UP FOREVER ULTRA HD FOUNDATION.csv',
        '7 - Foundation Findr output _ links - MAKE UP FOREVER Water Blend Face & Body FOUNDATION.csv',
        '8 - Foundation Findr output _ links - MAKE UP FOREVER Matte Velvet Skin Foundation.csv',
        '9 - Foundation Findr output _ links - BENEFIT COSMETICS SOFT BLUR FOUNDATION.csv',
        '10 - Foundation Findr output _ links - DIOR.csv'
    ]
    for file in files:
        i = 0
        rc = 0
        data_single = []
        with open("new_csv/" + file, "r") as fileReader:
            print(file)
            spamreader = csv.reader(fileReader, delimiter=',', quotechar='|')
            
            valueAdded = 0
            for row in spamreader:
                i += 1
                if i > 1:
                    if avg_color >= int(row[1]) and avg_color < int(row[2]):
                        # No of column you want to see
                        if valueAdded == 0:
                            Links_list.append(rc)
                            valueAdded = 1
                    data_single.append([row[4], row[3], row[0]])
                    rc += 1
        data_total.append(data_single)
    return Links_list, data_total

def findProducts(IMG_PATH):
    result = {}
    print("[info] Recognizing face in image")
    image = cv2.imread(IMG_PATH)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) < 1:
        print("[info] No Face found. Please retry capturing image")
        return False
    for (x,y,w,h) in faces:
        roi_color = image[y:y+h, x:x+w]
    
    print("[info] Getting best products for this face")
    imgGray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    h,w = imgGray.shape
    avg_color_per_row = numpy.average(imgGray, axis=0)
    avg_color = int(numpy.average(avg_color_per_row, axis=0))

    all_urls_index, all_data_list = getDatabaseLink(avg_color)
    result['index'] = all_urls_index
    result['data'] = all_data_list
    print("[info] Your suggestions are as follows:")
    print(all_data_list)
    return result
#end of models

#Flask App Structure
app = Flask(__name__)
app.config.from_object(__name__)

#Cors Setup
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#server route working
@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/check')
def check():
  data = findProducts("./1.jpeg")
  return jsonify(data)
#    return "Server Up and Running"




#AddFormData
@app.route('/predict', methods=['POST'])
@cross_origin()
def registerMissingReq():
    status = "not-success"
    if request.files['image']:
        image = request.files["image"]
        image.save('uploads/' + "output.jpg")
        data = findProducts('./uploads/'+"output.jpg")
        return jsonify(data)
    else:
        return status

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='8080',debug=True)