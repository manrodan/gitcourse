#!/usr/bin/env python
# coding: utf-8


#forecasting, getting bounding boxes, labels and probabilities
def forecast(meter):
    from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
    from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
    from msrest.authentication import ApiKeyCredentials
# Now there is a trained endpoint that can be used to make a prediction
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": "5cebf412b5504687bbbe08255e810e88"})
    predictor = CustomVisionPredictionClient(endpoint="https://uksouth.api.cognitive.microsoft.com/", credentials=credentials)
    
    with open(meter, mode="rb") as image_contents:
        results = predictor.detect_image(
            "b667b9c2-2604-4f86-be8e-818852441327", "Iteration8", image_contents)
   # Display the results.
        
        #for prediction in results.predictions:
         #   print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
    return results.predictions


# creating dict with values for each tag {tag:[prob,left, top, width, height]}
def dict(prediction):

    parameters={}
    numbers=[]
    num_iter=len(prediction)
    for i in range(num_iter):
        item=prediction[i]
    
        numbers.append(item.tag_name)
        
        lista=[item.probability, item.bounding_box.left,item.bounding_box.top,item.bounding_box.width,item.bounding_box.height]
        parameters.update( {i: lista})
    return parameters,numbers



#getting left box and digit
def digit(meter):
    from collections import OrderedDict 
    sort_box={}
    lefts={}
    count=0
    prediction=forecast(meter)
    parameters,numbers=dict(prediction)
    drawing_bound(meter, prediction,parameters,numbers)
    for key in parameters:
        dig=numbers[int(key)]
        new_key=parameters[key][1]
        
        if round(new_key,2) in sort_box:
            
            if lefts[round(new_key,2)]<parameters[key][0]:
                
                lefts.update({round(parameters[key][1],2):parameters[key][0]})
            
                if (parameters[key][0]>0.2)& (count<5): #probability > 0.1
                    count+=1 # not read more than 6 digits
                    sort_box.update({round(new_key,2): dig})
        else:
            lefts.update({round(parameters[key][1],2):parameters[key][0]})
            
            if (parameters[key][0]>0.2)& (count<5): #probability > 0.1
                count+=1 # not read more than 6 digits
                sort_box.update({round(new_key,2): dig})
   
    sort_box_new = OrderedDict(sorted(sort_box.items()))
    reading_tag=[]
    for i in sort_box_new:
        value=sort_box_new[i]
        reading_tag.append(int(value))
    return (reading_tag)


#drawing bounding_boxes on meter
def drawing_bound(meter,prediction,parameters,numbers):
    import numpy as np
    import math
    import cv2
    import os
    import sys
    
    from PIL import Image
    
    import math
    from scipy import ndimage
    ima=Image.open(meter)
    wh=ima.size # (width,height) tuple
    im=cv2.imread(meter)
    for key in parameters:
        prob=parameters[key][0]
        prob = float("{:.2f}".format(prob))
        x_= int(parameters[key][1]*wh[0])
        y_= int(parameters[key][2]*wh[1])
        rect_width=int(parameters[key][3]*wh[0])
        rect_height=int(parameters[key][4]*wh[1])
        cv2.rectangle(im,(x_,y_),(x_+rect_width,y_+rect_height),(255,0,0),1)
        prob_key=str(numbers[key])+str(';')+str(prob)
        cv2.putText(im, str(prob_key), (x_,y_), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        crop_img = im[y_:y_+rect_height, x_:x_+rect_width]
        name="cropped"+str(key)+".jpg"
        cv2.imwrite(name,crop_img)
    nombre='bounding_'+str(meter)
    #print('grabacion imagen',nombre)
    cv2.imwrite(nombre,im)


#getting readings for meters 
from azureml.core import Workspace, Dataset
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
subscription_id = '3d0248c8-9a35-4ccb-9c2a-b2dd017ab44b'
resource_group = 'Modelos'
workspace_name = 'ManuelModelos'
workspace = Workspace(subscription_id, resource_group, workspace_name)   
import os

import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
UPLOAD_FOLDER = 'static/uploads/'
#app = Flask(__name__, template_folder='templates')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
    if request.method == 'POST':
        f = request.files['file']
        #y_pred = digit(str(f.filename))
        #https://contadores-new.uksouth.instances.azureml.net/notebooks/Users/00022079/Forecast_one_meter.ipynb
        
        
        secure_filename(f.filename)
        f.save(secure_filename(f.filename))
        print('file uploaded successfully')
        y_pred=digit(str(f.filename))
        image=Image.open(f.filename)
        image.show()
        
        return ' Predicted Number: ' + str(y_pred)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
if __name__ == '__main__':
    print(("Flask starting server..."
        "please wait until server has fully started"))
    app.run(debug = True)
    
  
