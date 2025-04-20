from flask import Flask, flash, render_template, request, send_from_directory, redirect, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
import tensorflow as tf 
import os
import cv2
import numpy as np
from keras.preprocessing import image as keras_image  
from tensorflow import keras
from keras.models import load_model
from werkzeug.utils import secure_filename
import gdown


def download_model_from_drive():
    model_url = "https://drive.google.com/file/d/1q53qq1eXceUbpGJzf3oAoHjAKYadc3Ad/view?usp=drive_link"  # Replace FILE_ID with the actual ID from the shareable link
    model_path = "final_tomato.h5s"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, model_path, quiet=False)
    return model_path

def predic_num(file_add):
    model_url = "https://drive.google.com/uc?id=1q53qq1eXceUbpGJzf3oAoHjAKYadc3Ad"  # Direct download link
    model_path = "temp_model.h5"  # Temporary file path for the model

    # Download the model from Google Drive if not already downloaded
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, model_path, quiet=False)

    # Load the model
    load_mod = load_model(model_path)
    load_mod.summary()
    tomato_disease_mapping = {
        'Tomato___Bacterial_spot': 0,
        'Tomato___Early_blight': 1,
        'Tomato___Late_blight': 2,
        'Tomato___Leaf_Mold': 3,
        'Tomato___Septoria_leaf_spot': 4,
        'Tomato___Spider_mites Two-spotted_spider_mite': 5,
        'Tomato___Target_Spot': 6,
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7,
        'Tomato___Tomato_mosaic_virus': 8,
        'Tomato___healthy': 9
    }
    # print(tomato_disease_mapping)
    ref=dict(zip(list(tomato_disease_mapping.values()), list(tomato_disease_mapping.keys())))
    ref=list(ref.values())

    model_path = "final_tomato.h5"

    # Define the path to the input image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_add)

    

    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the input image
    # img = keras_image.load_img(image_path, target_size=(228, 228))
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return "Error", "Unable to load image"

    img = cv2.resize(img, (228, 228))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Make the prediction and get the class with the highest probability
    predictions = model.predict(x)
    class_idx = np.argmax(predictions, axis=1)
    class_names = ref
    class_name = class_names[class_idx[0]]

    # Print the predicted class
    print('Predicted class:', class_name)
    x=int(class_idx)
    print(x)
    if x == 6:
        disease = "Target Spot"
        prevention = "To manage target spot on tomatoes, regular application of fungicides is preferred."
        caution = "Inspect the seedlings for target spot symptoms before transplanting. Manage weeds, which may serve as alternate hosts, and avoid the use of overhead irrigation as a precaution."
        sugg = f"{prevention}\n{caution}"

    elif x == 3:
        disease = "Leaf Mold"
        prevention = "Maintain adequate spacing between plants. Keep plants far enough away from walls and fences for good air circulation."
        caution = "Apply fungicides when symptoms first appear to reduce the spread of the leaf mold fungus."
        sugg = f"{prevention}\n{caution}"

    elif x == 8:
        disease = "Mosaic Virus"
        prevention = "Avoid handling plants (plant seed rather than transplants). Remove diseased plants, control weeds, and rotate crops."
        caution = "Use certified virus-free seeds when planting to reduce the risk of mosaic virus infections."
        sugg = f"{prevention}\n{caution}"
        
    elif x == 2:
        disease = "Late Blight"
        prevention = "Spraying fungicides is the most effective way to prevent late blight."
        caution = "Remove any volunteer tomato and potato plants, and any wild nightshades, from the garden."
        sugg = f"{prevention}\n{caution}"
    elif x == 4:
        disease = "Septoria Leaf Spot"
        prevention = "Eliminate the initial source of infection by removing infected plant debris and weeds."
        caution = "Avoid overhead watering or water early in the day to allow leaves to dry more quickly."
        sugg = f"{prevention}\n{caution}"
    elif x == 7:
        disease = "Yellow Leaf Curl Virus"
        prevention = "Intercrop with rows of non-susceptible plants such as squash and cucumber."
        caution = "Plant early to avoid peak populations of the whitefly."
        sugg = f"{prevention}\n{caution}"
    elif x == 1:
        disease = "Early Blight"
        prevention = "Maintain optimum growing conditions, including proper fertilization, irrigation, and pest management."
        caution = "Properly fertilize, irrigate, and manage other pests."
        sugg = f"{prevention}\n{caution}"
    elif x == 0:
        disease = "Bacterial Spot"
        prevention = "Avoid handling plants when they are wet."
        caution = "Use pathogen-free seed as the first step in disease management."
        sugg = f"{prevention}\n{caution}"
    elif x == 5:
        disease = "Two-Spotted Spider Mite"
        prevention = "Keep production areas free of weeds."
        caution = "Water plants thoroughly before spraying pesticides for spider mites."
        sugg = f"{prevention}\n{caution}"
    elif x==9:
        disease = "Healty"
        sugg="Your Plant is healthy"
    
    return disease,sugg


app=Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        disease, sugg = predic_num(filename)
        print(disease)
        print(sugg) 
        return render_template('report.html', filename=filename, dis=disease,suggest=sugg )
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    disease, sugg = predic_num(filename)
    print(disease)
    print(sugg) 
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    

if __name__ == '__main__':
    app.run(debug=True)