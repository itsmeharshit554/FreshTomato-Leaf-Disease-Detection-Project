from flask import Flask, render_template,request,send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired,FileAllowed
from wtforms import SubmitField
import tensorflow as tf
import os
import cv2
import numpy as np
from keras.preprocessing import image
import keras.utils as image
from tensorflow import keras
from keras.models import load_model
from werkzeug.utils import secure_filename
def predic_num(file_add):
    load_mod=load_model("final_tomato.h5")
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
    image_path =file_add

    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(228, 228))
    x = image.img_to_array(img)
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
        disease=print("Target Spot")
        prevention="To manage target spot on tomatoes, regular application of fungicides is preferred."
        precautions="Inspect the seedlings for target spot symptoms before transplanting. Manage weeds, which may serve as alternate hosts, and avoid the use of overhead irrigation as a precaution."
        sugg = f"{prevention}\n{precaution}"

    elif x == 3:
        disease = "Leaf Mold"
        prevention = "Maintain adequate spacing between plants. Keep plants far enough away from walls and fences for good air circulation."
        precaution = "Apply fungicides when symptoms first appear to reduce the spread of the leaf mold fungus."
        sugg = f"{prevention}\n{precaution}"

    elif x == 8:
        disease = "Mosaic Virus"
        prevention = "Avoid handling plants (plant seed rather than transplants). Remove diseased plants, control weeds, and rotate crops."
        precaution = "Use certified virus-free seeds when planting to reduce the risk of mosaic virus infections."
        sugg = f"{prevention}\n{precaution}"
        
    elif x == 2:
        disease = "Late Blight"
        prevention = "Spraying fungicides is the most effective way to prevent late blight."
        precaution = "Remove any volunteer tomato and potato plants, and any wild nightshades, from the garden."
        sugg = f"{prevention}\n{precaution}"
    elif x == 4:
        disease = "Septoria Leaf Spot"
        prevention = "Eliminate the initial source of infection by removing infected plant debris and weeds."
        precaution = "Avoid overhead watering or water early in the day to allow leaves to dry more quickly."
        sugg = f"{prevention}\n{precaution}"
    elif x == 7:
        disease = "Yellow Leaf Curl Virus"
        prevention = "Intercrop with rows of non-susceptible plants such as squash and cucumber."
        precaution = "Plant early to avoid peak populations of the whitefly."
        sugg = f"{prevention}\n{precaution}"
    elif x == 1:
        disease = "Early Blight"
        prevention = "Maintain optimum growing conditions, including proper fertilization, irrigation, and pest management."
        precaution = "Properly fertilize, irrigate, and manage other pests."
        sugg = f"{prevention}\n{precaution}"
    elif x == 0:
        disease = "Bacterial Spot"
        prevention = "Avoid handling plants when they are wet."
        precaution = "Use pathogen-free seed as the first step in disease management."
        sugg = f"{prevention}\n{precaution}"
    elif x == 5:
        disease = "Two-Spotted Spider Mite"
        prevention = "Keep production areas free of weeds."
        precaution = "Water plants thoroughly before spraying pesticides for spider mites."
        sugg = f"{prevention}\n{precaution}"
    else:
        disease = "Healty"
        sugg="Your Plant is healthy"
    
    return disease,sugg


app=Flask(__name__)
app.config["SECRET_KEY"]="noice123"
ALLOWED_EXTENSIONS= set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = "/d/Normal Stuff/Secret Project Don't Open/I Said don't open it/The consequences will not be right OK/So you didn't listen/pp/Secret Project/Leaf Detection Website/static/uploads"
photos=UploadSet('photos',IMAGES)

# configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo=FileField(
        validators=[
            FileAllowed(photos,"Only images are allowed"),
            FileRequired('File field should not be empty')
        ]
    )
    submit=SubmitField("Upload")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=['GET','POST'])
def index():
    return render_template(('home.html'))

    

@app.route("/upload/<filename>",methods=['POST'])
    
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file=form.user_image.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            # Perform conditions and alterations here
            plant_disease, suggest = predic_num(file_path)

            # Return strings and image path to the HTML file
            return render_template('home.html', dis=plant_disease, sug=suggest, img_path=f'/uploads/{filename}')

if __name__ == '__main__':
    app.run(debug=True)