from flask import Flask, render_template, request, redirect
from flask_api import status
from model import ModelSegmentationTumor
import os 


allowed_extensions = ['jpg', 'jpeg', 'png']

#  Intialisation de la Flask App
app = Flask(__name__)

# Définition de notre objet de la classe ModelSegmentationTumor()
model_seg_tumor = ModelSegmentationTumor()

# Sur la route initiale ('/'), l'utilisateur sera redirigé vers la route ('/home')
@app.route('/')
def default():
    return redirect('/home')

# La route ('/home') présentera la page HTML contenue dans home.html
@app.route('/home', methods=["GET"])
def home():
    return render_template('home.html')

# L'utilisateur va rentrer des informations sur cette page (l'image sur laquelle il utilisera le modèle), on utilise la méthode GET
# On va récupérer la prédiction ailleurs pour la transmettre à l'utilisateur, on rajoute la méthode POST
@app.route('/prediction', methods=["POST", "GET"])
def pred():

    if request.method != "POST":
        return render_template("pred.html")
    
    file = request.files['image']
    extension = file.filename.split('.')[-1]
                                         
    if extension.lower() not in allowed_extensions:
        app.logger.error('Bad Input format')
        return 'Bad input format', status.HTTP_400_BAD_REQUEST
    
    filepath = './tmp/' + os.path.basename(file.filename)
    file.save(filepath)

    # prediction
    img, output = model_seg_tumor.predict(filepath)
    filepath_output = model_seg_tumor.show(img, output)

    return render_template("pred.html", output = filepath_output)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 5000, debug = True)





