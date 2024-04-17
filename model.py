import numpy as np
import cv2 
import yaml
import torch
import os
import matplotlib.pyplot as plt
import string
import random


#  Implémentation de la classe qui appelera notre modèle pour prédire sur des images

class ModelSegmentationTumor():

    def __init__(self)->None:
        self.started = False

    # Initialisation
    def _set_up(self) -> None:

        with open('/workspace/model/config.yaml','r') as f:
            config = yaml.safe_load(f)

        self.size = config['size']
        # CPU
        self.device = torch.device('cpu')
        self.model = torch.load('/workspace/model/best_model_ultrasound.pth', map_location = self.device)
        self.model.to(self.device)
        self.model.eval()
        print('Model Loaded !')
        self.started = True

    #  Prédiction
    def predict(self, im_path : str):

        if not self.started :
            self._set_up()

        #  On traite l'image de la même manière que pendant l'entraînement
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        solo_test = cv2.resize(img, dsize=(self.size, self.size), interpolation = cv2.INTER_NEAREST)
        solo_test = np.expand_dims(solo_test, axis=0)
        solo_test = np.expand_dims(solo_test, axis=0) 
        # Autre méthode :  solo_test = solo_test[None, None, ...]
        solo_test = torch.Tensor(solo_test) / 255.0
        #  À ce niveau, les dimensions de solo_test sont : (1, 1, 256, 256) 

        solo_test = solo_test.to(self.device)

        out = self.model(solo_test)
        preds = torch.sigmoid(out)

        output = (preds > 0.5) * 1.0
        output = output.squeeze(0).detach().numpy()

        return img, output
        
    # Affcichage prédictions
    def show(self, img, out ):

        init_size = img.shape[:2]
        out = out[0,:,:]
        out = cv2.resize(out, dsize=(init_size[1], init_size[0]), interpolation = cv2.INTER_NEAREST)
        out = np.expand_dims(out, axis=2)

        fig, axes = plt.subplots(1,2, figsize=(12,6))

        axes[0].imshow(img, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Image origine')

        axes[1].imshow(img, cmap='gray')
        axes[1].imshow(out, alpha=0.4,  cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Prédiction ! ')

        plt.tight_layout()

        random_string = ''.join(random.choices(string.ascii_letters, k=5))
        filepath_output = random_string + '.jpg'
        plt.savefig('/workspace/static/' + filepath_output)

        return filepath_output
    
    





