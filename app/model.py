import json
import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, name: str):
        self.name = name
        self.path = f"models/{name}/"
        self.labels = json.load(open(self.path+'labels.json', 'r'))
        self.params = json.load(open(self.path+'params.json', 'r'))
        self.IMG_HEIGHT = self.params.get('IMG_HEIGHT', 224)
        self.IMG_WIDTH = self.params.get('IMG_WIDTH', 224)
        self.OUTPUT = self.params.get('OUTPUT', 3)
        self.LEGACY = self.params.get('LEGACY', True)
        if self.LEGACY:
            self.loaded_model = tf.saved_model.load(self.path)
            self.predict_fn = self.loaded_model.signatures['serving_default']
        else:
            self.loaded_model = tf.keras.models.load_model(self.path+"model.h5")
            
    def predict(self, img, k: int):
        if self.LEGACY:
            return self.predict_legacy(img, k)
        return self.predict_h5(img, k)
    
    def predict_h5(self, img, k: int):
        # Call the prediction function to get the model's prediction for your test image
        predictions = self.loaded_model.predict(img)
        
        # Extract the predicted class probabilities from the output dictionary
        predicted_probabilities = predictions[0]
        
        # Get the top 3 predicted class indices and their corresponding probabilities
        top_k_indices = np.argsort(predicted_probabilities)[::-1][:k]
        top_k_probabilities = predicted_probabilities[top_k_indices] * 100
        return [(self.labels[str(top_k_indices[i])], top_k_probabilities[i]) for i in range(len(top_k_indices))]
        
    def predict_legacy(self, img, k: int):
        # Call the prediction function to get the model's prediction for your test image
        predictions = self.predict_fn(tf.constant(img))

        # Extract the predicted class probabilities from the output dictionary
        predicted_probabilities = predictions[self.OUTPUT].numpy()[0]

        # Get the top 3 predicted class indices and their corresponding probabilities
        top_k = k
        top_k_indices = np.argsort(predicted_probabilities)[::-1][:top_k]
        top_k_probabilities = predicted_probabilities[top_k_indices] * 100
        return [(self.labels[str(top_k_indices[i])], top_k_probabilities[i]) for i in range(len(top_k_indices))]

