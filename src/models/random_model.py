import random
import numpy as np
from models.model import Model
from tensorflow import keras
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        # very similar to transfered_model.py, the only difference is that you should randomize the weights
        # load your basic model with keras's load_model function
        # freeze the weights of the loaded model to make sure the training doesn't affect them
        # (check the number of total params, trainable params and non-trainable params in your summary generated by train_transfer.py)
        # randomize the weights of the loaded model, possibly by using _randomize_layers
        # use this model by removing the last layer, adding dense layers and an output layer
        print(f"SHAPE: {input_shape}")
        print(f"CATEGORIES: {categories_count}")
        model = keras.models.load_model(r"results\yeah")
        model.layers.pop()
        self.model = Sequential(
            [
                layers.MaxPooling2D()(model.layers[-1].output)
            ]
        )

        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        pass

    @staticmethod
    def _randomize_layers(model):
        # Your code goes here
        rand = random.randint(1, 10)
        rand_pm = random.randint(0, 1)
        weights = model.get_weights()
        for weight in weights: 
            if rand_pm == 1:
                model.set_weights(weight + rand)
        # you can write a function here to set the weights to a random value
        # use this function in _define_model to randomize the weights of your loaded model
        pass
