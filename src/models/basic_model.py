from models.model import Model
import tensorflow.keras as keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        print(f"SHAPE: {input_shape}")
        print(f"CATEGORIES: {categories_count}")
        self.model = Sequential(
            [
#                keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
#                layers.BatchNormalization(axis=-1, momentum=0.5),
#                keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
#                layers.BatchNormalization(axis=-1, momentum=0.5),
#                keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
#                layers.BatchNormalization(axis=-1, momentum=0.5),
#                keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
#                layers.BatchNormalization(axis=-1, momentum=0.5),
                #keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
#                layers.MaxPooling2D(pool_size=(1,1), padding="same"),

                keras.layers.Conv2D(filters=5, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
                layers.MaxPooling2D(pool_size=(2,2), padding="same"),
                keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
                layers.MaxPooling2D(pool_size=(2,2), padding="same"),
                keras.layers.Conv2D(filters=20, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
                layers.MaxPooling2D(pool_size=(2,2), padding="same"),

                # Flatten to 1D
                layers.Flatten(input_shape=input_shape),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu", input_shape=input_shape),
                layers.Dense(8, activation="relu", input_shape=input_shape),
                layers.Dense(4, activation="relu", input_shape=input_shape),
                layers.Dense(categories_count, activation="softmax", name="output"),
            ]
        )
        self.model.build(input_shape)
        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        

"""
keras.layers.Conv2D(3, 3, activation='relu', input_shape=input_shape),
layers.BatchNormalization(axis=1),
keras.layers.Conv2D(3, 3, activation='relu', input_shape=input_shape),
keras.layers.Conv2D(3, 3, input_shape=input_shape),
layers.BatchNormalization(axis=1),
keras.layers.Conv2D(3, 3, activation='relu', input_shape=input_shape),
layers.MaxPooling2D(pool_size=(1,1), padding="same"),
"""


# 0.5709 acc
"""
                keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
                layers.MaxPooling2D(pool_size=(2,2), padding="same"),
                keras.layers.Conv2D(filters=20, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding="same"),
                layers.MaxPooling2D(pool_size=(2,2), padding="same"),

                # Flatten to 1D
                layers.Flatten(input_shape=input_shape),
                layers.Dropout(0.2),
                layers.Dense(256, activation="relu", input_shape=input_shape),
                layers.Dense(128, activation="relu", input_shape=input_shape),
                layers.Dense(32, activation="relu", input_shape=input_shape),
                layers.Dense(categories_count, activation="softmax", name="output"),
"""