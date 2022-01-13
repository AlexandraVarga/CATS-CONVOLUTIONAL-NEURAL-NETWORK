## Importing the libraries
import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import sklearn.metrics as metrics
import seaborn as sns

## Hyperparams ##

LEARNING_RATE = 0.0001 # learning rate to be used by the model
BATCH_SIZE = 4
EPOCHS = 5
CLASSES = ['MaineCoon', 'PersianCat', 'AmericanShorthair', 'SiameseCat'] # 
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224)


"""
Function: build_model
Description: Builds the model by using the pretrained model and adding the dropout layer and the dense layer
"""
def build_model(NUM_CLASSES, LEARNING_RATE):

    ## Input Layer ##

    input_layer = keras.layers.Input(shape=(224, 224, 3))

    ## EfficientNet B0 pretrain
    model = hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1', trainable=False)

    ## Dropout layer
    dropout = keras.layers.Dropout(rate=0.2)

    ## Dense layer
    dense = keras.layers.Dense(NUM_CLASSES, activation='softmax')

    ## Build model by adding the pretrained model and the new layers
    model = keras.Sequential([
        input_layer,
        model,
        dropout,
        dense
    ])

    ## Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = build_model(NUM_CLASSES, LEARNING_RATE)

"""
Function : load_images
Description: Loads the images from the directory

"""
def load_dataset(directory, batch_size, image_size):

    ## ImageDataGenerator is used to perform data augmentation
    ## We use the following augmentation techniques to increase dataset size:
    ## - Horizontal flip: Flip the image horizontally, ie left to right
    ## - Rotate: Rotate the image by a random angle between -20 and 20 degrees
    ## - Zoom: Zoom in or out by a random factor between 0.8 and 1.2 times
    ## - Shift: Shift the image by a random horizontal and vertical offset of 20% of the image width or height
    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_dataset = train_data_gen.flow_from_directory(
        directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_dataset = train_data_gen.flow_from_directory(
        directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_dataset, validation_dataset

## Load the dataset
train_dataset, validation_dataset = load_dataset('data/train', BATCH_SIZE, IMG_SIZE)


"""
Function: train
Description: Trains the model, saves the model and plots the training and validation loss and accuracy

"""

def train(model, train_dataset, validation_dataset, callbacks, epochs):

    history = model.fit_generator(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        verbose=1,
        callbacks=callbacks
    )

    return history

### Callbacks that will be called while training the model
## - ModelCheckpoint: Saves the model after every epoch
## - TensorBoard: Visualize the training and validation on browser

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='models/model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

history = train(model, train_dataset, validation_dataset, callbacks=callbacks, epochs=EPOCHS)

## Plot the training and validation loss and accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png')