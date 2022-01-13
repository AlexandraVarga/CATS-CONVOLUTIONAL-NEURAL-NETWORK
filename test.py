## Importing the libraries
import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import seaborn as sns
import sklearn.metrics as metrics

CLASSES = ['MaineCoon', 'PersianCat', 'AmericanShorthair', 'SiameseCat'] 
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 4
IMG_SIZE = (224, 224)

## Test dataset load

test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.0,
    fill_mode='nearest',
)

test_dataset = test_data_gen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

## Load model

model = tf.keras.models.load_model('models/model.h5', custom_objects={'KerasLayer': hub.KerasLayer})


## Do predictions on the test set
predictions = model.predict(test_dataset)


## Get the original labels
test_labels = test_dataset.labels

## Compute confusion matrix
confusion = metrics.confusion_matrix(test_labels, predictions.argmax(1)).astype('float32')
confusion = confusion.astype("float") / confusion.sum(axis=1)

## Get heatmap
ax = sns.heatmap(
    confusion, xticklabels=CLASSES, yticklabels=CLASSES,
    cmap='Blues')

ax.figure.subplots_adjust(left = 0.3, bottom=0.4)

plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")

plt.savefig('test_confusion_matrix.png')