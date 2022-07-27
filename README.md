# Convolutional-Neutral-Network

This project is to demonstrate how CNNs can be use for building models to classify objects in custom dataset.
For this, we select cat images and ask to classify between cat breeds of
 - 'MaineCoon'
 - 'PersianCat'
 - 'AmericanShorthair'
 - 'SiameseCat'
The images are downloaded from google images using the `download_cat_images.py` script. We try to download about 350 images for each class. 
Then the data is cleaned (some non-images maybe downloaded) using `clean_data.py` script.

The training is done in `train.py` script. 
 - Use <b>Efficientnet-b0</b> model as the base model followed by a dense layer that has 4 units (1 for each class)
 - Use `Cross-Entropy` loss to train
 - Optimizer is `Adam` with `0.0001` learning rate

Once model is trained, we should got two plots `loss.png` and `accuracy.png` that has the loss and accuracy plots for training and validation.

We can then test the model on the held out dataset by running `test.py` script. This will generate a `test_confusion_matrix.png` which shows the 
confusion matrix on the test data.
