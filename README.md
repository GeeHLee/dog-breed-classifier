
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.
    

---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_count = 0
for human in human_files_short:
    if face_detector(human) == True:
        human_count += 1
        
dog_count = 0
for dog in dog_files_short:
    if face_detector(dog) == True:
        dog_count += 1

print('There are %d per 100 human face detecor.' % human_count)
print('There are %d per 100 dog considered as human.' % dog_count)
```

    There are 99 per 100 human face detecor.
    There are 11 per 100 dog considered as human.
    

__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__ 

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm
from IPython.display import display

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

count_human = 0
for human in human_files_short:
    if dog_detector(human) == True:
        count_human += 1

count_dog = 0
for dog in dog_files_short:
    if dog_detector(dog) == True:
        count_dog += 1
        
print('There are %d human face considered as dog.' % count_human)
print('There are %d dog detector.' % count_dog)
```

    There are 1 human face considered as dog.
    There are 100 dog detector.
    

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile     
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [01:23<00:00, 79.82it/s] 
    100%|██████████| 835/835 [00:09<00:00, 89.92it/s] 
    100%|██████████| 836/836 [00:07<00:00, 106.37it/s]
    

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


### TODO: Define your architecture.
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, padding = "same", input_shape = (224,224,3), activation="relu"))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation="relu"))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(133, activation="softmax"))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 28, 28, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 50176)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 500)               25088500  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               66633     
    =================================================================
    Total params: 25,165,677
    Trainable params: 25,165,677
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/10
    4180/6680 [=================>............] - ETA: 22:15 - loss: 4.9202 - acc: 0.0000e+ - ETA: 11:35 - loss: 9.3179 - acc: 0.0000e+ - ETA: 8:06 - loss: 8.7661 - acc: 0.0000e+00 - ETA: 6:20 - loss: 7.9521 - acc: 0.0000e+0 - ETA: 5:16 - loss: 7.3337 - acc: 0.0100    - ETA: 4:34 - loss: 6.9754 - acc: 0.016 - ETA: 4:03 - loss: 6.6769 - acc: 0.014 - ETA: 3:43 - loss: 6.4619 - acc: 0.012 - ETA: 3:25 - loss: 6.2884 - acc: 0.011 - ETA: 3:10 - loss: 6.1454 - acc: 0.015 - ETA: 2:59 - loss: 6.0423 - acc: 0.018 - ETA: 2:48 - loss: 5.9440 - acc: 0.016 - ETA: 2:39 - loss: 5.8661 - acc: 0.015 - ETA: 2:31 - loss: 5.7943 - acc: 0.017 - ETA: 2:25 - loss: 5.7360 - acc: 0.016 - ETA: 2:19 - loss: 5.6830 - acc: 0.018 - ETA: 2:14 - loss: 5.6347 - acc: 0.017 - ETA: 2:09 - loss: 5.5900 - acc: 0.016 - ETA: 2:05 - loss: 5.5559 - acc: 0.015 - ETA: 2:01 - loss: 5.5227 - acc: 0.015 - ETA: 1:58 - loss: 5.4931 - acc: 0.016 - ETA: 1:54 - loss: 5.4663 - acc: 0.015 - ETA: 1:52 - loss: 5.4404 - acc: 0.015 - ETA: 1:49 - loss: 5.4184 - acc: 0.014 - ETA: 1:47 - loss: 5.3976 - acc: 0.016 - ETA: 1:45 - loss: 5.3787 - acc: 0.015 - ETA: 1:42 - loss: 5.3598 - acc: 0.014 - ETA: 1:40 - loss: 5.3426 - acc: 0.014 - ETA: 1:39 - loss: 5.3275 - acc: 0.013 - ETA: 1:37 - loss: 5.3128 - acc: 0.013 - ETA: 1:35 - loss: 5.2987 - acc: 0.012 - ETA: 1:33 - loss: 5.2864 - acc: 0.012 - ETA: 1:32 - loss: 5.2746 - acc: 0.012 - ETA: 1:31 - loss: 5.2632 - acc: 0.011 - ETA: 1:29 - loss: 5.2527 - acc: 0.011 - ETA: 1:28 - loss: 5.2424 - acc: 0.012 - ETA: 1:27 - loss: 5.2326 - acc: 0.012 - ETA: 1:26 - loss: 5.2234 - acc: 0.013 - ETA: 1:25 - loss: 5.2146 - acc: 0.012 - ETA: 1:23 - loss: 5.2062 - acc: 0.013 - ETA: 1:22 - loss: 5.1977 - acc: 0.013 - ETA: 1:22 - loss: 5.1924 - acc: 0.013 - ETA: 1:21 - loss: 5.1858 - acc: 0.014 - ETA: 1:20 - loss: 5.1791 - acc: 0.013 - ETA: 1:19 - loss: 5.1724 - acc: 0.013 - ETA: 1:18 - loss: 5.1659 - acc: 0.013 - ETA: 1:17 - loss: 5.1602 - acc: 0.012 - ETA: 1:16 - loss: 5.1544 - acc: 0.012 - ETA: 1:16 - loss: 5.1487 - acc: 0.013 - ETA: 1:15 - loss: 5.1435 - acc: 0.013 - ETA: 1:14 - loss: 5.1378 - acc: 0.013 - ETA: 1:14 - loss: 5.1323 - acc: 0.013 - ETA: 1:13 - loss: 5.1280 - acc: 0.013 - ETA: 1:12 - loss: 5.1236 - acc: 0.013 - ETA: 1:12 - loss: 5.1190 - acc: 0.013 - ETA: 1:11 - loss: 5.1150 - acc: 0.013 - ETA: 1:10 - loss: 5.1105 - acc: 0.013 - ETA: 1:10 - loss: 5.1069 - acc: 0.013 - ETA: 1:09 - loss: 5.1031 - acc: 0.014 - ETA: 1:09 - loss: 5.0994 - acc: 0.014 - ETA: 1:08 - loss: 5.0967 - acc: 0.013 - ETA: 1:08 - loss: 5.0933 - acc: 0.013 - ETA: 1:07 - loss: 5.0902 - acc: 0.013 - ETA: 1:07 - loss: 5.0875 - acc: 0.013 - ETA: 1:06 - loss: 5.0846 - acc: 0.013 - ETA: 1:06 - loss: 5.0816 - acc: 0.012 - ETA: 1:05 - loss: 5.0782 - acc: 0.013 - ETA: 1:05 - loss: 5.0756 - acc: 0.013 - ETA: 1:04 - loss: 5.0734 - acc: 0.013 - ETA: 1:04 - loss: 5.0709 - acc: 0.012 - ETA: 1:03 - loss: 5.0684 - acc: 0.012 - ETA: 1:03 - loss: 5.0660 - acc: 0.012 - ETA: 1:02 - loss: 5.0634 - acc: 0.012 - ETA: 1:02 - loss: 5.0611 - acc: 0.012 - ETA: 1:02 - loss: 5.0587 - acc: 0.012 - ETA: 1:01 - loss: 5.0563 - acc: 0.011 - ETA: 1:01 - loss: 5.0535 - acc: 0.011 - ETA: 1:00 - loss: 5.0509 - acc: 0.011 - ETA: 1:00 - loss: 5.0482 - acc: 0.011 - ETA: 1:00 - loss: 5.0463 - acc: 0.011 - ETA: 59s - loss: 5.0448 - acc: 0.011 - ETA: 59s - loss: 5.0428 - acc: 0.01 - ETA: 58s - loss: 5.0408 - acc: 0.01 - ETA: 58s - loss: 5.0388 - acc: 0.01 - ETA: 58s - loss: 5.0365 - acc: 0.01 - ETA: 57s - loss: 5.0354 - acc: 0.01 - ETA: 57s - loss: 5.0335 - acc: 0.01 - ETA: 57s - loss: 5.0316 - acc: 0.01 - ETA: 56s - loss: 5.0305 - acc: 0.01 - ETA: 56s - loss: 5.0290 - acc: 0.01 - ETA: 56s - loss: 5.0276 - acc: 0.01 - ETA: 55s - loss: 5.0257 - acc: 0.01 - ETA: 55s - loss: 5.0240 - acc: 0.01 - ETA: 55s - loss: 5.0222 - acc: 0.01 - ETA: 54s - loss: 5.0212 - acc: 0.01 - ETA: 54s - loss: 5.0193 - acc: 0.01 - ETA: 54s - loss: 5.0181 - acc: 0.01 - ETA: 53s - loss: 5.0170 - acc: 0.01 - ETA: 53s - loss: 5.0153 - acc: 0.01 - ETA: 53s - loss: 5.0137 - acc: 0.01 - ETA: 52s - loss: 5.0118 - acc: 0.01 - ETA: 52s - loss: 5.0110 - acc: 0.01 - ETA: 52s - loss: 5.0104 - acc: 0.01 - ETA: 51s - loss: 5.0089 - acc: 0.01 - ETA: 51s - loss: 5.0077 - acc: 0.01 - ETA: 51s - loss: 5.0062 - acc: 0.01 - ETA: 50s - loss: 5.0050 - acc: 0.01 - ETA: 50s - loss: 5.0036 - acc: 0.01 - ETA: 50s - loss: 5.0038 - acc: 0.01 - ETA: 50s - loss: 5.0026 - acc: 0.01 - ETA: 49s - loss: 5.0019 - acc: 0.01 - ETA: 49s - loss: 5.0008 - acc: 0.01 - ETA: 49s - loss: 4.9996 - acc: 0.01 - ETA: 48s - loss: 4.9982 - acc: 0.01 - ETA: 48s - loss: 4.9970 - acc: 0.01 - ETA: 48s - loss: 4.9964 - acc: 0.01 - ETA: 47s - loss: 4.9954 - acc: 0.01 - ETA: 47s - loss: 4.9944 - acc: 0.01 - ETA: 47s - loss: 4.9938 - acc: 0.01 - ETA: 47s - loss: 4.9930 - acc: 0.01 - ETA: 46s - loss: 4.9924 - acc: 0.01 - ETA: 46s - loss: 4.9914 - acc: 0.01 - ETA: 46s - loss: 4.9905 - acc: 0.01 - ETA: 46s - loss: 4.9900 - acc: 0.01 - ETA: 45s - loss: 4.9890 - acc: 0.01 - ETA: 45s - loss: 4.9883 - acc: 0.01 - ETA: 45s - loss: 4.9876 - acc: 0.01 - ETA: 44s - loss: 4.9867 - acc: 0.01 - ETA: 44s - loss: 4.9857 - acc: 0.01 - ETA: 44s - loss: 4.9846 - acc: 0.01 - ETA: 44s - loss: 4.9835 - acc: 0.01 - ETA: 43s - loss: 4.9824 - acc: 0.01 - ETA: 43s - loss: 4.9821 - acc: 0.01 - ETA: 43s - loss: 4.9810 - acc: 0.01 - ETA: 43s - loss: 4.9812 - acc: 0.01 - ETA: 42s - loss: 4.9800 - acc: 0.01 - ETA: 42s - loss: 4.9798 - acc: 0.01 - ETA: 42s - loss: 4.9795 - acc: 0.01 - ETA: 42s - loss: 4.9789 - acc: 0.01 - ETA: 41s - loss: 4.9783 - acc: 0.01 - ETA: 41s - loss: 4.9778 - acc: 0.00 - ETA: 41s - loss: 4.9769 - acc: 0.00 - ETA: 40s - loss: 4.9764 - acc: 0.00 - ETA: 40s - loss: 4.9756 - acc: 0.00 - ETA: 40s - loss: 4.9747 - acc: 0.00 - ETA: 40s - loss: 4.9745 - acc: 0.00 - ETA: 39s - loss: 4.9737 - acc: 0.00 - ETA: 39s - loss: 4.9729 - acc: 0.00 - ETA: 39s - loss: 4.9722 - acc: 0.00 - ETA: 39s - loss: 4.9712 - acc: 0.01 - ETA: 39s - loss: 4.9713 - acc: 0.00 - ETA: 38s - loss: 4.9708 - acc: 0.00 - ETA: 38s - loss: 4.9701 - acc: 0.01 - ETA: 38s - loss: 4.9694 - acc: 0.01 - ETA: 38s - loss: 4.9686 - acc: 0.01 - ETA: 37s - loss: 4.9671 - acc: 0.01 - ETA: 37s - loss: 4.9676 - acc: 0.01 - ETA: 37s - loss: 4.9664 - acc: 0.01 - ETA: 37s - loss: 4.9661 - acc: 0.01 - ETA: 36s - loss: 4.9651 - acc: 0.01 - ETA: 36s - loss: 4.9647 - acc: 0.01 - ETA: 36s - loss: 4.9637 - acc: 0.01 - ETA: 36s - loss: 4.9638 - acc: 0.01 - ETA: 35s - loss: 4.9636 - acc: 0.01 - ETA: 35s - loss: 4.9629 - acc: 0.01 - ETA: 35s - loss: 4.9626 - acc: 0.01 - ETA: 35s - loss: 4.9625 - acc: 0.01 - ETA: 35s - loss: 4.9619 - acc: 0.01 - ETA: 34s - loss: 4.9613 - acc: 0.01 - ETA: 34s - loss: 4.9609 - acc: 0.01 - ETA: 34s - loss: 4.9604 - acc: 0.01 - ETA: 34s - loss: 4.9595 - acc: 0.01 - ETA: 33s - loss: 4.9594 - acc: 0.01 - ETA: 33s - loss: 4.9594 - acc: 0.01 - ETA: 33s - loss: 4.9589 - acc: 0.01 - ETA: 33s - loss: 4.9586 - acc: 0.00 - ETA: 32s - loss: 4.9582 - acc: 0.00 - ETA: 32s - loss: 4.9578 - acc: 0.00 - ETA: 32s - loss: 4.9571 - acc: 0.00 - ETA: 32s - loss: 4.9567 - acc: 0.00 - ETA: 32s - loss: 4.9564 - acc: 0.00 - ETA: 31s - loss: 4.9560 - acc: 0.00 - ETA: 31s - loss: 4.9554 - acc: 0.00 - ETA: 31s - loss: 4.9550 - acc: 0.00 - ETA: 31s - loss: 4.9544 - acc: 0.00 - ETA: 30s - loss: 4.9541 - acc: 0.00 - ETA: 30s - loss: 4.9536 - acc: 0.00 - ETA: 30s - loss: 4.9531 - acc: 0.00 - ETA: 30s - loss: 4.9524 - acc: 0.00 - ETA: 29s - loss: 4.9518 - acc: 0.01 - ETA: 29s - loss: 4.9511 - acc: 0.01 - ETA: 29s - loss: 4.9503 - acc: 0.01 - ETA: 29s - loss: 4.9498 - acc: 0.01 - ETA: 29s - loss: 4.9500 - acc: 0.01 - ETA: 28s - loss: 4.9496 - acc: 0.01 - ETA: 28s - loss: 4.9496 - acc: 0.01 - ETA: 28s - loss: 4.9492 - acc: 0.01 - ETA: 28s - loss: 4.9488 - acc: 0.01 - ETA: 27s - loss: 4.9487 - acc: 0.01 - ETA: 27s - loss: 4.9485 - acc: 0.01 - ETA: 27s - loss: 4.9482 - acc: 0.01 - ETA: 27s - loss: 4.9479 - acc: 0.01 - ETA: 27s - loss: 4.9477 - acc: 0.01 - ETA: 26s - loss: 4.9470 - acc: 0.01 - ETA: 26s - loss: 4.9468 - acc: 0.01 - ETA: 26s - loss: 4.9464 - acc: 0.01 - ETA: 26s - loss: 4.9462 - acc: 0.01 - ETA: 25s - loss: 4.9461 - acc: 0.01 - ETA: 25s - loss: 4.9458 - acc: 0.016660/6680 [============================>.] - ETA: 25s - loss: 4.9455 - acc: 0.01 - ETA: 25s - loss: 4.9454 - acc: 0.01 - ETA: 25s - loss: 4.9449 - acc: 0.01 - ETA: 24s - loss: 4.9446 - acc: 0.01 - ETA: 24s - loss: 4.9440 - acc: 0.01 - ETA: 24s - loss: 4.9443 - acc: 0.01 - ETA: 24s - loss: 4.9440 - acc: 0.01 - ETA: 24s - loss: 4.9438 - acc: 0.01 - ETA: 23s - loss: 4.9433 - acc: 0.01 - ETA: 23s - loss: 4.9430 - acc: 0.01 - ETA: 23s - loss: 4.9425 - acc: 0.01 - ETA: 23s - loss: 4.9419 - acc: 0.01 - ETA: 22s - loss: 4.9410 - acc: 0.01 - ETA: 22s - loss: 4.9411 - acc: 0.01 - ETA: 22s - loss: 4.9405 - acc: 0.01 - ETA: 22s - loss: 4.9400 - acc: 0.01 - ETA: 22s - loss: 4.9401 - acc: 0.01 - ETA: 21s - loss: 4.9394 - acc: 0.00 - ETA: 21s - loss: 4.9399 - acc: 0.00 - ETA: 21s - loss: 4.9396 - acc: 0.00 - ETA: 21s - loss: 4.9394 - acc: 0.00 - ETA: 21s - loss: 4.9391 - acc: 0.00 - ETA: 20s - loss: 4.9386 - acc: 0.01 - ETA: 20s - loss: 4.9380 - acc: 0.01 - ETA: 20s - loss: 4.9378 - acc: 0.01 - ETA: 20s - loss: 4.9377 - acc: 0.01 - ETA: 20s - loss: 4.9374 - acc: 0.01 - ETA: 19s - loss: 4.9371 - acc: 0.00 - ETA: 19s - loss: 4.9368 - acc: 0.01 - ETA: 19s - loss: 4.9367 - acc: 0.01 - ETA: 19s - loss: 4.9362 - acc: 0.01 - ETA: 18s - loss: 4.9355 - acc: 0.01 - ETA: 18s - loss: 4.9351 - acc: 0.01 - ETA: 18s - loss: 4.9347 - acc: 0.01 - ETA: 18s - loss: 4.9339 - acc: 0.01 - ETA: 18s - loss: 4.9343 - acc: 0.01 - ETA: 17s - loss: 4.9343 - acc: 0.01 - ETA: 17s - loss: 4.9339 - acc: 0.01 - ETA: 17s - loss: 4.9337 - acc: 0.01 - ETA: 17s - loss: 4.9336 - acc: 0.01 - ETA: 17s - loss: 4.9332 - acc: 0.01 - ETA: 16s - loss: 4.9332 - acc: 0.01 - ETA: 16s - loss: 4.9327 - acc: 0.01 - ETA: 16s - loss: 4.9320 - acc: 0.01 - ETA: 16s - loss: 4.9318 - acc: 0.01 - ETA: 16s - loss: 4.9313 - acc: 0.01 - ETA: 15s - loss: 4.9311 - acc: 0.01 - ETA: 15s - loss: 4.9309 - acc: 0.01 - ETA: 15s - loss: 4.9306 - acc: 0.01 - ETA: 15s - loss: 4.9300 - acc: 0.01 - ETA: 15s - loss: 4.9298 - acc: 0.01 - ETA: 14s - loss: 4.9297 - acc: 0.01 - ETA: 14s - loss: 4.9293 - acc: 0.01 - ETA: 14s - loss: 4.9293 - acc: 0.01 - ETA: 14s - loss: 4.9288 - acc: 0.01 - ETA: 13s - loss: 4.9283 - acc: 0.01 - ETA: 13s - loss: 4.9282 - acc: 0.01 - ETA: 13s - loss: 4.9274 - acc: 0.01 - ETA: 13s - loss: 4.9280 - acc: 0.01 - ETA: 13s - loss: 4.9275 - acc: 0.01 - ETA: 12s - loss: 4.9268 - acc: 0.01 - ETA: 12s - loss: 4.9263 - acc: 0.01 - ETA: 12s - loss: 4.9259 - acc: 0.01 - ETA: 12s - loss: 4.9255 - acc: 0.01 - ETA: 12s - loss: 4.9257 - acc: 0.01 - ETA: 11s - loss: 4.9252 - acc: 0.01 - ETA: 11s - loss: 4.9246 - acc: 0.01 - ETA: 11s - loss: 4.9239 - acc: 0.01 - ETA: 11s - loss: 4.9247 - acc: 0.01 - ETA: 11s - loss: 4.9243 - acc: 0.01 - ETA: 10s - loss: 4.9247 - acc: 0.01 - ETA: 10s - loss: 4.9243 - acc: 0.01 - ETA: 10s - loss: 4.9242 - acc: 0.01 - ETA: 10s - loss: 4.9241 - acc: 0.01 - ETA: 10s - loss: 4.9237 - acc: 0.01 - ETA: 9s - loss: 4.9237 - acc: 0.0112 - ETA: 9s - loss: 4.9234 - acc: 0.011 - ETA: 9s - loss: 4.9230 - acc: 0.011 - ETA: 9s - loss: 4.9229 - acc: 0.011 - ETA: 9s - loss: 4.9222 - acc: 0.011 - ETA: 8s - loss: 4.9218 - acc: 0.011 - ETA: 8s - loss: 4.9211 - acc: 0.011 - ETA: 8s - loss: 4.9208 - acc: 0.011 - ETA: 8s - loss: 4.9217 - acc: 0.011 - ETA: 8s - loss: 4.9216 - acc: 0.011 - ETA: 7s - loss: 4.9216 - acc: 0.011 - ETA: 7s - loss: 4.9214 - acc: 0.011 - ETA: 7s - loss: 4.9213 - acc: 0.011 - ETA: 7s - loss: 4.9209 - acc: 0.011 - ETA: 7s - loss: 4.9203 - acc: 0.011 - ETA: 6s - loss: 4.9200 - acc: 0.011 - ETA: 6s - loss: 4.9198 - acc: 0.011 - ETA: 6s - loss: 4.9196 - acc: 0.011 - ETA: 6s - loss: 4.9196 - acc: 0.011 - ETA: 6s - loss: 4.9196 - acc: 0.011 - ETA: 5s - loss: 4.9194 - acc: 0.011 - ETA: 5s - loss: 4.9189 - acc: 0.011 - ETA: 5s - loss: 4.9185 - acc: 0.011 - ETA: 5s - loss: 4.9177 - acc: 0.012 - ETA: 5s - loss: 4.9176 - acc: 0.012 - ETA: 4s - loss: 4.9174 - acc: 0.012 - ETA: 4s - loss: 4.9173 - acc: 0.012 - ETA: 4s - loss: 4.9169 - acc: 0.012 - ETA: 4s - loss: 4.9165 - acc: 0.012 - ETA: 4s - loss: 4.9163 - acc: 0.012 - ETA: 3s - loss: 4.9159 - acc: 0.012 - ETA: 3s - loss: 4.9154 - acc: 0.012 - ETA: 3s - loss: 4.9147 - acc: 0.012 - ETA: 3s - loss: 4.9144 - acc: 0.012 - ETA: 3s - loss: 4.9141 - acc: 0.012 - ETA: 2s - loss: 4.9138 - acc: 0.012 - ETA: 2s - loss: 4.9135 - acc: 0.012 - ETA: 2s - loss: 4.9138 - acc: 0.012 - ETA: 2s - loss: 4.9135 - acc: 0.012 - ETA: 2s - loss: 4.9133 - acc: 0.012 - ETA: 1s - loss: 4.9131 - acc: 0.012 - ETA: 1s - loss: 4.9124 - acc: 0.012 - ETA: 1s - loss: 4.9118 - acc: 0.012 - ETA: 1s - loss: 4.9112 - acc: 0.012 - ETA: 1s - loss: 4.9111 - acc: 0.012 - ETA: 0s - loss: 4.9107 - acc: 0.012 - ETA: 0s - loss: 4.9111 - acc: 0.012 - ETA: 0s - loss: 4.9106 - acc: 0.012 - ETA: 0s - loss: 4.9103 - acc: 0.0126Epoch 00001: val_loss improved from inf to 4.72612, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 71s 11ms/step - loss: 4.9097 - acc: 0.0126 - val_loss: 4.7261 - val_acc: 0.0251
    Epoch 2/10
    4300/6680 [==================>...........] - ETA: 55s - loss: 4.5888 - acc: 0.20 - ETA: 57s - loss: 4.7537 - acc: 0.10 - ETA: 58s - loss: 4.6772 - acc: 0.08 - ETA: 58s - loss: 4.7232 - acc: 0.07 - ETA: 57s - loss: 4.7271 - acc: 0.07 - ETA: 58s - loss: 4.7070 - acc: 0.06 - ETA: 58s - loss: 4.6943 - acc: 0.05 - ETA: 58s - loss: 4.6909 - acc: 0.05 - ETA: 57s - loss: 4.6700 - acc: 0.05 - ETA: 57s - loss: 4.6439 - acc: 0.05 - ETA: 57s - loss: 4.6511 - acc: 0.05 - ETA: 57s - loss: 4.6498 - acc: 0.05 - ETA: 57s - loss: 4.6456 - acc: 0.06 - ETA: 56s - loss: 4.6494 - acc: 0.05 - ETA: 56s - loss: 4.6673 - acc: 0.05 - ETA: 56s - loss: 4.6562 - acc: 0.05 - ETA: 56s - loss: 4.6625 - acc: 0.05 - ETA: 55s - loss: 4.6760 - acc: 0.04 - ETA: 55s - loss: 4.6883 - acc: 0.04 - ETA: 55s - loss: 4.6825 - acc: 0.04 - ETA: 54s - loss: 4.6776 - acc: 0.04 - ETA: 54s - loss: 4.6726 - acc: 0.04 - ETA: 54s - loss: 4.6830 - acc: 0.03 - ETA: 54s - loss: 4.6857 - acc: 0.03 - ETA: 53s - loss: 4.6868 - acc: 0.03 - ETA: 53s - loss: 4.6894 - acc: 0.03 - ETA: 53s - loss: 4.6793 - acc: 0.03 - ETA: 53s - loss: 4.6871 - acc: 0.03 - ETA: 52s - loss: 4.6854 - acc: 0.03 - ETA: 52s - loss: 4.6724 - acc: 0.03 - ETA: 52s - loss: 4.6778 - acc: 0.03 - ETA: 52s - loss: 4.6708 - acc: 0.03 - ETA: 52s - loss: 4.6772 - acc: 0.03 - ETA: 51s - loss: 4.6782 - acc: 0.03 - ETA: 51s - loss: 4.6784 - acc: 0.03 - ETA: 51s - loss: 4.6844 - acc: 0.03 - ETA: 51s - loss: 4.6848 - acc: 0.03 - ETA: 51s - loss: 4.6834 - acc: 0.03 - ETA: 51s - loss: 4.6785 - acc: 0.03 - ETA: 50s - loss: 4.6786 - acc: 0.03 - ETA: 50s - loss: 4.6804 - acc: 0.03 - ETA: 50s - loss: 4.6796 - acc: 0.03 - ETA: 50s - loss: 4.6821 - acc: 0.03 - ETA: 50s - loss: 4.6823 - acc: 0.03 - ETA: 49s - loss: 4.6788 - acc: 0.03 - ETA: 49s - loss: 4.6781 - acc: 0.03 - ETA: 49s - loss: 4.6804 - acc: 0.03 - ETA: 49s - loss: 4.6775 - acc: 0.03 - ETA: 49s - loss: 4.6775 - acc: 0.03 - ETA: 48s - loss: 4.6762 - acc: 0.03 - ETA: 48s - loss: 4.6805 - acc: 0.03 - ETA: 48s - loss: 4.6796 - acc: 0.03 - ETA: 48s - loss: 4.6769 - acc: 0.03 - ETA: 48s - loss: 4.6725 - acc: 0.03 - ETA: 47s - loss: 4.6688 - acc: 0.03 - ETA: 47s - loss: 4.6689 - acc: 0.03 - ETA: 47s - loss: 4.6670 - acc: 0.03 - ETA: 47s - loss: 4.6635 - acc: 0.03 - ETA: 47s - loss: 4.6677 - acc: 0.03 - ETA: 47s - loss: 4.6592 - acc: 0.03 - ETA: 46s - loss: 4.6579 - acc: 0.03 - ETA: 46s - loss: 4.6545 - acc: 0.03 - ETA: 46s - loss: 4.6574 - acc: 0.03 - ETA: 46s - loss: 4.6583 - acc: 0.03 - ETA: 46s - loss: 4.6566 - acc: 0.03 - ETA: 46s - loss: 4.6568 - acc: 0.03 - ETA: 45s - loss: 4.6561 - acc: 0.03 - ETA: 45s - loss: 4.6551 - acc: 0.03 - ETA: 45s - loss: 4.6553 - acc: 0.03 - ETA: 45s - loss: 4.6516 - acc: 0.03 - ETA: 45s - loss: 4.6486 - acc: 0.03 - ETA: 44s - loss: 4.6515 - acc: 0.03 - ETA: 44s - loss: 4.6515 - acc: 0.03 - ETA: 44s - loss: 4.6517 - acc: 0.03 - ETA: 44s - loss: 4.6504 - acc: 0.03 - ETA: 44s - loss: 4.6538 - acc: 0.03 - ETA: 44s - loss: 4.6521 - acc: 0.03 - ETA: 43s - loss: 4.6517 - acc: 0.03 - ETA: 43s - loss: 4.6515 - acc: 0.03 - ETA: 43s - loss: 4.6515 - acc: 0.03 - ETA: 43s - loss: 4.6513 - acc: 0.03 - ETA: 43s - loss: 4.6486 - acc: 0.03 - ETA: 42s - loss: 4.6471 - acc: 0.03 - ETA: 42s - loss: 4.6475 - acc: 0.03 - ETA: 42s - loss: 4.6475 - acc: 0.03 - ETA: 42s - loss: 4.6479 - acc: 0.03 - ETA: 42s - loss: 4.6443 - acc: 0.03 - ETA: 41s - loss: 4.6422 - acc: 0.03 - ETA: 41s - loss: 4.6430 - acc: 0.03 - ETA: 41s - loss: 4.6455 - acc: 0.03 - ETA: 41s - loss: 4.6425 - acc: 0.03 - ETA: 41s - loss: 4.6419 - acc: 0.03 - ETA: 41s - loss: 4.6410 - acc: 0.03 - ETA: 40s - loss: 4.6390 - acc: 0.03 - ETA: 40s - loss: 4.6402 - acc: 0.03 - ETA: 40s - loss: 4.6396 - acc: 0.03 - ETA: 40s - loss: 4.6400 - acc: 0.03 - ETA: 40s - loss: 4.6387 - acc: 0.03 - ETA: 40s - loss: 4.6398 - acc: 0.03 - ETA: 39s - loss: 4.6397 - acc: 0.03 - ETA: 39s - loss: 4.6405 - acc: 0.03 - ETA: 39s - loss: 4.6384 - acc: 0.03 - ETA: 39s - loss: 4.6390 - acc: 0.03 - ETA: 39s - loss: 4.6403 - acc: 0.03 - ETA: 39s - loss: 4.6415 - acc: 0.03 - ETA: 38s - loss: 4.6390 - acc: 0.03 - ETA: 38s - loss: 4.6389 - acc: 0.03 - ETA: 38s - loss: 4.6387 - acc: 0.03 - ETA: 38s - loss: 4.6373 - acc: 0.03 - ETA: 38s - loss: 4.6376 - acc: 0.03 - ETA: 38s - loss: 4.6389 - acc: 0.03 - ETA: 37s - loss: 4.6385 - acc: 0.03 - ETA: 37s - loss: 4.6397 - acc: 0.03 - ETA: 37s - loss: 4.6401 - acc: 0.03 - ETA: 37s - loss: 4.6392 - acc: 0.03 - ETA: 37s - loss: 4.6395 - acc: 0.03 - ETA: 37s - loss: 4.6403 - acc: 0.03 - ETA: 36s - loss: 4.6397 - acc: 0.03 - ETA: 36s - loss: 4.6417 - acc: 0.03 - ETA: 36s - loss: 4.6401 - acc: 0.03 - ETA: 36s - loss: 4.6423 - acc: 0.03 - ETA: 36s - loss: 4.6413 - acc: 0.03 - ETA: 36s - loss: 4.6405 - acc: 0.03 - ETA: 35s - loss: 4.6423 - acc: 0.03 - ETA: 35s - loss: 4.6412 - acc: 0.03 - ETA: 35s - loss: 4.6405 - acc: 0.03 - ETA: 35s - loss: 4.6408 - acc: 0.03 - ETA: 35s - loss: 4.6382 - acc: 0.03 - ETA: 35s - loss: 4.6361 - acc: 0.03 - ETA: 34s - loss: 4.6361 - acc: 0.03 - ETA: 34s - loss: 4.6331 - acc: 0.03 - ETA: 34s - loss: 4.6352 - acc: 0.03 - ETA: 34s - loss: 4.6353 - acc: 0.03 - ETA: 34s - loss: 4.6344 - acc: 0.03 - ETA: 34s - loss: 4.6348 - acc: 0.03 - ETA: 33s - loss: 4.6347 - acc: 0.03 - ETA: 33s - loss: 4.6345 - acc: 0.03 - ETA: 33s - loss: 4.6346 - acc: 0.03 - ETA: 33s - loss: 4.6349 - acc: 0.03 - ETA: 33s - loss: 4.6335 - acc: 0.03 - ETA: 32s - loss: 4.6332 - acc: 0.03 - ETA: 32s - loss: 4.6320 - acc: 0.03 - ETA: 32s - loss: 4.6328 - acc: 0.03 - ETA: 32s - loss: 4.6326 - acc: 0.03 - ETA: 32s - loss: 4.6309 - acc: 0.04 - ETA: 32s - loss: 4.6299 - acc: 0.04 - ETA: 31s - loss: 4.6307 - acc: 0.04 - ETA: 31s - loss: 4.6313 - acc: 0.04 - ETA: 31s - loss: 4.6315 - acc: 0.04 - ETA: 31s - loss: 4.6287 - acc: 0.04 - ETA: 31s - loss: 4.6299 - acc: 0.04 - ETA: 31s - loss: 4.6300 - acc: 0.04 - ETA: 30s - loss: 4.6286 - acc: 0.04 - ETA: 30s - loss: 4.6298 - acc: 0.04 - ETA: 30s - loss: 4.6290 - acc: 0.04 - ETA: 30s - loss: 4.6290 - acc: 0.04 - ETA: 30s - loss: 4.6291 - acc: 0.04 - ETA: 29s - loss: 4.6285 - acc: 0.04 - ETA: 29s - loss: 4.6303 - acc: 0.04 - ETA: 29s - loss: 4.6301 - acc: 0.04 - ETA: 29s - loss: 4.6287 - acc: 0.04 - ETA: 29s - loss: 4.6292 - acc: 0.04 - ETA: 29s - loss: 4.6290 - acc: 0.04 - ETA: 28s - loss: 4.6289 - acc: 0.04 - ETA: 28s - loss: 4.6296 - acc: 0.04 - ETA: 28s - loss: 4.6307 - acc: 0.04 - ETA: 28s - loss: 4.6306 - acc: 0.04 - ETA: 28s - loss: 4.6300 - acc: 0.04 - ETA: 28s - loss: 4.6305 - acc: 0.04 - ETA: 27s - loss: 4.6289 - acc: 0.04 - ETA: 27s - loss: 4.6285 - acc: 0.04 - ETA: 27s - loss: 4.6274 - acc: 0.04 - ETA: 27s - loss: 4.6279 - acc: 0.04 - ETA: 27s - loss: 4.6271 - acc: 0.04 - ETA: 27s - loss: 4.6274 - acc: 0.04 - ETA: 26s - loss: 4.6258 - acc: 0.04 - ETA: 26s - loss: 4.6259 - acc: 0.04 - ETA: 26s - loss: 4.6250 - acc: 0.04 - ETA: 26s - loss: 4.6248 - acc: 0.04 - ETA: 26s - loss: 4.6236 - acc: 0.04 - ETA: 26s - loss: 4.6217 - acc: 0.04 - ETA: 25s - loss: 4.6227 - acc: 0.04 - ETA: 25s - loss: 4.6227 - acc: 0.04 - ETA: 25s - loss: 4.6219 - acc: 0.04 - ETA: 25s - loss: 4.6204 - acc: 0.04 - ETA: 25s - loss: 4.6197 - acc: 0.04 - ETA: 25s - loss: 4.6198 - acc: 0.04 - ETA: 24s - loss: 4.6188 - acc: 0.04 - ETA: 24s - loss: 4.6167 - acc: 0.04 - ETA: 24s - loss: 4.6167 - acc: 0.04 - ETA: 24s - loss: 4.6166 - acc: 0.04 - ETA: 24s - loss: 4.6162 - acc: 0.04 - ETA: 24s - loss: 4.6161 - acc: 0.04 - ETA: 23s - loss: 4.6144 - acc: 0.04 - ETA: 23s - loss: 4.6136 - acc: 0.04 - ETA: 23s - loss: 4.6120 - acc: 0.04 - ETA: 23s - loss: 4.6103 - acc: 0.04 - ETA: 23s - loss: 4.6083 - acc: 0.04 - ETA: 23s - loss: 4.6067 - acc: 0.04 - ETA: 22s - loss: 4.6062 - acc: 0.04 - ETA: 22s - loss: 4.6066 - acc: 0.04 - ETA: 22s - loss: 4.6064 - acc: 0.04 - ETA: 22s - loss: 4.6051 - acc: 0.04 - ETA: 22s - loss: 4.6044 - acc: 0.04 - ETA: 21s - loss: 4.6033 - acc: 0.04 - ETA: 21s - loss: 4.6027 - acc: 0.04 - ETA: 21s - loss: 4.6023 - acc: 0.04 - ETA: 21s - loss: 4.6035 - acc: 0.04 - ETA: 21s - loss: 4.6024 - acc: 0.04 - ETA: 21s - loss: 4.6021 - acc: 0.04 - ETA: 20s - loss: 4.6018 - acc: 0.04 - ETA: 20s - loss: 4.6019 - acc: 0.04 - ETA: 20s - loss: 4.6013 - acc: 0.04 - ETA: 20s - loss: 4.6019 - acc: 0.04 - ETA: 20s - loss: 4.6017 - acc: 0.04236660/6680 [============================>.] - ETA: 20s - loss: 4.6021 - acc: 0.04 - ETA: 19s - loss: 4.6029 - acc: 0.04 - ETA: 19s - loss: 4.6042 - acc: 0.04 - ETA: 19s - loss: 4.6043 - acc: 0.04 - ETA: 19s - loss: 4.6039 - acc: 0.04 - ETA: 19s - loss: 4.6025 - acc: 0.04 - ETA: 19s - loss: 4.6032 - acc: 0.04 - ETA: 18s - loss: 4.6027 - acc: 0.04 - ETA: 18s - loss: 4.6030 - acc: 0.04 - ETA: 18s - loss: 4.6029 - acc: 0.04 - ETA: 18s - loss: 4.6035 - acc: 0.04 - ETA: 18s - loss: 4.6034 - acc: 0.04 - ETA: 18s - loss: 4.6031 - acc: 0.04 - ETA: 17s - loss: 4.6028 - acc: 0.04 - ETA: 17s - loss: 4.6014 - acc: 0.04 - ETA: 17s - loss: 4.6012 - acc: 0.04 - ETA: 17s - loss: 4.6010 - acc: 0.04 - ETA: 17s - loss: 4.5992 - acc: 0.04 - ETA: 17s - loss: 4.5981 - acc: 0.04 - ETA: 16s - loss: 4.5972 - acc: 0.04 - ETA: 16s - loss: 4.5957 - acc: 0.04 - ETA: 16s - loss: 4.5972 - acc: 0.04 - ETA: 16s - loss: 4.5975 - acc: 0.04 - ETA: 16s - loss: 4.5975 - acc: 0.04 - ETA: 16s - loss: 4.5971 - acc: 0.04 - ETA: 15s - loss: 4.5978 - acc: 0.04 - ETA: 15s - loss: 4.5979 - acc: 0.04 - ETA: 15s - loss: 4.5983 - acc: 0.04 - ETA: 15s - loss: 4.5983 - acc: 0.04 - ETA: 15s - loss: 4.5986 - acc: 0.04 - ETA: 14s - loss: 4.5988 - acc: 0.04 - ETA: 14s - loss: 4.5983 - acc: 0.04 - ETA: 14s - loss: 4.5973 - acc: 0.04 - ETA: 14s - loss: 4.5985 - acc: 0.04 - ETA: 14s - loss: 4.5977 - acc: 0.04 - ETA: 14s - loss: 4.5974 - acc: 0.04 - ETA: 13s - loss: 4.5970 - acc: 0.04 - ETA: 13s - loss: 4.5964 - acc: 0.04 - ETA: 13s - loss: 4.5957 - acc: 0.04 - ETA: 13s - loss: 4.5953 - acc: 0.04 - ETA: 13s - loss: 4.5951 - acc: 0.04 - ETA: 13s - loss: 4.5952 - acc: 0.04 - ETA: 12s - loss: 4.5955 - acc: 0.04 - ETA: 12s - loss: 4.5948 - acc: 0.04 - ETA: 12s - loss: 4.5942 - acc: 0.04 - ETA: 12s - loss: 4.5933 - acc: 0.04 - ETA: 12s - loss: 4.5925 - acc: 0.04 - ETA: 12s - loss: 4.5931 - acc: 0.04 - ETA: 11s - loss: 4.5935 - acc: 0.04 - ETA: 11s - loss: 4.5928 - acc: 0.04 - ETA: 11s - loss: 4.5919 - acc: 0.04 - ETA: 11s - loss: 4.5912 - acc: 0.04 - ETA: 11s - loss: 4.5905 - acc: 0.04 - ETA: 11s - loss: 4.5901 - acc: 0.04 - ETA: 10s - loss: 4.5909 - acc: 0.04 - ETA: 10s - loss: 4.5904 - acc: 0.04 - ETA: 10s - loss: 4.5905 - acc: 0.04 - ETA: 10s - loss: 4.5898 - acc: 0.04 - ETA: 10s - loss: 4.5902 - acc: 0.04 - ETA: 10s - loss: 4.5906 - acc: 0.04 - ETA: 9s - loss: 4.5897 - acc: 0.0420 - ETA: 9s - loss: 4.5903 - acc: 0.041 - ETA: 9s - loss: 4.5899 - acc: 0.042 - ETA: 9s - loss: 4.5886 - acc: 0.042 - ETA: 9s - loss: 4.5890 - acc: 0.042 - ETA: 9s - loss: 4.5890 - acc: 0.042 - ETA: 8s - loss: 4.5889 - acc: 0.042 - ETA: 8s - loss: 4.5894 - acc: 0.042 - ETA: 8s - loss: 4.5895 - acc: 0.042 - ETA: 8s - loss: 4.5886 - acc: 0.042 - ETA: 8s - loss: 4.5882 - acc: 0.042 - ETA: 7s - loss: 4.5872 - acc: 0.042 - ETA: 7s - loss: 4.5870 - acc: 0.042 - ETA: 7s - loss: 4.5870 - acc: 0.042 - ETA: 7s - loss: 4.5875 - acc: 0.042 - ETA: 7s - loss: 4.5875 - acc: 0.042 - ETA: 7s - loss: 4.5860 - acc: 0.042 - ETA: 6s - loss: 4.5849 - acc: 0.042 - ETA: 6s - loss: 4.5841 - acc: 0.042 - ETA: 6s - loss: 4.5836 - acc: 0.042 - ETA: 6s - loss: 4.5825 - acc: 0.042 - ETA: 6s - loss: 4.5815 - acc: 0.042 - ETA: 6s - loss: 4.5800 - acc: 0.043 - ETA: 5s - loss: 4.5795 - acc: 0.042 - ETA: 5s - loss: 4.5800 - acc: 0.042 - ETA: 5s - loss: 4.5795 - acc: 0.042 - ETA: 5s - loss: 4.5801 - acc: 0.042 - ETA: 5s - loss: 4.5801 - acc: 0.042 - ETA: 5s - loss: 4.5798 - acc: 0.042 - ETA: 4s - loss: 4.5792 - acc: 0.042 - ETA: 4s - loss: 4.5791 - acc: 0.042 - ETA: 4s - loss: 4.5783 - acc: 0.043 - ETA: 4s - loss: 4.5778 - acc: 0.042 - ETA: 4s - loss: 4.5775 - acc: 0.043 - ETA: 4s - loss: 4.5758 - acc: 0.043 - ETA: 3s - loss: 4.5761 - acc: 0.043 - ETA: 3s - loss: 4.5757 - acc: 0.043 - ETA: 3s - loss: 4.5748 - acc: 0.043 - ETA: 3s - loss: 4.5761 - acc: 0.043 - ETA: 3s - loss: 4.5763 - acc: 0.043 - ETA: 3s - loss: 4.5760 - acc: 0.043 - ETA: 2s - loss: 4.5757 - acc: 0.043 - ETA: 2s - loss: 4.5761 - acc: 0.043 - ETA: 2s - loss: 4.5751 - acc: 0.043 - ETA: 2s - loss: 4.5743 - acc: 0.043 - ETA: 2s - loss: 4.5741 - acc: 0.043 - ETA: 2s - loss: 4.5740 - acc: 0.043 - ETA: 1s - loss: 4.5730 - acc: 0.043 - ETA: 1s - loss: 4.5728 - acc: 0.043 - ETA: 1s - loss: 4.5729 - acc: 0.043 - ETA: 1s - loss: 4.5715 - acc: 0.043 - ETA: 1s - loss: 4.5713 - acc: 0.043 - ETA: 1s - loss: 4.5716 - acc: 0.043 - ETA: 0s - loss: 4.5704 - acc: 0.043 - ETA: 0s - loss: 4.5708 - acc: 0.043 - ETA: 0s - loss: 4.5709 - acc: 0.043 - ETA: 0s - loss: 4.5713 - acc: 0.043 - ETA: 0s - loss: 4.5711 - acc: 0.0435Epoch 00002: val_loss improved from 4.72612 to 4.42941, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 60s 9ms/step - loss: 4.5713 - acc: 0.0437 - val_loss: 4.4294 - val_acc: 0.0491
    Epoch 3/10
    4300/6680 [==================>...........] - ETA: 1:00 - loss: 4.2199 - acc: 0.050 - ETA: 59s - loss: 4.1883 - acc: 0.100 - ETA: 58s - loss: 4.2633 - acc: 0.06 - ETA: 58s - loss: 4.1919 - acc: 0.08 - ETA: 58s - loss: 4.0942 - acc: 0.13 - ETA: 57s - loss: 4.0968 - acc: 0.11 - ETA: 57s - loss: 4.1004 - acc: 0.12 - ETA: 56s - loss: 4.1540 - acc: 0.11 - ETA: 56s - loss: 4.1788 - acc: 0.10 - ETA: 56s - loss: 4.1602 - acc: 0.11 - ETA: 55s - loss: 4.1713 - acc: 0.10 - ETA: 56s - loss: 4.1718 - acc: 0.11 - ETA: 55s - loss: 4.2062 - acc: 0.11 - ETA: 55s - loss: 4.2274 - acc: 0.10 - ETA: 55s - loss: 4.2223 - acc: 0.10 - ETA: 55s - loss: 4.2245 - acc: 0.10 - ETA: 55s - loss: 4.2056 - acc: 0.10 - ETA: 55s - loss: 4.2051 - acc: 0.10 - ETA: 54s - loss: 4.1965 - acc: 0.10 - ETA: 54s - loss: 4.1936 - acc: 0.10 - ETA: 54s - loss: 4.2202 - acc: 0.09 - ETA: 54s - loss: 4.2166 - acc: 0.09 - ETA: 53s - loss: 4.2108 - acc: 0.09 - ETA: 53s - loss: 4.2144 - acc: 0.09 - ETA: 53s - loss: 4.2214 - acc: 0.09 - ETA: 53s - loss: 4.2217 - acc: 0.09 - ETA: 53s - loss: 4.2462 - acc: 0.08 - ETA: 53s - loss: 4.2531 - acc: 0.08 - ETA: 53s - loss: 4.2503 - acc: 0.08 - ETA: 52s - loss: 4.2495 - acc: 0.08 - ETA: 52s - loss: 4.2473 - acc: 0.08 - ETA: 52s - loss: 4.2547 - acc: 0.08 - ETA: 52s - loss: 4.2380 - acc: 0.08 - ETA: 52s - loss: 4.2438 - acc: 0.08 - ETA: 51s - loss: 4.2509 - acc: 0.08 - ETA: 51s - loss: 4.2422 - acc: 0.08 - ETA: 51s - loss: 4.2397 - acc: 0.08 - ETA: 51s - loss: 4.2293 - acc: 0.08 - ETA: 51s - loss: 4.2358 - acc: 0.08 - ETA: 51s - loss: 4.2331 - acc: 0.08 - ETA: 50s - loss: 4.2278 - acc: 0.08 - ETA: 50s - loss: 4.2272 - acc: 0.08 - ETA: 50s - loss: 4.2168 - acc: 0.08 - ETA: 50s - loss: 4.2200 - acc: 0.08 - ETA: 50s - loss: 4.2183 - acc: 0.08 - ETA: 50s - loss: 4.2184 - acc: 0.08 - ETA: 49s - loss: 4.2309 - acc: 0.08 - ETA: 49s - loss: 4.2260 - acc: 0.08 - ETA: 49s - loss: 4.2213 - acc: 0.08 - ETA: 49s - loss: 4.2208 - acc: 0.08 - ETA: 49s - loss: 4.2217 - acc: 0.08 - ETA: 49s - loss: 4.2145 - acc: 0.08 - ETA: 48s - loss: 4.2203 - acc: 0.08 - ETA: 48s - loss: 4.2176 - acc: 0.08 - ETA: 48s - loss: 4.2193 - acc: 0.08 - ETA: 48s - loss: 4.2187 - acc: 0.08 - ETA: 48s - loss: 4.2175 - acc: 0.08 - ETA: 48s - loss: 4.2128 - acc: 0.08 - ETA: 47s - loss: 4.2112 - acc: 0.08 - ETA: 47s - loss: 4.2160 - acc: 0.08 - ETA: 47s - loss: 4.2134 - acc: 0.08 - ETA: 47s - loss: 4.2171 - acc: 0.08 - ETA: 47s - loss: 4.2168 - acc: 0.08 - ETA: 46s - loss: 4.2189 - acc: 0.08 - ETA: 46s - loss: 4.2155 - acc: 0.08 - ETA: 46s - loss: 4.2136 - acc: 0.08 - ETA: 46s - loss: 4.2120 - acc: 0.08 - ETA: 46s - loss: 4.2103 - acc: 0.08 - ETA: 45s - loss: 4.2020 - acc: 0.08 - ETA: 45s - loss: 4.1959 - acc: 0.08 - ETA: 45s - loss: 4.1974 - acc: 0.08 - ETA: 45s - loss: 4.1982 - acc: 0.08 - ETA: 45s - loss: 4.1985 - acc: 0.08 - ETA: 45s - loss: 4.1952 - acc: 0.08 - ETA: 44s - loss: 4.1992 - acc: 0.08 - ETA: 44s - loss: 4.1967 - acc: 0.08 - ETA: 44s - loss: 4.1977 - acc: 0.08 - ETA: 44s - loss: 4.1948 - acc: 0.08 - ETA: 44s - loss: 4.1947 - acc: 0.08 - ETA: 43s - loss: 4.1938 - acc: 0.08 - ETA: 43s - loss: 4.1939 - acc: 0.08 - ETA: 43s - loss: 4.1962 - acc: 0.08 - ETA: 43s - loss: 4.1954 - acc: 0.08 - ETA: 43s - loss: 4.1980 - acc: 0.08 - ETA: 43s - loss: 4.1992 - acc: 0.08 - ETA: 42s - loss: 4.2005 - acc: 0.08 - ETA: 42s - loss: 4.1959 - acc: 0.08 - ETA: 42s - loss: 4.1923 - acc: 0.08 - ETA: 42s - loss: 4.1944 - acc: 0.08 - ETA: 42s - loss: 4.1997 - acc: 0.08 - ETA: 42s - loss: 4.2010 - acc: 0.08 - ETA: 41s - loss: 4.2009 - acc: 0.08 - ETA: 41s - loss: 4.2004 - acc: 0.08 - ETA: 41s - loss: 4.2007 - acc: 0.08 - ETA: 41s - loss: 4.1981 - acc: 0.08 - ETA: 41s - loss: 4.1953 - acc: 0.08 - ETA: 40s - loss: 4.1951 - acc: 0.08 - ETA: 40s - loss: 4.1976 - acc: 0.08 - ETA: 40s - loss: 4.1990 - acc: 0.08 - ETA: 40s - loss: 4.1941 - acc: 0.08 - ETA: 40s - loss: 4.1979 - acc: 0.08 - ETA: 40s - loss: 4.2018 - acc: 0.08 - ETA: 39s - loss: 4.2007 - acc: 0.08 - ETA: 39s - loss: 4.2037 - acc: 0.08 - ETA: 39s - loss: 4.2041 - acc: 0.08 - ETA: 39s - loss: 4.2022 - acc: 0.08 - ETA: 39s - loss: 4.2013 - acc: 0.08 - ETA: 39s - loss: 4.2020 - acc: 0.08 - ETA: 38s - loss: 4.2019 - acc: 0.08 - ETA: 38s - loss: 4.2031 - acc: 0.08 - ETA: 38s - loss: 4.2042 - acc: 0.08 - ETA: 38s - loss: 4.2050 - acc: 0.08 - ETA: 38s - loss: 4.2059 - acc: 0.08 - ETA: 37s - loss: 4.2058 - acc: 0.08 - ETA: 37s - loss: 4.2038 - acc: 0.08 - ETA: 37s - loss: 4.2034 - acc: 0.08 - ETA: 37s - loss: 4.2070 - acc: 0.08 - ETA: 37s - loss: 4.2084 - acc: 0.08 - ETA: 37s - loss: 4.2091 - acc: 0.08 - ETA: 36s - loss: 4.2070 - acc: 0.08 - ETA: 36s - loss: 4.2045 - acc: 0.08 - ETA: 36s - loss: 4.2074 - acc: 0.08 - ETA: 36s - loss: 4.2108 - acc: 0.08 - ETA: 36s - loss: 4.2127 - acc: 0.08 - ETA: 36s - loss: 4.2149 - acc: 0.08 - ETA: 35s - loss: 4.2147 - acc: 0.08 - ETA: 35s - loss: 4.2161 - acc: 0.08 - ETA: 35s - loss: 4.2161 - acc: 0.08 - ETA: 35s - loss: 4.2142 - acc: 0.08 - ETA: 35s - loss: 4.2160 - acc: 0.08 - ETA: 34s - loss: 4.2176 - acc: 0.08 - ETA: 34s - loss: 4.2156 - acc: 0.08 - ETA: 34s - loss: 4.2157 - acc: 0.08 - ETA: 34s - loss: 4.2154 - acc: 0.08 - ETA: 34s - loss: 4.2172 - acc: 0.08 - ETA: 34s - loss: 4.2182 - acc: 0.08 - ETA: 33s - loss: 4.2197 - acc: 0.08 - ETA: 33s - loss: 4.2180 - acc: 0.08 - ETA: 33s - loss: 4.2184 - acc: 0.08 - ETA: 33s - loss: 4.2185 - acc: 0.08 - ETA: 33s - loss: 4.2205 - acc: 0.08 - ETA: 32s - loss: 4.2203 - acc: 0.08 - ETA: 32s - loss: 4.2219 - acc: 0.08 - ETA: 32s - loss: 4.2221 - acc: 0.08 - ETA: 32s - loss: 4.2211 - acc: 0.08 - ETA: 32s - loss: 4.2203 - acc: 0.08 - ETA: 32s - loss: 4.2194 - acc: 0.08 - ETA: 31s - loss: 4.2196 - acc: 0.08 - ETA: 31s - loss: 4.2201 - acc: 0.08 - ETA: 31s - loss: 4.2215 - acc: 0.08 - ETA: 31s - loss: 4.2187 - acc: 0.08 - ETA: 31s - loss: 4.2173 - acc: 0.08 - ETA: 31s - loss: 4.2161 - acc: 0.08 - ETA: 30s - loss: 4.2182 - acc: 0.08 - ETA: 30s - loss: 4.2182 - acc: 0.08 - ETA: 30s - loss: 4.2153 - acc: 0.08 - ETA: 30s - loss: 4.2178 - acc: 0.08 - ETA: 30s - loss: 4.2151 - acc: 0.08 - ETA: 30s - loss: 4.2139 - acc: 0.08 - ETA: 29s - loss: 4.2103 - acc: 0.08 - ETA: 29s - loss: 4.2099 - acc: 0.08 - ETA: 29s - loss: 4.2099 - acc: 0.08 - ETA: 29s - loss: 4.2090 - acc: 0.08 - ETA: 29s - loss: 4.2083 - acc: 0.08 - ETA: 28s - loss: 4.2047 - acc: 0.08 - ETA: 28s - loss: 4.2084 - acc: 0.08 - ETA: 28s - loss: 4.2091 - acc: 0.08 - ETA: 28s - loss: 4.2101 - acc: 0.08 - ETA: 28s - loss: 4.2115 - acc: 0.08 - ETA: 28s - loss: 4.2120 - acc: 0.08 - ETA: 27s - loss: 4.2101 - acc: 0.08 - ETA: 27s - loss: 4.2094 - acc: 0.08 - ETA: 27s - loss: 4.2094 - acc: 0.08 - ETA: 27s - loss: 4.2092 - acc: 0.08 - ETA: 27s - loss: 4.2105 - acc: 0.08 - ETA: 27s - loss: 4.2109 - acc: 0.08 - ETA: 26s - loss: 4.2106 - acc: 0.08 - ETA: 26s - loss: 4.2105 - acc: 0.08 - ETA: 26s - loss: 4.2099 - acc: 0.08 - ETA: 26s - loss: 4.2099 - acc: 0.08 - ETA: 26s - loss: 4.2124 - acc: 0.08 - ETA: 26s - loss: 4.2113 - acc: 0.08 - ETA: 25s - loss: 4.2090 - acc: 0.08 - ETA: 25s - loss: 4.2070 - acc: 0.08 - ETA: 25s - loss: 4.2069 - acc: 0.08 - ETA: 25s - loss: 4.2094 - acc: 0.08 - ETA: 25s - loss: 4.2109 - acc: 0.08 - ETA: 24s - loss: 4.2115 - acc: 0.08 - ETA: 24s - loss: 4.2131 - acc: 0.08 - ETA: 24s - loss: 4.2120 - acc: 0.08 - ETA: 24s - loss: 4.2096 - acc: 0.08 - ETA: 24s - loss: 4.2111 - acc: 0.08 - ETA: 24s - loss: 4.2122 - acc: 0.08 - ETA: 23s - loss: 4.2116 - acc: 0.08 - ETA: 23s - loss: 4.2105 - acc: 0.08 - ETA: 23s - loss: 4.2070 - acc: 0.08 - ETA: 23s - loss: 4.2062 - acc: 0.08 - ETA: 23s - loss: 4.2068 - acc: 0.08 - ETA: 23s - loss: 4.2046 - acc: 0.08 - ETA: 22s - loss: 4.2042 - acc: 0.08 - ETA: 22s - loss: 4.2038 - acc: 0.08 - ETA: 22s - loss: 4.2054 - acc: 0.08 - ETA: 22s - loss: 4.2056 - acc: 0.08 - ETA: 22s - loss: 4.2031 - acc: 0.08 - ETA: 22s - loss: 4.2035 - acc: 0.08 - ETA: 21s - loss: 4.2028 - acc: 0.08 - ETA: 21s - loss: 4.2029 - acc: 0.08 - ETA: 21s - loss: 4.2042 - acc: 0.08 - ETA: 21s - loss: 4.2038 - acc: 0.08 - ETA: 21s - loss: 4.2026 - acc: 0.08 - ETA: 20s - loss: 4.2020 - acc: 0.08 - ETA: 20s - loss: 4.2010 - acc: 0.08 - ETA: 20s - loss: 4.2007 - acc: 0.08 - ETA: 20s - loss: 4.2029 - acc: 0.08 - ETA: 20s - loss: 4.2007 - acc: 0.08356660/6680 [============================>.] - ETA: 20s - loss: 4.2017 - acc: 0.08 - ETA: 19s - loss: 4.2019 - acc: 0.08 - ETA: 19s - loss: 4.2016 - acc: 0.08 - ETA: 19s - loss: 4.2007 - acc: 0.08 - ETA: 19s - loss: 4.1994 - acc: 0.08 - ETA: 19s - loss: 4.1985 - acc: 0.08 - ETA: 19s - loss: 4.1985 - acc: 0.08 - ETA: 18s - loss: 4.1961 - acc: 0.08 - ETA: 18s - loss: 4.1942 - acc: 0.08 - ETA: 18s - loss: 4.1943 - acc: 0.08 - ETA: 18s - loss: 4.1937 - acc: 0.08 - ETA: 18s - loss: 4.1935 - acc: 0.08 - ETA: 18s - loss: 4.1916 - acc: 0.08 - ETA: 17s - loss: 4.1905 - acc: 0.08 - ETA: 17s - loss: 4.1892 - acc: 0.08 - ETA: 17s - loss: 4.1888 - acc: 0.08 - ETA: 17s - loss: 4.1898 - acc: 0.08 - ETA: 17s - loss: 4.1879 - acc: 0.08 - ETA: 17s - loss: 4.1851 - acc: 0.08 - ETA: 16s - loss: 4.1838 - acc: 0.08 - ETA: 16s - loss: 4.1846 - acc: 0.08 - ETA: 16s - loss: 4.1838 - acc: 0.08 - ETA: 16s - loss: 4.1847 - acc: 0.08 - ETA: 16s - loss: 4.1853 - acc: 0.08 - ETA: 16s - loss: 4.1836 - acc: 0.08 - ETA: 15s - loss: 4.1826 - acc: 0.08 - ETA: 15s - loss: 4.1850 - acc: 0.08 - ETA: 15s - loss: 4.1847 - acc: 0.08 - ETA: 15s - loss: 4.1845 - acc: 0.08 - ETA: 15s - loss: 4.1838 - acc: 0.08 - ETA: 14s - loss: 4.1841 - acc: 0.08 - ETA: 14s - loss: 4.1846 - acc: 0.08 - ETA: 14s - loss: 4.1837 - acc: 0.08 - ETA: 14s - loss: 4.1837 - acc: 0.08 - ETA: 14s - loss: 4.1836 - acc: 0.08 - ETA: 14s - loss: 4.1844 - acc: 0.08 - ETA: 13s - loss: 4.1846 - acc: 0.08 - ETA: 13s - loss: 4.1841 - acc: 0.08 - ETA: 13s - loss: 4.1842 - acc: 0.08 - ETA: 13s - loss: 4.1838 - acc: 0.08 - ETA: 13s - loss: 4.1836 - acc: 0.08 - ETA: 13s - loss: 4.1831 - acc: 0.08 - ETA: 12s - loss: 4.1820 - acc: 0.08 - ETA: 12s - loss: 4.1827 - acc: 0.08 - ETA: 12s - loss: 4.1828 - acc: 0.08 - ETA: 12s - loss: 4.1834 - acc: 0.08 - ETA: 12s - loss: 4.1817 - acc: 0.08 - ETA: 12s - loss: 4.1813 - acc: 0.08 - ETA: 11s - loss: 4.1807 - acc: 0.08 - ETA: 11s - loss: 4.1803 - acc: 0.08 - ETA: 11s - loss: 4.1792 - acc: 0.08 - ETA: 11s - loss: 4.1795 - acc: 0.08 - ETA: 11s - loss: 4.1788 - acc: 0.08 - ETA: 11s - loss: 4.1803 - acc: 0.08 - ETA: 10s - loss: 4.1806 - acc: 0.08 - ETA: 10s - loss: 4.1810 - acc: 0.08 - ETA: 10s - loss: 4.1819 - acc: 0.08 - ETA: 10s - loss: 4.1827 - acc: 0.08 - ETA: 10s - loss: 4.1819 - acc: 0.08 - ETA: 10s - loss: 4.1815 - acc: 0.08 - ETA: 9s - loss: 4.1795 - acc: 0.0864 - ETA: 9s - loss: 4.1794 - acc: 0.086 - ETA: 9s - loss: 4.1803 - acc: 0.086 - ETA: 9s - loss: 4.1796 - acc: 0.086 - ETA: 9s - loss: 4.1793 - acc: 0.085 - ETA: 9s - loss: 4.1781 - acc: 0.085 - ETA: 8s - loss: 4.1778 - acc: 0.085 - ETA: 8s - loss: 4.1779 - acc: 0.085 - ETA: 8s - loss: 4.1797 - acc: 0.085 - ETA: 8s - loss: 4.1802 - acc: 0.085 - ETA: 8s - loss: 4.1796 - acc: 0.085 - ETA: 8s - loss: 4.1788 - acc: 0.085 - ETA: 7s - loss: 4.1792 - acc: 0.085 - ETA: 7s - loss: 4.1782 - acc: 0.085 - ETA: 7s - loss: 4.1792 - acc: 0.085 - ETA: 7s - loss: 4.1789 - acc: 0.086 - ETA: 7s - loss: 4.1788 - acc: 0.086 - ETA: 6s - loss: 4.1766 - acc: 0.086 - ETA: 6s - loss: 4.1759 - acc: 0.086 - ETA: 6s - loss: 4.1766 - acc: 0.086 - ETA: 6s - loss: 4.1765 - acc: 0.086 - ETA: 6s - loss: 4.1773 - acc: 0.086 - ETA: 6s - loss: 4.1774 - acc: 0.086 - ETA: 5s - loss: 4.1768 - acc: 0.086 - ETA: 5s - loss: 4.1772 - acc: 0.086 - ETA: 5s - loss: 4.1767 - acc: 0.085 - ETA: 5s - loss: 4.1755 - acc: 0.086 - ETA: 5s - loss: 4.1764 - acc: 0.086 - ETA: 5s - loss: 4.1773 - acc: 0.085 - ETA: 4s - loss: 4.1785 - acc: 0.085 - ETA: 4s - loss: 4.1780 - acc: 0.085 - ETA: 4s - loss: 4.1767 - acc: 0.085 - ETA: 4s - loss: 4.1760 - acc: 0.085 - ETA: 4s - loss: 4.1756 - acc: 0.085 - ETA: 4s - loss: 4.1761 - acc: 0.084 - ETA: 3s - loss: 4.1756 - acc: 0.084 - ETA: 3s - loss: 4.1765 - acc: 0.084 - ETA: 3s - loss: 4.1755 - acc: 0.084 - ETA: 3s - loss: 4.1758 - acc: 0.084 - ETA: 3s - loss: 4.1758 - acc: 0.084 - ETA: 3s - loss: 4.1758 - acc: 0.084 - ETA: 2s - loss: 4.1765 - acc: 0.084 - ETA: 2s - loss: 4.1765 - acc: 0.084 - ETA: 2s - loss: 4.1769 - acc: 0.084 - ETA: 2s - loss: 4.1763 - acc: 0.084 - ETA: 2s - loss: 4.1755 - acc: 0.084 - ETA: 2s - loss: 4.1739 - acc: 0.084 - ETA: 1s - loss: 4.1735 - acc: 0.085 - ETA: 1s - loss: 4.1741 - acc: 0.084 - ETA: 1s - loss: 4.1738 - acc: 0.084 - ETA: 1s - loss: 4.1729 - acc: 0.084 - ETA: 1s - loss: 4.1721 - acc: 0.084 - ETA: 1s - loss: 4.1725 - acc: 0.084 - ETA: 0s - loss: 4.1730 - acc: 0.084 - ETA: 0s - loss: 4.1722 - acc: 0.084 - ETA: 0s - loss: 4.1720 - acc: 0.084 - ETA: 0s - loss: 4.1711 - acc: 0.084 - ETA: 0s - loss: 4.1712 - acc: 0.0845Epoch 00003: val_loss improved from 4.42941 to 4.27077, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 60s 9ms/step - loss: 4.1723 - acc: 0.0846 - val_loss: 4.2708 - val_acc: 0.0731
    Epoch 4/10
    4300/6680 [==================>...........] - ETA: 55s - loss: 3.8081 - acc: 0.10 - ETA: 55s - loss: 3.6448 - acc: 0.15 - ETA: 55s - loss: 3.7196 - acc: 0.11 - ETA: 55s - loss: 3.6063 - acc: 0.15 - ETA: 54s - loss: 3.6740 - acc: 0.15 - ETA: 55s - loss: 3.6902 - acc: 0.14 - ETA: 55s - loss: 3.6345 - acc: 0.15 - ETA: 54s - loss: 3.6496 - acc: 0.13 - ETA: 54s - loss: 3.6737 - acc: 0.13 - ETA: 54s - loss: 3.6791 - acc: 0.13 - ETA: 54s - loss: 3.6577 - acc: 0.14 - ETA: 54s - loss: 3.7161 - acc: 0.13 - ETA: 53s - loss: 3.7237 - acc: 0.13 - ETA: 53s - loss: 3.7327 - acc: 0.13 - ETA: 53s - loss: 3.7251 - acc: 0.14 - ETA: 53s - loss: 3.7162 - acc: 0.15 - ETA: 53s - loss: 3.6866 - acc: 0.16 - ETA: 53s - loss: 3.7080 - acc: 0.16 - ETA: 53s - loss: 3.6890 - acc: 0.16 - ETA: 53s - loss: 3.6738 - acc: 0.17 - ETA: 53s - loss: 3.6680 - acc: 0.17 - ETA: 53s - loss: 3.6879 - acc: 0.17 - ETA: 52s - loss: 3.6868 - acc: 0.16 - ETA: 52s - loss: 3.7022 - acc: 0.16 - ETA: 52s - loss: 3.6865 - acc: 0.16 - ETA: 52s - loss: 3.6832 - acc: 0.16 - ETA: 52s - loss: 3.6867 - acc: 0.17 - ETA: 51s - loss: 3.6876 - acc: 0.16 - ETA: 51s - loss: 3.6912 - acc: 0.16 - ETA: 51s - loss: 3.6915 - acc: 0.16 - ETA: 51s - loss: 3.6879 - acc: 0.16 - ETA: 51s - loss: 3.6810 - acc: 0.16 - ETA: 50s - loss: 3.6690 - acc: 0.16 - ETA: 50s - loss: 3.6639 - acc: 0.16 - ETA: 50s - loss: 3.6774 - acc: 0.16 - ETA: 50s - loss: 3.6896 - acc: 0.16 - ETA: 50s - loss: 3.6740 - acc: 0.17 - ETA: 49s - loss: 3.6787 - acc: 0.16 - ETA: 49s - loss: 3.6730 - acc: 0.17 - ETA: 49s - loss: 3.6864 - acc: 0.17 - ETA: 49s - loss: 3.6797 - acc: 0.17 - ETA: 49s - loss: 3.6787 - acc: 0.17 - ETA: 49s - loss: 3.6721 - acc: 0.17 - ETA: 48s - loss: 3.6689 - acc: 0.17 - ETA: 48s - loss: 3.6651 - acc: 0.17 - ETA: 48s - loss: 3.6600 - acc: 0.17 - ETA: 48s - loss: 3.6587 - acc: 0.17 - ETA: 48s - loss: 3.6614 - acc: 0.17 - ETA: 47s - loss: 3.6699 - acc: 0.17 - ETA: 47s - loss: 3.6554 - acc: 0.18 - ETA: 47s - loss: 3.6549 - acc: 0.18 - ETA: 47s - loss: 3.6409 - acc: 0.18 - ETA: 47s - loss: 3.6355 - acc: 0.18 - ETA: 47s - loss: 3.6447 - acc: 0.18 - ETA: 46s - loss: 3.6496 - acc: 0.18 - ETA: 46s - loss: 3.6453 - acc: 0.18 - ETA: 46s - loss: 3.6494 - acc: 0.18 - ETA: 46s - loss: 3.6504 - acc: 0.18 - ETA: 46s - loss: 3.6604 - acc: 0.17 - ETA: 46s - loss: 3.6608 - acc: 0.18 - ETA: 45s - loss: 3.6660 - acc: 0.18 - ETA: 45s - loss: 3.6679 - acc: 0.18 - ETA: 45s - loss: 3.6668 - acc: 0.18 - ETA: 45s - loss: 3.6639 - acc: 0.18 - ETA: 45s - loss: 3.6662 - acc: 0.18 - ETA: 45s - loss: 3.6568 - acc: 0.18 - ETA: 44s - loss: 3.6533 - acc: 0.18 - ETA: 44s - loss: 3.6617 - acc: 0.18 - ETA: 44s - loss: 3.6731 - acc: 0.18 - ETA: 44s - loss: 3.6699 - acc: 0.18 - ETA: 44s - loss: 3.6718 - acc: 0.18 - ETA: 43s - loss: 3.6693 - acc: 0.18 - ETA: 43s - loss: 3.6714 - acc: 0.18 - ETA: 43s - loss: 3.6747 - acc: 0.18 - ETA: 43s - loss: 3.6744 - acc: 0.18 - ETA: 43s - loss: 3.6735 - acc: 0.18 - ETA: 43s - loss: 3.6800 - acc: 0.17 - ETA: 42s - loss: 3.6762 - acc: 0.18 - ETA: 42s - loss: 3.6783 - acc: 0.18 - ETA: 42s - loss: 3.6751 - acc: 0.18 - ETA: 42s - loss: 3.6741 - acc: 0.18 - ETA: 42s - loss: 3.6694 - acc: 0.18 - ETA: 42s - loss: 3.6738 - acc: 0.17 - ETA: 41s - loss: 3.6743 - acc: 0.17 - ETA: 41s - loss: 3.6741 - acc: 0.17 - ETA: 41s - loss: 3.6751 - acc: 0.17 - ETA: 41s - loss: 3.6775 - acc: 0.17 - ETA: 41s - loss: 3.6803 - acc: 0.17 - ETA: 41s - loss: 3.6744 - acc: 0.17 - ETA: 40s - loss: 3.6745 - acc: 0.18 - ETA: 40s - loss: 3.6743 - acc: 0.18 - ETA: 40s - loss: 3.6759 - acc: 0.18 - ETA: 40s - loss: 3.6769 - acc: 0.17 - ETA: 40s - loss: 3.6776 - acc: 0.17 - ETA: 40s - loss: 3.6795 - acc: 0.17 - ETA: 39s - loss: 3.6749 - acc: 0.17 - ETA: 39s - loss: 3.6726 - acc: 0.17 - ETA: 39s - loss: 3.6658 - acc: 0.18 - ETA: 39s - loss: 3.6669 - acc: 0.17 - ETA: 39s - loss: 3.6683 - acc: 0.17 - ETA: 39s - loss: 3.6629 - acc: 0.17 - ETA: 39s - loss: 3.6595 - acc: 0.17 - ETA: 38s - loss: 3.6521 - acc: 0.17 - ETA: 38s - loss: 3.6540 - acc: 0.17 - ETA: 38s - loss: 3.6503 - acc: 0.17 - ETA: 38s - loss: 3.6495 - acc: 0.17 - ETA: 38s - loss: 3.6557 - acc: 0.17 - ETA: 38s - loss: 3.6570 - acc: 0.17 - ETA: 37s - loss: 3.6576 - acc: 0.17 - ETA: 37s - loss: 3.6602 - acc: 0.17 - ETA: 37s - loss: 3.6654 - acc: 0.17 - ETA: 37s - loss: 3.6630 - acc: 0.17 - ETA: 37s - loss: 3.6636 - acc: 0.17 - ETA: 36s - loss: 3.6596 - acc: 0.17 - ETA: 36s - loss: 3.6593 - acc: 0.17 - ETA: 36s - loss: 3.6540 - acc: 0.17 - ETA: 36s - loss: 3.6553 - acc: 0.17 - ETA: 36s - loss: 3.6523 - acc: 0.17 - ETA: 36s - loss: 3.6564 - acc: 0.17 - ETA: 35s - loss: 3.6566 - acc: 0.17 - ETA: 35s - loss: 3.6593 - acc: 0.17 - ETA: 35s - loss: 3.6574 - acc: 0.17 - ETA: 35s - loss: 3.6558 - acc: 0.17 - ETA: 35s - loss: 3.6589 - acc: 0.17 - ETA: 35s - loss: 3.6602 - acc: 0.17 - ETA: 34s - loss: 3.6645 - acc: 0.17 - ETA: 34s - loss: 3.6656 - acc: 0.17 - ETA: 34s - loss: 3.6622 - acc: 0.17 - ETA: 34s - loss: 3.6629 - acc: 0.17 - ETA: 34s - loss: 3.6635 - acc: 0.17 - ETA: 34s - loss: 3.6603 - acc: 0.17 - ETA: 33s - loss: 3.6587 - acc: 0.17 - ETA: 33s - loss: 3.6576 - acc: 0.17 - ETA: 33s - loss: 3.6591 - acc: 0.17 - ETA: 33s - loss: 3.6575 - acc: 0.17 - ETA: 33s - loss: 3.6570 - acc: 0.17 - ETA: 33s - loss: 3.6572 - acc: 0.17 - ETA: 32s - loss: 3.6516 - acc: 0.17 - ETA: 32s - loss: 3.6548 - acc: 0.17 - ETA: 32s - loss: 3.6572 - acc: 0.17 - ETA: 32s - loss: 3.6573 - acc: 0.17 - ETA: 32s - loss: 3.6560 - acc: 0.17 - ETA: 32s - loss: 3.6547 - acc: 0.17 - ETA: 31s - loss: 3.6584 - acc: 0.17 - ETA: 31s - loss: 3.6570 - acc: 0.17 - ETA: 31s - loss: 3.6571 - acc: 0.17 - ETA: 31s - loss: 3.6600 - acc: 0.17 - ETA: 31s - loss: 3.6583 - acc: 0.17 - ETA: 31s - loss: 3.6617 - acc: 0.17 - ETA: 30s - loss: 3.6615 - acc: 0.17 - ETA: 30s - loss: 3.6604 - acc: 0.17 - ETA: 30s - loss: 3.6588 - acc: 0.17 - ETA: 30s - loss: 3.6579 - acc: 0.17 - ETA: 30s - loss: 3.6579 - acc: 0.17 - ETA: 30s - loss: 3.6567 - acc: 0.17 - ETA: 29s - loss: 3.6595 - acc: 0.17 - ETA: 29s - loss: 3.6618 - acc: 0.17 - ETA: 29s - loss: 3.6583 - acc: 0.17 - ETA: 29s - loss: 3.6587 - acc: 0.17 - ETA: 29s - loss: 3.6600 - acc: 0.17 - ETA: 29s - loss: 3.6593 - acc: 0.17 - ETA: 29s - loss: 3.6581 - acc: 0.17 - ETA: 28s - loss: 3.6579 - acc: 0.17 - ETA: 28s - loss: 3.6576 - acc: 0.17 - ETA: 28s - loss: 3.6605 - acc: 0.17 - ETA: 28s - loss: 3.6620 - acc: 0.17 - ETA: 28s - loss: 3.6597 - acc: 0.17 - ETA: 28s - loss: 3.6619 - acc: 0.17 - ETA: 27s - loss: 3.6597 - acc: 0.17 - ETA: 27s - loss: 3.6587 - acc: 0.17 - ETA: 27s - loss: 3.6572 - acc: 0.17 - ETA: 27s - loss: 3.6617 - acc: 0.17 - ETA: 27s - loss: 3.6614 - acc: 0.17 - ETA: 27s - loss: 3.6573 - acc: 0.17 - ETA: 26s - loss: 3.6594 - acc: 0.17 - ETA: 26s - loss: 3.6574 - acc: 0.17 - ETA: 26s - loss: 3.6561 - acc: 0.17 - ETA: 26s - loss: 3.6558 - acc: 0.17 - ETA: 26s - loss: 3.6574 - acc: 0.17 - ETA: 26s - loss: 3.6543 - acc: 0.17 - ETA: 25s - loss: 3.6536 - acc: 0.17 - ETA: 25s - loss: 3.6526 - acc: 0.17 - ETA: 25s - loss: 3.6551 - acc: 0.17 - ETA: 25s - loss: 3.6533 - acc: 0.17 - ETA: 25s - loss: 3.6559 - acc: 0.17 - ETA: 25s - loss: 3.6562 - acc: 0.17 - ETA: 24s - loss: 3.6590 - acc: 0.17 - ETA: 24s - loss: 3.6591 - acc: 0.17 - ETA: 24s - loss: 3.6577 - acc: 0.17 - ETA: 24s - loss: 3.6589 - acc: 0.17 - ETA: 24s - loss: 3.6585 - acc: 0.17 - ETA: 24s - loss: 3.6584 - acc: 0.17 - ETA: 23s - loss: 3.6596 - acc: 0.17 - ETA: 23s - loss: 3.6600 - acc: 0.17 - ETA: 23s - loss: 3.6590 - acc: 0.17 - ETA: 23s - loss: 3.6601 - acc: 0.17 - ETA: 23s - loss: 3.6610 - acc: 0.17 - ETA: 23s - loss: 3.6620 - acc: 0.17 - ETA: 22s - loss: 3.6582 - acc: 0.17 - ETA: 22s - loss: 3.6585 - acc: 0.17 - ETA: 22s - loss: 3.6577 - acc: 0.17 - ETA: 22s - loss: 3.6576 - acc: 0.17 - ETA: 22s - loss: 3.6551 - acc: 0.17 - ETA: 22s - loss: 3.6569 - acc: 0.17 - ETA: 21s - loss: 3.6581 - acc: 0.17 - ETA: 21s - loss: 3.6562 - acc: 0.17 - ETA: 21s - loss: 3.6560 - acc: 0.17 - ETA: 21s - loss: 3.6557 - acc: 0.17 - ETA: 21s - loss: 3.6540 - acc: 0.17 - ETA: 21s - loss: 3.6530 - acc: 0.17 - ETA: 20s - loss: 3.6546 - acc: 0.17 - ETA: 20s - loss: 3.6577 - acc: 0.17 - ETA: 20s - loss: 3.6597 - acc: 0.17 - ETA: 20s - loss: 3.6612 - acc: 0.17 - ETA: 20s - loss: 3.6619 - acc: 0.17336660/6680 [============================>.] - ETA: 20s - loss: 3.6628 - acc: 0.17 - ETA: 19s - loss: 3.6634 - acc: 0.17 - ETA: 19s - loss: 3.6655 - acc: 0.17 - ETA: 19s - loss: 3.6649 - acc: 0.17 - ETA: 19s - loss: 3.6633 - acc: 0.17 - ETA: 19s - loss: 3.6630 - acc: 0.17 - ETA: 19s - loss: 3.6654 - acc: 0.17 - ETA: 18s - loss: 3.6658 - acc: 0.17 - ETA: 18s - loss: 3.6702 - acc: 0.17 - ETA: 18s - loss: 3.6712 - acc: 0.17 - ETA: 18s - loss: 3.6704 - acc: 0.17 - ETA: 18s - loss: 3.6709 - acc: 0.17 - ETA: 18s - loss: 3.6726 - acc: 0.17 - ETA: 17s - loss: 3.6716 - acc: 0.17 - ETA: 17s - loss: 3.6719 - acc: 0.17 - ETA: 17s - loss: 3.6704 - acc: 0.17 - ETA: 17s - loss: 3.6692 - acc: 0.17 - ETA: 17s - loss: 3.6665 - acc: 0.17 - ETA: 17s - loss: 3.6691 - acc: 0.17 - ETA: 16s - loss: 3.6677 - acc: 0.17 - ETA: 16s - loss: 3.6672 - acc: 0.17 - ETA: 16s - loss: 3.6666 - acc: 0.17 - ETA: 16s - loss: 3.6668 - acc: 0.17 - ETA: 16s - loss: 3.6645 - acc: 0.17 - ETA: 15s - loss: 3.6648 - acc: 0.17 - ETA: 15s - loss: 3.6625 - acc: 0.17 - ETA: 15s - loss: 3.6602 - acc: 0.17 - ETA: 15s - loss: 3.6580 - acc: 0.17 - ETA: 15s - loss: 3.6603 - acc: 0.17 - ETA: 15s - loss: 3.6594 - acc: 0.17 - ETA: 14s - loss: 3.6574 - acc: 0.17 - ETA: 14s - loss: 3.6575 - acc: 0.17 - ETA: 14s - loss: 3.6572 - acc: 0.17 - ETA: 14s - loss: 3.6580 - acc: 0.17 - ETA: 14s - loss: 3.6567 - acc: 0.17 - ETA: 14s - loss: 3.6568 - acc: 0.17 - ETA: 13s - loss: 3.6584 - acc: 0.17 - ETA: 13s - loss: 3.6580 - acc: 0.17 - ETA: 13s - loss: 3.6607 - acc: 0.17 - ETA: 13s - loss: 3.6603 - acc: 0.17 - ETA: 13s - loss: 3.6588 - acc: 0.17 - ETA: 13s - loss: 3.6591 - acc: 0.17 - ETA: 12s - loss: 3.6608 - acc: 0.17 - ETA: 12s - loss: 3.6604 - acc: 0.17 - ETA: 12s - loss: 3.6601 - acc: 0.17 - ETA: 12s - loss: 3.6609 - acc: 0.17 - ETA: 12s - loss: 3.6597 - acc: 0.17 - ETA: 12s - loss: 3.6590 - acc: 0.17 - ETA: 11s - loss: 3.6583 - acc: 0.17 - ETA: 11s - loss: 3.6575 - acc: 0.17 - ETA: 11s - loss: 3.6599 - acc: 0.17 - ETA: 11s - loss: 3.6610 - acc: 0.17 - ETA: 11s - loss: 3.6608 - acc: 0.17 - ETA: 11s - loss: 3.6584 - acc: 0.17 - ETA: 10s - loss: 3.6554 - acc: 0.17 - ETA: 10s - loss: 3.6577 - acc: 0.17 - ETA: 10s - loss: 3.6584 - acc: 0.17 - ETA: 10s - loss: 3.6582 - acc: 0.17 - ETA: 10s - loss: 3.6595 - acc: 0.17 - ETA: 10s - loss: 3.6590 - acc: 0.17 - ETA: 9s - loss: 3.6603 - acc: 0.1728 - ETA: 9s - loss: 3.6582 - acc: 0.173 - ETA: 9s - loss: 3.6574 - acc: 0.173 - ETA: 9s - loss: 3.6598 - acc: 0.173 - ETA: 9s - loss: 3.6574 - acc: 0.173 - ETA: 9s - loss: 3.6553 - acc: 0.173 - ETA: 8s - loss: 3.6568 - acc: 0.173 - ETA: 8s - loss: 3.6558 - acc: 0.173 - ETA: 8s - loss: 3.6572 - acc: 0.173 - ETA: 8s - loss: 3.6570 - acc: 0.173 - ETA: 8s - loss: 3.6568 - acc: 0.172 - ETA: 8s - loss: 3.6557 - acc: 0.173 - ETA: 7s - loss: 3.6560 - acc: 0.172 - ETA: 7s - loss: 3.6527 - acc: 0.173 - ETA: 7s - loss: 3.6526 - acc: 0.173 - ETA: 7s - loss: 3.6531 - acc: 0.173 - ETA: 7s - loss: 3.6526 - acc: 0.173 - ETA: 6s - loss: 3.6531 - acc: 0.173 - ETA: 6s - loss: 3.6544 - acc: 0.173 - ETA: 6s - loss: 3.6522 - acc: 0.173 - ETA: 6s - loss: 3.6511 - acc: 0.174 - ETA: 6s - loss: 3.6517 - acc: 0.174 - ETA: 6s - loss: 3.6531 - acc: 0.173 - ETA: 5s - loss: 3.6519 - acc: 0.173 - ETA: 5s - loss: 3.6496 - acc: 0.174 - ETA: 5s - loss: 3.6509 - acc: 0.173 - ETA: 5s - loss: 3.6510 - acc: 0.174 - ETA: 5s - loss: 3.6503 - acc: 0.174 - ETA: 5s - loss: 3.6490 - acc: 0.174 - ETA: 4s - loss: 3.6482 - acc: 0.174 - ETA: 4s - loss: 3.6478 - acc: 0.174 - ETA: 4s - loss: 3.6487 - acc: 0.174 - ETA: 4s - loss: 3.6484 - acc: 0.174 - ETA: 4s - loss: 3.6464 - acc: 0.174 - ETA: 4s - loss: 3.6444 - acc: 0.174 - ETA: 3s - loss: 3.6454 - acc: 0.174 - ETA: 3s - loss: 3.6454 - acc: 0.174 - ETA: 3s - loss: 3.6454 - acc: 0.174 - ETA: 3s - loss: 3.6447 - acc: 0.174 - ETA: 3s - loss: 3.6444 - acc: 0.174 - ETA: 3s - loss: 3.6426 - acc: 0.174 - ETA: 2s - loss: 3.6421 - acc: 0.174 - ETA: 2s - loss: 3.6433 - acc: 0.174 - ETA: 2s - loss: 3.6446 - acc: 0.173 - ETA: 2s - loss: 3.6446 - acc: 0.173 - ETA: 2s - loss: 3.6448 - acc: 0.173 - ETA: 2s - loss: 3.6457 - acc: 0.173 - ETA: 1s - loss: 3.6449 - acc: 0.173 - ETA: 1s - loss: 3.6443 - acc: 0.173 - ETA: 1s - loss: 3.6433 - acc: 0.174 - ETA: 1s - loss: 3.6443 - acc: 0.174 - ETA: 1s - loss: 3.6438 - acc: 0.173 - ETA: 1s - loss: 3.6419 - acc: 0.173 - ETA: 0s - loss: 3.6408 - acc: 0.173 - ETA: 0s - loss: 3.6419 - acc: 0.173 - ETA: 0s - loss: 3.6423 - acc: 0.173 - ETA: 0s - loss: 3.6419 - acc: 0.173 - ETA: 0s - loss: 3.6428 - acc: 0.1733Epoch 00004: val_loss improved from 4.27077 to 4.26343, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 60s 9ms/step - loss: 3.6412 - acc: 0.1735 - val_loss: 4.2634 - val_acc: 0.0850
    Epoch 5/10
    4300/6680 [==================>...........] - ETA: 1:00 - loss: 3.0834 - acc: 0.350 - ETA: 59s - loss: 3.0364 - acc: 0.400 - ETA: 59s - loss: 2.9438 - acc: 0.40 - ETA: 57s - loss: 3.0033 - acc: 0.36 - ETA: 57s - loss: 2.9659 - acc: 0.34 - ETA: 56s - loss: 3.0051 - acc: 0.32 - ETA: 56s - loss: 2.9964 - acc: 0.30 - ETA: 56s - loss: 3.0400 - acc: 0.29 - ETA: 55s - loss: 3.0656 - acc: 0.28 - ETA: 55s - loss: 3.0631 - acc: 0.29 - ETA: 55s - loss: 3.0343 - acc: 0.29 - ETA: 55s - loss: 3.0189 - acc: 0.28 - ETA: 55s - loss: 3.0222 - acc: 0.28 - ETA: 54s - loss: 3.0103 - acc: 0.28 - ETA: 54s - loss: 2.9898 - acc: 0.29 - ETA: 54s - loss: 3.0220 - acc: 0.28 - ETA: 54s - loss: 3.0211 - acc: 0.28 - ETA: 54s - loss: 3.0011 - acc: 0.28 - ETA: 54s - loss: 2.9927 - acc: 0.28 - ETA: 54s - loss: 3.0221 - acc: 0.28 - ETA: 53s - loss: 3.0110 - acc: 0.28 - ETA: 53s - loss: 3.0021 - acc: 0.28 - ETA: 53s - loss: 2.9886 - acc: 0.28 - ETA: 53s - loss: 2.9717 - acc: 0.28 - ETA: 52s - loss: 2.9613 - acc: 0.28 - ETA: 52s - loss: 2.9827 - acc: 0.28 - ETA: 52s - loss: 2.9927 - acc: 0.28 - ETA: 52s - loss: 2.9922 - acc: 0.28 - ETA: 52s - loss: 2.9660 - acc: 0.29 - ETA: 51s - loss: 2.9597 - acc: 0.29 - ETA: 51s - loss: 2.9769 - acc: 0.29 - ETA: 51s - loss: 2.9869 - acc: 0.29 - ETA: 51s - loss: 2.9968 - acc: 0.29 - ETA: 51s - loss: 3.0035 - acc: 0.28 - ETA: 51s - loss: 3.0036 - acc: 0.28 - ETA: 51s - loss: 3.0019 - acc: 0.28 - ETA: 50s - loss: 3.0042 - acc: 0.28 - ETA: 50s - loss: 3.0059 - acc: 0.28 - ETA: 50s - loss: 3.0028 - acc: 0.28 - ETA: 50s - loss: 2.9923 - acc: 0.29 - ETA: 50s - loss: 2.9955 - acc: 0.29 - ETA: 50s - loss: 3.0093 - acc: 0.28 - ETA: 49s - loss: 3.0089 - acc: 0.28 - ETA: 49s - loss: 3.0052 - acc: 0.28 - ETA: 49s - loss: 2.9861 - acc: 0.28 - ETA: 49s - loss: 2.9847 - acc: 0.28 - ETA: 49s - loss: 2.9819 - acc: 0.28 - ETA: 49s - loss: 2.9802 - acc: 0.29 - ETA: 49s - loss: 2.9834 - acc: 0.28 - ETA: 48s - loss: 2.9762 - acc: 0.29 - ETA: 48s - loss: 2.9855 - acc: 0.29 - ETA: 48s - loss: 2.9809 - acc: 0.29 - ETA: 48s - loss: 2.9838 - acc: 0.29 - ETA: 48s - loss: 2.9907 - acc: 0.28 - ETA: 48s - loss: 2.9860 - acc: 0.29 - ETA: 47s - loss: 2.9982 - acc: 0.29 - ETA: 47s - loss: 2.9956 - acc: 0.29 - ETA: 47s - loss: 2.9932 - acc: 0.29 - ETA: 47s - loss: 2.9921 - acc: 0.29 - ETA: 47s - loss: 2.9846 - acc: 0.29 - ETA: 47s - loss: 2.9783 - acc: 0.29 - ETA: 46s - loss: 2.9684 - acc: 0.29 - ETA: 46s - loss: 2.9614 - acc: 0.29 - ETA: 46s - loss: 2.9608 - acc: 0.29 - ETA: 46s - loss: 2.9563 - acc: 0.30 - ETA: 46s - loss: 2.9629 - acc: 0.29 - ETA: 45s - loss: 2.9630 - acc: 0.30 - ETA: 45s - loss: 2.9634 - acc: 0.30 - ETA: 45s - loss: 2.9680 - acc: 0.30 - ETA: 45s - loss: 2.9704 - acc: 0.30 - ETA: 45s - loss: 2.9675 - acc: 0.30 - ETA: 45s - loss: 2.9637 - acc: 0.30 - ETA: 44s - loss: 2.9596 - acc: 0.30 - ETA: 44s - loss: 2.9510 - acc: 0.30 - ETA: 44s - loss: 2.9557 - acc: 0.30 - ETA: 44s - loss: 2.9547 - acc: 0.30 - ETA: 44s - loss: 2.9683 - acc: 0.30 - ETA: 44s - loss: 2.9666 - acc: 0.30 - ETA: 43s - loss: 2.9670 - acc: 0.30 - ETA: 43s - loss: 2.9807 - acc: 0.30 - ETA: 43s - loss: 2.9811 - acc: 0.30 - ETA: 43s - loss: 2.9791 - acc: 0.30 - ETA: 43s - loss: 2.9816 - acc: 0.30 - ETA: 43s - loss: 2.9868 - acc: 0.30 - ETA: 42s - loss: 2.9830 - acc: 0.30 - ETA: 42s - loss: 2.9836 - acc: 0.30 - ETA: 42s - loss: 2.9878 - acc: 0.30 - ETA: 42s - loss: 2.9878 - acc: 0.30 - ETA: 42s - loss: 2.9897 - acc: 0.30 - ETA: 42s - loss: 2.9901 - acc: 0.30 - ETA: 41s - loss: 2.9830 - acc: 0.30 - ETA: 41s - loss: 2.9777 - acc: 0.30 - ETA: 41s - loss: 2.9784 - acc: 0.30 - ETA: 41s - loss: 2.9865 - acc: 0.30 - ETA: 41s - loss: 2.9849 - acc: 0.30 - ETA: 40s - loss: 2.9901 - acc: 0.30 - ETA: 40s - loss: 2.9971 - acc: 0.30 - ETA: 40s - loss: 2.9979 - acc: 0.30 - ETA: 40s - loss: 2.9944 - acc: 0.30 - ETA: 40s - loss: 2.9966 - acc: 0.30 - ETA: 40s - loss: 2.9934 - acc: 0.30 - ETA: 40s - loss: 2.9888 - acc: 0.30 - ETA: 39s - loss: 2.9975 - acc: 0.30 - ETA: 39s - loss: 2.9994 - acc: 0.30 - ETA: 39s - loss: 2.9987 - acc: 0.30 - ETA: 39s - loss: 3.0012 - acc: 0.30 - ETA: 39s - loss: 3.0020 - acc: 0.30 - ETA: 39s - loss: 3.0051 - acc: 0.30 - ETA: 38s - loss: 3.0098 - acc: 0.30 - ETA: 38s - loss: 3.0084 - acc: 0.30 - ETA: 38s - loss: 3.0064 - acc: 0.30 - ETA: 38s - loss: 3.0066 - acc: 0.29 - ETA: 38s - loss: 3.0036 - acc: 0.30 - ETA: 38s - loss: 3.0019 - acc: 0.30 - ETA: 37s - loss: 2.9952 - acc: 0.30 - ETA: 37s - loss: 2.9952 - acc: 0.30 - ETA: 37s - loss: 2.9919 - acc: 0.30 - ETA: 37s - loss: 2.9968 - acc: 0.30 - ETA: 37s - loss: 2.9999 - acc: 0.30 - ETA: 36s - loss: 3.0010 - acc: 0.30 - ETA: 36s - loss: 3.0007 - acc: 0.30 - ETA: 36s - loss: 2.9951 - acc: 0.30 - ETA: 36s - loss: 2.9948 - acc: 0.30 - ETA: 36s - loss: 2.9918 - acc: 0.30 - ETA: 36s - loss: 2.9887 - acc: 0.30 - ETA: 35s - loss: 2.9857 - acc: 0.30 - ETA: 35s - loss: 2.9869 - acc: 0.30 - ETA: 35s - loss: 2.9870 - acc: 0.30 - ETA: 35s - loss: 2.9907 - acc: 0.29 - ETA: 35s - loss: 2.9883 - acc: 0.29 - ETA: 35s - loss: 2.9817 - acc: 0.30 - ETA: 34s - loss: 2.9814 - acc: 0.30 - ETA: 34s - loss: 2.9782 - acc: 0.30 - ETA: 34s - loss: 2.9761 - acc: 0.30 - ETA: 34s - loss: 2.9757 - acc: 0.30 - ETA: 34s - loss: 2.9744 - acc: 0.30 - ETA: 33s - loss: 2.9706 - acc: 0.30 - ETA: 33s - loss: 2.9688 - acc: 0.30 - ETA: 33s - loss: 2.9668 - acc: 0.30 - ETA: 33s - loss: 2.9658 - acc: 0.30 - ETA: 33s - loss: 2.9655 - acc: 0.30 - ETA: 33s - loss: 2.9662 - acc: 0.30 - ETA: 32s - loss: 2.9632 - acc: 0.30 - ETA: 32s - loss: 2.9682 - acc: 0.30 - ETA: 32s - loss: 2.9704 - acc: 0.30 - ETA: 32s - loss: 2.9696 - acc: 0.30 - ETA: 32s - loss: 2.9662 - acc: 0.30 - ETA: 31s - loss: 2.9655 - acc: 0.30 - ETA: 31s - loss: 2.9642 - acc: 0.30 - ETA: 31s - loss: 2.9635 - acc: 0.30 - ETA: 31s - loss: 2.9645 - acc: 0.30 - ETA: 31s - loss: 2.9646 - acc: 0.30 - ETA: 31s - loss: 2.9622 - acc: 0.30 - ETA: 30s - loss: 2.9643 - acc: 0.30 - ETA: 30s - loss: 2.9655 - acc: 0.29 - ETA: 30s - loss: 2.9649 - acc: 0.29 - ETA: 30s - loss: 2.9640 - acc: 0.29 - ETA: 30s - loss: 2.9585 - acc: 0.30 - ETA: 30s - loss: 2.9563 - acc: 0.30 - ETA: 29s - loss: 2.9585 - acc: 0.30 - ETA: 29s - loss: 2.9552 - acc: 0.30 - ETA: 29s - loss: 2.9558 - acc: 0.30 - ETA: 29s - loss: 2.9508 - acc: 0.30 - ETA: 29s - loss: 2.9495 - acc: 0.30 - ETA: 29s - loss: 2.9491 - acc: 0.30 - ETA: 28s - loss: 2.9486 - acc: 0.30 - ETA: 28s - loss: 2.9488 - acc: 0.30 - ETA: 28s - loss: 2.9450 - acc: 0.30 - ETA: 28s - loss: 2.9452 - acc: 0.30 - ETA: 28s - loss: 2.9433 - acc: 0.30 - ETA: 27s - loss: 2.9412 - acc: 0.30 - ETA: 27s - loss: 2.9404 - acc: 0.30 - ETA: 27s - loss: 2.9417 - acc: 0.30 - ETA: 27s - loss: 2.9443 - acc: 0.30 - ETA: 27s - loss: 2.9451 - acc: 0.30 - ETA: 27s - loss: 2.9402 - acc: 0.30 - ETA: 26s - loss: 2.9376 - acc: 0.30 - ETA: 26s - loss: 2.9361 - acc: 0.30 - ETA: 26s - loss: 2.9317 - acc: 0.30 - ETA: 26s - loss: 2.9320 - acc: 0.30 - ETA: 26s - loss: 2.9334 - acc: 0.30 - ETA: 26s - loss: 2.9356 - acc: 0.30 - ETA: 25s - loss: 2.9383 - acc: 0.30 - ETA: 25s - loss: 2.9360 - acc: 0.30 - ETA: 25s - loss: 2.9390 - acc: 0.30 - ETA: 25s - loss: 2.9402 - acc: 0.30 - ETA: 25s - loss: 2.9398 - acc: 0.30 - ETA: 25s - loss: 2.9432 - acc: 0.30 - ETA: 24s - loss: 2.9412 - acc: 0.30 - ETA: 24s - loss: 2.9393 - acc: 0.30 - ETA: 24s - loss: 2.9402 - acc: 0.30 - ETA: 24s - loss: 2.9386 - acc: 0.30 - ETA: 24s - loss: 2.9398 - acc: 0.30 - ETA: 24s - loss: 2.9385 - acc: 0.30 - ETA: 23s - loss: 2.9371 - acc: 0.30 - ETA: 23s - loss: 2.9361 - acc: 0.30 - ETA: 23s - loss: 2.9368 - acc: 0.30 - ETA: 23s - loss: 2.9356 - acc: 0.30 - ETA: 23s - loss: 2.9355 - acc: 0.30 - ETA: 22s - loss: 2.9355 - acc: 0.30 - ETA: 22s - loss: 2.9386 - acc: 0.30 - ETA: 22s - loss: 2.9371 - acc: 0.30 - ETA: 22s - loss: 2.9367 - acc: 0.30 - ETA: 22s - loss: 2.9332 - acc: 0.30 - ETA: 22s - loss: 2.9338 - acc: 0.30 - ETA: 21s - loss: 2.9316 - acc: 0.30 - ETA: 21s - loss: 2.9330 - acc: 0.30 - ETA: 21s - loss: 2.9326 - acc: 0.30 - ETA: 21s - loss: 2.9323 - acc: 0.30 - ETA: 21s - loss: 2.9325 - acc: 0.30 - ETA: 21s - loss: 2.9313 - acc: 0.30 - ETA: 20s - loss: 2.9341 - acc: 0.30 - ETA: 20s - loss: 2.9349 - acc: 0.30 - ETA: 20s - loss: 2.9359 - acc: 0.30 - ETA: 20s - loss: 2.9347 - acc: 0.30586660/6680 [============================>.] - ETA: 20s - loss: 2.9327 - acc: 0.30 - ETA: 20s - loss: 2.9325 - acc: 0.30 - ETA: 19s - loss: 2.9328 - acc: 0.30 - ETA: 19s - loss: 2.9316 - acc: 0.30 - ETA: 19s - loss: 2.9315 - acc: 0.30 - ETA: 19s - loss: 2.9332 - acc: 0.30 - ETA: 19s - loss: 2.9339 - acc: 0.30 - ETA: 18s - loss: 2.9337 - acc: 0.30 - ETA: 18s - loss: 2.9346 - acc: 0.30 - ETA: 18s - loss: 2.9372 - acc: 0.30 - ETA: 18s - loss: 2.9363 - acc: 0.30 - ETA: 18s - loss: 2.9387 - acc: 0.30 - ETA: 18s - loss: 2.9396 - acc: 0.30 - ETA: 17s - loss: 2.9422 - acc: 0.30 - ETA: 17s - loss: 2.9414 - acc: 0.30 - ETA: 17s - loss: 2.9412 - acc: 0.30 - ETA: 17s - loss: 2.9408 - acc: 0.30 - ETA: 17s - loss: 2.9433 - acc: 0.30 - ETA: 17s - loss: 2.9428 - acc: 0.30 - ETA: 16s - loss: 2.9403 - acc: 0.30 - ETA: 16s - loss: 2.9397 - acc: 0.30 - ETA: 16s - loss: 2.9438 - acc: 0.30 - ETA: 16s - loss: 2.9443 - acc: 0.30 - ETA: 16s - loss: 2.9443 - acc: 0.30 - ETA: 16s - loss: 2.9420 - acc: 0.30 - ETA: 15s - loss: 2.9403 - acc: 0.30 - ETA: 15s - loss: 2.9392 - acc: 0.30 - ETA: 15s - loss: 2.9384 - acc: 0.30 - ETA: 15s - loss: 2.9386 - acc: 0.30 - ETA: 15s - loss: 2.9409 - acc: 0.30 - ETA: 15s - loss: 2.9414 - acc: 0.30 - ETA: 14s - loss: 2.9400 - acc: 0.30 - ETA: 14s - loss: 2.9385 - acc: 0.30 - ETA: 14s - loss: 2.9381 - acc: 0.30 - ETA: 14s - loss: 2.9374 - acc: 0.30 - ETA: 14s - loss: 2.9368 - acc: 0.30 - ETA: 14s - loss: 2.9398 - acc: 0.30 - ETA: 13s - loss: 2.9388 - acc: 0.30 - ETA: 13s - loss: 2.9410 - acc: 0.30 - ETA: 13s - loss: 2.9421 - acc: 0.30 - ETA: 13s - loss: 2.9438 - acc: 0.30 - ETA: 13s - loss: 2.9444 - acc: 0.30 - ETA: 12s - loss: 2.9460 - acc: 0.30 - ETA: 12s - loss: 2.9488 - acc: 0.30 - ETA: 12s - loss: 2.9466 - acc: 0.30 - ETA: 12s - loss: 2.9464 - acc: 0.30 - ETA: 12s - loss: 2.9468 - acc: 0.30 - ETA: 12s - loss: 2.9480 - acc: 0.30 - ETA: 11s - loss: 2.9475 - acc: 0.30 - ETA: 11s - loss: 2.9466 - acc: 0.30 - ETA: 11s - loss: 2.9474 - acc: 0.30 - ETA: 11s - loss: 2.9487 - acc: 0.30 - ETA: 11s - loss: 2.9508 - acc: 0.29 - ETA: 11s - loss: 2.9513 - acc: 0.29 - ETA: 10s - loss: 2.9528 - acc: 0.29 - ETA: 10s - loss: 2.9520 - acc: 0.29 - ETA: 10s - loss: 2.9517 - acc: 0.29 - ETA: 10s - loss: 2.9541 - acc: 0.29 - ETA: 10s - loss: 2.9529 - acc: 0.29 - ETA: 10s - loss: 2.9511 - acc: 0.29 - ETA: 9s - loss: 2.9500 - acc: 0.2995 - ETA: 9s - loss: 2.9522 - acc: 0.299 - ETA: 9s - loss: 2.9497 - acc: 0.299 - ETA: 9s - loss: 2.9469 - acc: 0.300 - ETA: 9s - loss: 2.9460 - acc: 0.300 - ETA: 9s - loss: 2.9478 - acc: 0.300 - ETA: 8s - loss: 2.9451 - acc: 0.300 - ETA: 8s - loss: 2.9453 - acc: 0.300 - ETA: 8s - loss: 2.9455 - acc: 0.301 - ETA: 8s - loss: 2.9456 - acc: 0.300 - ETA: 8s - loss: 2.9455 - acc: 0.300 - ETA: 8s - loss: 2.9480 - acc: 0.300 - ETA: 7s - loss: 2.9484 - acc: 0.299 - ETA: 7s - loss: 2.9488 - acc: 0.299 - ETA: 7s - loss: 2.9490 - acc: 0.299 - ETA: 7s - loss: 2.9508 - acc: 0.299 - ETA: 7s - loss: 2.9514 - acc: 0.299 - ETA: 6s - loss: 2.9528 - acc: 0.299 - ETA: 6s - loss: 2.9525 - acc: 0.299 - ETA: 6s - loss: 2.9541 - acc: 0.299 - ETA: 6s - loss: 2.9531 - acc: 0.299 - ETA: 6s - loss: 2.9509 - acc: 0.300 - ETA: 6s - loss: 2.9517 - acc: 0.300 - ETA: 5s - loss: 2.9538 - acc: 0.299 - ETA: 5s - loss: 2.9531 - acc: 0.299 - ETA: 5s - loss: 2.9540 - acc: 0.299 - ETA: 5s - loss: 2.9542 - acc: 0.299 - ETA: 5s - loss: 2.9534 - acc: 0.300 - ETA: 5s - loss: 2.9547 - acc: 0.299 - ETA: 4s - loss: 2.9525 - acc: 0.299 - ETA: 4s - loss: 2.9488 - acc: 0.300 - ETA: 4s - loss: 2.9502 - acc: 0.300 - ETA: 4s - loss: 2.9500 - acc: 0.300 - ETA: 4s - loss: 2.9530 - acc: 0.299 - ETA: 4s - loss: 2.9542 - acc: 0.299 - ETA: 3s - loss: 2.9521 - acc: 0.298 - ETA: 3s - loss: 2.9530 - acc: 0.298 - ETA: 3s - loss: 2.9513 - acc: 0.299 - ETA: 3s - loss: 2.9534 - acc: 0.298 - ETA: 3s - loss: 2.9526 - acc: 0.298 - ETA: 3s - loss: 2.9518 - acc: 0.298 - ETA: 2s - loss: 2.9505 - acc: 0.298 - ETA: 2s - loss: 2.9524 - acc: 0.298 - ETA: 2s - loss: 2.9515 - acc: 0.298 - ETA: 2s - loss: 2.9513 - acc: 0.298 - ETA: 2s - loss: 2.9531 - acc: 0.297 - ETA: 2s - loss: 2.9542 - acc: 0.297 - ETA: 1s - loss: 2.9537 - acc: 0.297 - ETA: 1s - loss: 2.9528 - acc: 0.298 - ETA: 1s - loss: 2.9559 - acc: 0.297 - ETA: 1s - loss: 2.9565 - acc: 0.297 - ETA: 1s - loss: 2.9576 - acc: 0.296 - ETA: 1s - loss: 2.9580 - acc: 0.297 - ETA: 0s - loss: 2.9582 - acc: 0.297 - ETA: 0s - loss: 2.9577 - acc: 0.297 - ETA: 0s - loss: 2.9569 - acc: 0.298 - ETA: 0s - loss: 2.9577 - acc: 0.298 - ETA: 0s - loss: 2.9572 - acc: 0.2982Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 59s 9ms/step - loss: 2.9572 - acc: 0.2978 - val_loss: 4.2757 - val_acc: 0.0874
    Epoch 6/10
    4300/6680 [==================>...........] - ETA: 55s - loss: 1.8287 - acc: 0.50 - ETA: 56s - loss: 2.0182 - acc: 0.42 - ETA: 56s - loss: 2.0831 - acc: 0.41 - ETA: 55s - loss: 2.2158 - acc: 0.38 - ETA: 55s - loss: 2.2218 - acc: 0.38 - ETA: 55s - loss: 2.2242 - acc: 0.38 - ETA: 55s - loss: 2.2014 - acc: 0.39 - ETA: 55s - loss: 2.2061 - acc: 0.40 - ETA: 55s - loss: 2.2195 - acc: 0.40 - ETA: 55s - loss: 2.1907 - acc: 0.41 - ETA: 55s - loss: 2.1877 - acc: 0.41 - ETA: 54s - loss: 2.2164 - acc: 0.41 - ETA: 54s - loss: 2.2407 - acc: 0.41 - ETA: 55s - loss: 2.2614 - acc: 0.42 - ETA: 54s - loss: 2.2443 - acc: 0.42 - ETA: 54s - loss: 2.2214 - acc: 0.42 - ETA: 54s - loss: 2.2220 - acc: 0.42 - ETA: 54s - loss: 2.2183 - acc: 0.42 - ETA: 54s - loss: 2.2593 - acc: 0.41 - ETA: 54s - loss: 2.2651 - acc: 0.41 - ETA: 54s - loss: 2.2475 - acc: 0.42 - ETA: 54s - loss: 2.2349 - acc: 0.42 - ETA: 53s - loss: 2.2309 - acc: 0.43 - ETA: 53s - loss: 2.2505 - acc: 0.43 - ETA: 53s - loss: 2.2584 - acc: 0.43 - ETA: 53s - loss: 2.2805 - acc: 0.43 - ETA: 53s - loss: 2.2836 - acc: 0.43 - ETA: 53s - loss: 2.2720 - acc: 0.43 - ETA: 53s - loss: 2.2952 - acc: 0.43 - ETA: 52s - loss: 2.2895 - acc: 0.43 - ETA: 52s - loss: 2.2916 - acc: 0.43 - ETA: 52s - loss: 2.2691 - acc: 0.43 - ETA: 52s - loss: 2.2588 - acc: 0.43 - ETA: 52s - loss: 2.2574 - acc: 0.44 - ETA: 51s - loss: 2.2540 - acc: 0.44 - ETA: 51s - loss: 2.2305 - acc: 0.45 - ETA: 51s - loss: 2.2102 - acc: 0.45 - ETA: 51s - loss: 2.2149 - acc: 0.45 - ETA: 51s - loss: 2.2164 - acc: 0.45 - ETA: 50s - loss: 2.2321 - acc: 0.45 - ETA: 50s - loss: 2.2322 - acc: 0.45 - ETA: 50s - loss: 2.2250 - acc: 0.45 - ETA: 50s - loss: 2.2234 - acc: 0.45 - ETA: 49s - loss: 2.2249 - acc: 0.44 - ETA: 49s - loss: 2.2239 - acc: 0.45 - ETA: 49s - loss: 2.2089 - acc: 0.45 - ETA: 49s - loss: 2.2096 - acc: 0.45 - ETA: 49s - loss: 2.1962 - acc: 0.45 - ETA: 49s - loss: 2.1978 - acc: 0.45 - ETA: 48s - loss: 2.1946 - acc: 0.46 - ETA: 48s - loss: 2.2134 - acc: 0.45 - ETA: 48s - loss: 2.2118 - acc: 0.45 - ETA: 48s - loss: 2.2164 - acc: 0.45 - ETA: 48s - loss: 2.2153 - acc: 0.45 - ETA: 47s - loss: 2.2126 - acc: 0.45 - ETA: 47s - loss: 2.2128 - acc: 0.45 - ETA: 47s - loss: 2.2122 - acc: 0.45 - ETA: 47s - loss: 2.2038 - acc: 0.45 - ETA: 47s - loss: 2.2160 - acc: 0.45 - ETA: 47s - loss: 2.2164 - acc: 0.45 - ETA: 46s - loss: 2.2158 - acc: 0.45 - ETA: 46s - loss: 2.2114 - acc: 0.45 - ETA: 46s - loss: 2.2054 - acc: 0.45 - ETA: 46s - loss: 2.2066 - acc: 0.45 - ETA: 46s - loss: 2.2083 - acc: 0.45 - ETA: 45s - loss: 2.2072 - acc: 0.45 - ETA: 45s - loss: 2.2085 - acc: 0.45 - ETA: 45s - loss: 2.2018 - acc: 0.45 - ETA: 45s - loss: 2.1960 - acc: 0.45 - ETA: 45s - loss: 2.1861 - acc: 0.46 - ETA: 45s - loss: 2.1926 - acc: 0.45 - ETA: 44s - loss: 2.1827 - acc: 0.46 - ETA: 44s - loss: 2.1865 - acc: 0.46 - ETA: 44s - loss: 2.1794 - acc: 0.46 - ETA: 44s - loss: 2.1822 - acc: 0.46 - ETA: 44s - loss: 2.1890 - acc: 0.46 - ETA: 43s - loss: 2.1921 - acc: 0.46 - ETA: 43s - loss: 2.1967 - acc: 0.46 - ETA: 43s - loss: 2.1925 - acc: 0.46 - ETA: 43s - loss: 2.1901 - acc: 0.46 - ETA: 43s - loss: 2.1890 - acc: 0.45 - ETA: 43s - loss: 2.1934 - acc: 0.45 - ETA: 42s - loss: 2.1895 - acc: 0.45 - ETA: 42s - loss: 2.1906 - acc: 0.45 - ETA: 42s - loss: 2.1867 - acc: 0.45 - ETA: 42s - loss: 2.1804 - acc: 0.45 - ETA: 42s - loss: 2.1777 - acc: 0.45 - ETA: 41s - loss: 2.1893 - acc: 0.45 - ETA: 41s - loss: 2.1893 - acc: 0.45 - ETA: 41s - loss: 2.1867 - acc: 0.45 - ETA: 41s - loss: 2.1890 - acc: 0.45 - ETA: 41s - loss: 2.1954 - acc: 0.45 - ETA: 41s - loss: 2.1925 - acc: 0.45 - ETA: 40s - loss: 2.1964 - acc: 0.45 - ETA: 40s - loss: 2.2004 - acc: 0.45 - ETA: 40s - loss: 2.2047 - acc: 0.45 - ETA: 40s - loss: 2.2057 - acc: 0.45 - ETA: 40s - loss: 2.2064 - acc: 0.45 - ETA: 40s - loss: 2.2121 - acc: 0.44 - ETA: 39s - loss: 2.2138 - acc: 0.44 - ETA: 39s - loss: 2.2141 - acc: 0.45 - ETA: 39s - loss: 2.2076 - acc: 0.45 - ETA: 39s - loss: 2.2039 - acc: 0.45 - ETA: 39s - loss: 2.1974 - acc: 0.45 - ETA: 39s - loss: 2.1907 - acc: 0.45 - ETA: 38s - loss: 2.1980 - acc: 0.45 - ETA: 38s - loss: 2.1942 - acc: 0.45 - ETA: 38s - loss: 2.1986 - acc: 0.45 - ETA: 38s - loss: 2.2017 - acc: 0.45 - ETA: 38s - loss: 2.2084 - acc: 0.45 - ETA: 37s - loss: 2.2054 - acc: 0.45 - ETA: 37s - loss: 2.2050 - acc: 0.45 - ETA: 37s - loss: 2.2037 - acc: 0.45 - ETA: 37s - loss: 2.2041 - acc: 0.45 - ETA: 37s - loss: 2.2111 - acc: 0.45 - ETA: 37s - loss: 2.2076 - acc: 0.45 - ETA: 36s - loss: 2.2005 - acc: 0.45 - ETA: 36s - loss: 2.1949 - acc: 0.45 - ETA: 36s - loss: 2.1987 - acc: 0.45 - ETA: 36s - loss: 2.1918 - acc: 0.45 - ETA: 36s - loss: 2.1958 - acc: 0.45 - ETA: 36s - loss: 2.1949 - acc: 0.45 - ETA: 35s - loss: 2.1942 - acc: 0.45 - ETA: 35s - loss: 2.1982 - acc: 0.45 - ETA: 35s - loss: 2.2069 - acc: 0.45 - ETA: 35s - loss: 2.2025 - acc: 0.45 - ETA: 35s - loss: 2.2101 - acc: 0.45 - ETA: 35s - loss: 2.2093 - acc: 0.45 - ETA: 34s - loss: 2.2087 - acc: 0.45 - ETA: 34s - loss: 2.2108 - acc: 0.45 - ETA: 34s - loss: 2.2110 - acc: 0.45 - ETA: 34s - loss: 2.2088 - acc: 0.45 - ETA: 34s - loss: 2.2059 - acc: 0.45 - ETA: 34s - loss: 2.1990 - acc: 0.45 - ETA: 33s - loss: 2.1988 - acc: 0.45 - ETA: 33s - loss: 2.2000 - acc: 0.45 - ETA: 33s - loss: 2.1977 - acc: 0.45 - ETA: 33s - loss: 2.1985 - acc: 0.45 - ETA: 33s - loss: 2.2056 - acc: 0.45 - ETA: 32s - loss: 2.2141 - acc: 0.45 - ETA: 32s - loss: 2.2172 - acc: 0.45 - ETA: 32s - loss: 2.2175 - acc: 0.45 - ETA: 32s - loss: 2.2133 - acc: 0.45 - ETA: 32s - loss: 2.2111 - acc: 0.45 - ETA: 32s - loss: 2.2094 - acc: 0.45 - ETA: 31s - loss: 2.2094 - acc: 0.45 - ETA: 31s - loss: 2.2095 - acc: 0.45 - ETA: 31s - loss: 2.2110 - acc: 0.45 - ETA: 31s - loss: 2.2187 - acc: 0.45 - ETA: 31s - loss: 2.2190 - acc: 0.45 - ETA: 31s - loss: 2.2186 - acc: 0.45 - ETA: 30s - loss: 2.2162 - acc: 0.45 - ETA: 30s - loss: 2.2217 - acc: 0.45 - ETA: 30s - loss: 2.2183 - acc: 0.45 - ETA: 30s - loss: 2.2150 - acc: 0.45 - ETA: 30s - loss: 2.2145 - acc: 0.45 - ETA: 30s - loss: 2.2143 - acc: 0.45 - ETA: 29s - loss: 2.2149 - acc: 0.45 - ETA: 29s - loss: 2.2120 - acc: 0.45 - ETA: 29s - loss: 2.2162 - acc: 0.45 - ETA: 29s - loss: 2.2134 - acc: 0.45 - ETA: 29s - loss: 2.2132 - acc: 0.45 - ETA: 29s - loss: 2.2127 - acc: 0.45 - ETA: 29s - loss: 2.2198 - acc: 0.45 - ETA: 28s - loss: 2.2210 - acc: 0.45 - ETA: 28s - loss: 2.2228 - acc: 0.45 - ETA: 28s - loss: 2.2192 - acc: 0.45 - ETA: 28s - loss: 2.2183 - acc: 0.45 - ETA: 28s - loss: 2.2181 - acc: 0.45 - ETA: 28s - loss: 2.2213 - acc: 0.45 - ETA: 27s - loss: 2.2214 - acc: 0.45 - ETA: 27s - loss: 2.2245 - acc: 0.45 - ETA: 27s - loss: 2.2260 - acc: 0.45 - ETA: 27s - loss: 2.2235 - acc: 0.45 - ETA: 27s - loss: 2.2245 - acc: 0.45 - ETA: 27s - loss: 2.2231 - acc: 0.45 - ETA: 26s - loss: 2.2225 - acc: 0.45 - ETA: 26s - loss: 2.2269 - acc: 0.44 - ETA: 26s - loss: 2.2246 - acc: 0.44 - ETA: 26s - loss: 2.2224 - acc: 0.45 - ETA: 26s - loss: 2.2238 - acc: 0.44 - ETA: 25s - loss: 2.2271 - acc: 0.44 - ETA: 25s - loss: 2.2274 - acc: 0.44 - ETA: 25s - loss: 2.2268 - acc: 0.44 - ETA: 25s - loss: 2.2252 - acc: 0.44 - ETA: 25s - loss: 2.2250 - acc: 0.44 - ETA: 25s - loss: 2.2272 - acc: 0.44 - ETA: 24s - loss: 2.2278 - acc: 0.44 - ETA: 24s - loss: 2.2286 - acc: 0.44 - ETA: 24s - loss: 2.2265 - acc: 0.44 - ETA: 24s - loss: 2.2269 - acc: 0.44 - ETA: 24s - loss: 2.2282 - acc: 0.44 - ETA: 24s - loss: 2.2278 - acc: 0.44 - ETA: 23s - loss: 2.2313 - acc: 0.44 - ETA: 23s - loss: 2.2281 - acc: 0.44 - ETA: 23s - loss: 2.2279 - acc: 0.44 - ETA: 23s - loss: 2.2288 - acc: 0.44 - ETA: 23s - loss: 2.2314 - acc: 0.44 - ETA: 23s - loss: 2.2359 - acc: 0.44 - ETA: 22s - loss: 2.2394 - acc: 0.44 - ETA: 22s - loss: 2.2372 - acc: 0.44 - ETA: 22s - loss: 2.2365 - acc: 0.44 - ETA: 22s - loss: 2.2357 - acc: 0.44 - ETA: 22s - loss: 2.2371 - acc: 0.44 - ETA: 22s - loss: 2.2388 - acc: 0.44 - ETA: 21s - loss: 2.2353 - acc: 0.44 - ETA: 21s - loss: 2.2347 - acc: 0.44 - ETA: 21s - loss: 2.2377 - acc: 0.44 - ETA: 21s - loss: 2.2395 - acc: 0.44 - ETA: 21s - loss: 2.2393 - acc: 0.44 - ETA: 21s - loss: 2.2415 - acc: 0.44 - ETA: 20s - loss: 2.2374 - acc: 0.44 - ETA: 20s - loss: 2.2378 - acc: 0.44 - ETA: 20s - loss: 2.2383 - acc: 0.44 - ETA: 20s - loss: 2.2387 - acc: 0.44336660/6680 [============================>.] - ETA: 20s - loss: 2.2384 - acc: 0.44 - ETA: 20s - loss: 2.2380 - acc: 0.44 - ETA: 19s - loss: 2.2381 - acc: 0.44 - ETA: 19s - loss: 2.2363 - acc: 0.44 - ETA: 19s - loss: 2.2364 - acc: 0.44 - ETA: 19s - loss: 2.2389 - acc: 0.44 - ETA: 19s - loss: 2.2374 - acc: 0.44 - ETA: 19s - loss: 2.2354 - acc: 0.44 - ETA: 18s - loss: 2.2387 - acc: 0.44 - ETA: 18s - loss: 2.2387 - acc: 0.44 - ETA: 18s - loss: 2.2396 - acc: 0.44 - ETA: 18s - loss: 2.2401 - acc: 0.44 - ETA: 18s - loss: 2.2421 - acc: 0.44 - ETA: 17s - loss: 2.2454 - acc: 0.44 - ETA: 17s - loss: 2.2474 - acc: 0.44 - ETA: 17s - loss: 2.2502 - acc: 0.44 - ETA: 17s - loss: 2.2507 - acc: 0.44 - ETA: 17s - loss: 2.2489 - acc: 0.44 - ETA: 17s - loss: 2.2517 - acc: 0.44 - ETA: 16s - loss: 2.2516 - acc: 0.44 - ETA: 16s - loss: 2.2516 - acc: 0.44 - ETA: 16s - loss: 2.2486 - acc: 0.44 - ETA: 16s - loss: 2.2492 - acc: 0.44 - ETA: 16s - loss: 2.2487 - acc: 0.44 - ETA: 16s - loss: 2.2487 - acc: 0.44 - ETA: 15s - loss: 2.2489 - acc: 0.44 - ETA: 15s - loss: 2.2462 - acc: 0.44 - ETA: 15s - loss: 2.2476 - acc: 0.44 - ETA: 15s - loss: 2.2449 - acc: 0.44 - ETA: 15s - loss: 2.2462 - acc: 0.44 - ETA: 15s - loss: 2.2471 - acc: 0.44 - ETA: 14s - loss: 2.2472 - acc: 0.44 - ETA: 14s - loss: 2.2455 - acc: 0.44 - ETA: 14s - loss: 2.2463 - acc: 0.44 - ETA: 14s - loss: 2.2452 - acc: 0.44 - ETA: 14s - loss: 2.2459 - acc: 0.44 - ETA: 14s - loss: 2.2445 - acc: 0.44 - ETA: 13s - loss: 2.2447 - acc: 0.44 - ETA: 13s - loss: 2.2460 - acc: 0.44 - ETA: 13s - loss: 2.2476 - acc: 0.44 - ETA: 13s - loss: 2.2481 - acc: 0.44 - ETA: 13s - loss: 2.2501 - acc: 0.44 - ETA: 13s - loss: 2.2514 - acc: 0.44 - ETA: 12s - loss: 2.2535 - acc: 0.44 - ETA: 12s - loss: 2.2518 - acc: 0.44 - ETA: 12s - loss: 2.2528 - acc: 0.44 - ETA: 12s - loss: 2.2544 - acc: 0.44 - ETA: 12s - loss: 2.2571 - acc: 0.44 - ETA: 12s - loss: 2.2593 - acc: 0.44 - ETA: 11s - loss: 2.2586 - acc: 0.44 - ETA: 11s - loss: 2.2584 - acc: 0.44 - ETA: 11s - loss: 2.2623 - acc: 0.44 - ETA: 11s - loss: 2.2647 - acc: 0.43 - ETA: 11s - loss: 2.2672 - acc: 0.43 - ETA: 10s - loss: 2.2689 - acc: 0.43 - ETA: 10s - loss: 2.2672 - acc: 0.43 - ETA: 10s - loss: 2.2662 - acc: 0.43 - ETA: 10s - loss: 2.2644 - acc: 0.43 - ETA: 10s - loss: 2.2674 - acc: 0.43 - ETA: 10s - loss: 2.2655 - acc: 0.43 - ETA: 9s - loss: 2.2644 - acc: 0.4395 - ETA: 9s - loss: 2.2643 - acc: 0.439 - ETA: 9s - loss: 2.2622 - acc: 0.439 - ETA: 9s - loss: 2.2610 - acc: 0.439 - ETA: 9s - loss: 2.2614 - acc: 0.438 - ETA: 9s - loss: 2.2604 - acc: 0.439 - ETA: 8s - loss: 2.2626 - acc: 0.439 - ETA: 8s - loss: 2.2631 - acc: 0.439 - ETA: 8s - loss: 2.2631 - acc: 0.438 - ETA: 8s - loss: 2.2634 - acc: 0.438 - ETA: 8s - loss: 2.2638 - acc: 0.438 - ETA: 8s - loss: 2.2669 - acc: 0.438 - ETA: 7s - loss: 2.2697 - acc: 0.437 - ETA: 7s - loss: 2.2709 - acc: 0.437 - ETA: 7s - loss: 2.2726 - acc: 0.437 - ETA: 7s - loss: 2.2742 - acc: 0.436 - ETA: 7s - loss: 2.2741 - acc: 0.436 - ETA: 7s - loss: 2.2746 - acc: 0.435 - ETA: 6s - loss: 2.2730 - acc: 0.436 - ETA: 6s - loss: 2.2729 - acc: 0.436 - ETA: 6s - loss: 2.2732 - acc: 0.436 - ETA: 6s - loss: 2.2753 - acc: 0.436 - ETA: 6s - loss: 2.2736 - acc: 0.436 - ETA: 5s - loss: 2.2743 - acc: 0.436 - ETA: 5s - loss: 2.2727 - acc: 0.436 - ETA: 5s - loss: 2.2738 - acc: 0.436 - ETA: 5s - loss: 2.2735 - acc: 0.436 - ETA: 5s - loss: 2.2780 - acc: 0.435 - ETA: 5s - loss: 2.2798 - acc: 0.435 - ETA: 4s - loss: 2.2790 - acc: 0.435 - ETA: 4s - loss: 2.2805 - acc: 0.435 - ETA: 4s - loss: 2.2781 - acc: 0.435 - ETA: 4s - loss: 2.2789 - acc: 0.435 - ETA: 4s - loss: 2.2789 - acc: 0.435 - ETA: 4s - loss: 2.2789 - acc: 0.434 - ETA: 3s - loss: 2.2781 - acc: 0.434 - ETA: 3s - loss: 2.2791 - acc: 0.434 - ETA: 3s - loss: 2.2781 - acc: 0.434 - ETA: 3s - loss: 2.2766 - acc: 0.435 - ETA: 3s - loss: 2.2782 - acc: 0.434 - ETA: 3s - loss: 2.2761 - acc: 0.434 - ETA: 2s - loss: 2.2757 - acc: 0.435 - ETA: 2s - loss: 2.2748 - acc: 0.435 - ETA: 2s - loss: 2.2747 - acc: 0.435 - ETA: 2s - loss: 2.2745 - acc: 0.435 - ETA: 2s - loss: 2.2758 - acc: 0.435 - ETA: 2s - loss: 2.2767 - acc: 0.435 - ETA: 1s - loss: 2.2781 - acc: 0.435 - ETA: 1s - loss: 2.2769 - acc: 0.435 - ETA: 1s - loss: 2.2758 - acc: 0.435 - ETA: 1s - loss: 2.2775 - acc: 0.435 - ETA: 1s - loss: 2.2777 - acc: 0.435 - ETA: 1s - loss: 2.2754 - acc: 0.435 - ETA: 0s - loss: 2.2745 - acc: 0.436 - ETA: 0s - loss: 2.2740 - acc: 0.436 - ETA: 0s - loss: 2.2747 - acc: 0.436 - ETA: 0s - loss: 2.2743 - acc: 0.435 - ETA: 0s - loss: 2.2735 - acc: 0.4359Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 59s 9ms/step - loss: 2.2733 - acc: 0.4359 - val_loss: 4.7518 - val_acc: 0.0982
    Epoch 7/10
    4300/6680 [==================>...........] - ETA: 58s - loss: 1.1683 - acc: 0.55 - ETA: 56s - loss: 1.3811 - acc: 0.62 - ETA: 56s - loss: 1.5662 - acc: 0.58 - ETA: 56s - loss: 1.7192 - acc: 0.56 - ETA: 55s - loss: 1.5193 - acc: 0.61 - ETA: 55s - loss: 1.5079 - acc: 0.63 - ETA: 55s - loss: 1.4217 - acc: 0.65 - ETA: 54s - loss: 1.4392 - acc: 0.63 - ETA: 54s - loss: 1.4792 - acc: 0.62 - ETA: 54s - loss: 1.5653 - acc: 0.59 - ETA: 54s - loss: 1.5554 - acc: 0.60 - ETA: 54s - loss: 1.5729 - acc: 0.60 - ETA: 53s - loss: 1.5696 - acc: 0.60 - ETA: 53s - loss: 1.5403 - acc: 0.60 - ETA: 53s - loss: 1.5514 - acc: 0.59 - ETA: 53s - loss: 1.5464 - acc: 0.60 - ETA: 53s - loss: 1.5657 - acc: 0.59 - ETA: 52s - loss: 1.5669 - acc: 0.59 - ETA: 52s - loss: 1.5691 - acc: 0.59 - ETA: 52s - loss: 1.5512 - acc: 0.60 - ETA: 52s - loss: 1.5568 - acc: 0.60 - ETA: 52s - loss: 1.5571 - acc: 0.60 - ETA: 52s - loss: 1.5754 - acc: 0.59 - ETA: 51s - loss: 1.5700 - acc: 0.59 - ETA: 51s - loss: 1.5582 - acc: 0.60 - ETA: 51s - loss: 1.5442 - acc: 0.60 - ETA: 51s - loss: 1.5555 - acc: 0.60 - ETA: 51s - loss: 1.5663 - acc: 0.60 - ETA: 51s - loss: 1.5675 - acc: 0.60 - ETA: 51s - loss: 1.5699 - acc: 0.60 - ETA: 50s - loss: 1.5596 - acc: 0.60 - ETA: 50s - loss: 1.5633 - acc: 0.60 - ETA: 50s - loss: 1.5661 - acc: 0.60 - ETA: 50s - loss: 1.5725 - acc: 0.60 - ETA: 50s - loss: 1.5530 - acc: 0.61 - ETA: 50s - loss: 1.5513 - acc: 0.60 - ETA: 50s - loss: 1.5378 - acc: 0.60 - ETA: 50s - loss: 1.5389 - acc: 0.61 - ETA: 49s - loss: 1.5323 - acc: 0.61 - ETA: 49s - loss: 1.5349 - acc: 0.60 - ETA: 49s - loss: 1.5406 - acc: 0.60 - ETA: 49s - loss: 1.5495 - acc: 0.60 - ETA: 49s - loss: 1.5469 - acc: 0.60 - ETA: 48s - loss: 1.5432 - acc: 0.60 - ETA: 48s - loss: 1.5434 - acc: 0.60 - ETA: 48s - loss: 1.5292 - acc: 0.61 - ETA: 48s - loss: 1.5257 - acc: 0.61 - ETA: 48s - loss: 1.5174 - acc: 0.61 - ETA: 48s - loss: 1.5204 - acc: 0.61 - ETA: 47s - loss: 1.5126 - acc: 0.62 - ETA: 47s - loss: 1.5239 - acc: 0.61 - ETA: 47s - loss: 1.5254 - acc: 0.61 - ETA: 47s - loss: 1.5223 - acc: 0.61 - ETA: 47s - loss: 1.5146 - acc: 0.61 - ETA: 46s - loss: 1.5169 - acc: 0.61 - ETA: 46s - loss: 1.5226 - acc: 0.61 - ETA: 46s - loss: 1.5197 - acc: 0.61 - ETA: 46s - loss: 1.5266 - acc: 0.61 - ETA: 46s - loss: 1.5348 - acc: 0.61 - ETA: 46s - loss: 1.5400 - acc: 0.60 - ETA: 45s - loss: 1.5425 - acc: 0.60 - ETA: 45s - loss: 1.5360 - acc: 0.61 - ETA: 45s - loss: 1.5362 - acc: 0.61 - ETA: 45s - loss: 1.5411 - acc: 0.61 - ETA: 45s - loss: 1.5389 - acc: 0.61 - ETA: 45s - loss: 1.5495 - acc: 0.60 - ETA: 45s - loss: 1.5549 - acc: 0.60 - ETA: 44s - loss: 1.5525 - acc: 0.60 - ETA: 44s - loss: 1.5593 - acc: 0.60 - ETA: 44s - loss: 1.5593 - acc: 0.60 - ETA: 44s - loss: 1.5682 - acc: 0.60 - ETA: 44s - loss: 1.5781 - acc: 0.60 - ETA: 44s - loss: 1.5729 - acc: 0.60 - ETA: 43s - loss: 1.5749 - acc: 0.59 - ETA: 43s - loss: 1.5859 - acc: 0.59 - ETA: 43s - loss: 1.5904 - acc: 0.59 - ETA: 43s - loss: 1.6001 - acc: 0.59 - ETA: 43s - loss: 1.5973 - acc: 0.59 - ETA: 43s - loss: 1.5943 - acc: 0.59 - ETA: 42s - loss: 1.5985 - acc: 0.59 - ETA: 42s - loss: 1.6019 - acc: 0.59 - ETA: 42s - loss: 1.6040 - acc: 0.58 - ETA: 42s - loss: 1.5992 - acc: 0.59 - ETA: 42s - loss: 1.5982 - acc: 0.59 - ETA: 42s - loss: 1.5912 - acc: 0.59 - ETA: 41s - loss: 1.5922 - acc: 0.59 - ETA: 41s - loss: 1.5903 - acc: 0.59 - ETA: 41s - loss: 1.5939 - acc: 0.59 - ETA: 41s - loss: 1.5938 - acc: 0.59 - ETA: 41s - loss: 1.5988 - acc: 0.59 - ETA: 41s - loss: 1.5977 - acc: 0.59 - ETA: 40s - loss: 1.5987 - acc: 0.59 - ETA: 40s - loss: 1.6005 - acc: 0.59 - ETA: 40s - loss: 1.5960 - acc: 0.59 - ETA: 40s - loss: 1.5861 - acc: 0.59 - ETA: 40s - loss: 1.5908 - acc: 0.59 - ETA: 40s - loss: 1.5887 - acc: 0.59 - ETA: 39s - loss: 1.5928 - acc: 0.59 - ETA: 39s - loss: 1.5923 - acc: 0.59 - ETA: 39s - loss: 1.5914 - acc: 0.59 - ETA: 39s - loss: 1.5903 - acc: 0.59 - ETA: 39s - loss: 1.5904 - acc: 0.59 - ETA: 39s - loss: 1.5941 - acc: 0.59 - ETA: 38s - loss: 1.5952 - acc: 0.59 - ETA: 38s - loss: 1.5964 - acc: 0.59 - ETA: 38s - loss: 1.5984 - acc: 0.59 - ETA: 38s - loss: 1.5959 - acc: 0.59 - ETA: 38s - loss: 1.5945 - acc: 0.59 - ETA: 38s - loss: 1.5933 - acc: 0.59 - ETA: 37s - loss: 1.5960 - acc: 0.59 - ETA: 37s - loss: 1.5934 - acc: 0.59 - ETA: 37s - loss: 1.5927 - acc: 0.59 - ETA: 37s - loss: 1.5924 - acc: 0.59 - ETA: 37s - loss: 1.5936 - acc: 0.59 - ETA: 37s - loss: 1.6005 - acc: 0.59 - ETA: 36s - loss: 1.5989 - acc: 0.59 - ETA: 36s - loss: 1.5934 - acc: 0.59 - ETA: 36s - loss: 1.5864 - acc: 0.59 - ETA: 36s - loss: 1.5890 - acc: 0.59 - ETA: 36s - loss: 1.5808 - acc: 0.60 - ETA: 36s - loss: 1.5839 - acc: 0.60 - ETA: 35s - loss: 1.5837 - acc: 0.60 - ETA: 35s - loss: 1.5853 - acc: 0.59 - ETA: 35s - loss: 1.5866 - acc: 0.59 - ETA: 35s - loss: 1.5918 - acc: 0.59 - ETA: 35s - loss: 1.5868 - acc: 0.59 - ETA: 35s - loss: 1.5878 - acc: 0.59 - ETA: 34s - loss: 1.5872 - acc: 0.59 - ETA: 34s - loss: 1.5851 - acc: 0.59 - ETA: 34s - loss: 1.5876 - acc: 0.59 - ETA: 34s - loss: 1.5927 - acc: 0.59 - ETA: 34s - loss: 1.5939 - acc: 0.59 - ETA: 34s - loss: 1.5970 - acc: 0.59 - ETA: 33s - loss: 1.5980 - acc: 0.59 - ETA: 33s - loss: 1.5961 - acc: 0.59 - ETA: 33s - loss: 1.5991 - acc: 0.59 - ETA: 33s - loss: 1.5979 - acc: 0.59 - ETA: 33s - loss: 1.5970 - acc: 0.59 - ETA: 32s - loss: 1.5992 - acc: 0.59 - ETA: 32s - loss: 1.5959 - acc: 0.59 - ETA: 32s - loss: 1.5984 - acc: 0.59 - ETA: 32s - loss: 1.6003 - acc: 0.59 - ETA: 32s - loss: 1.6018 - acc: 0.59 - ETA: 32s - loss: 1.6032 - acc: 0.59 - ETA: 31s - loss: 1.6025 - acc: 0.59 - ETA: 31s - loss: 1.6026 - acc: 0.59 - ETA: 31s - loss: 1.6027 - acc: 0.59 - ETA: 31s - loss: 1.6017 - acc: 0.59 - ETA: 31s - loss: 1.5998 - acc: 0.59 - ETA: 31s - loss: 1.6007 - acc: 0.59 - ETA: 30s - loss: 1.6006 - acc: 0.59 - ETA: 30s - loss: 1.5993 - acc: 0.59 - ETA: 30s - loss: 1.6034 - acc: 0.59 - ETA: 30s - loss: 1.6060 - acc: 0.59 - ETA: 30s - loss: 1.6036 - acc: 0.59 - ETA: 30s - loss: 1.6026 - acc: 0.59 - ETA: 29s - loss: 1.6019 - acc: 0.59 - ETA: 29s - loss: 1.6013 - acc: 0.59 - ETA: 29s - loss: 1.6007 - acc: 0.59 - ETA: 29s - loss: 1.6023 - acc: 0.59 - ETA: 29s - loss: 1.6038 - acc: 0.59 - ETA: 29s - loss: 1.6081 - acc: 0.59 - ETA: 28s - loss: 1.6093 - acc: 0.59 - ETA: 28s - loss: 1.6074 - acc: 0.59 - ETA: 28s - loss: 1.6077 - acc: 0.59 - ETA: 28s - loss: 1.6066 - acc: 0.59 - ETA: 28s - loss: 1.6062 - acc: 0.59 - ETA: 28s - loss: 1.6075 - acc: 0.59 - ETA: 27s - loss: 1.6091 - acc: 0.59 - ETA: 27s - loss: 1.6119 - acc: 0.59 - ETA: 27s - loss: 1.6089 - acc: 0.59 - ETA: 27s - loss: 1.6098 - acc: 0.59 - ETA: 27s - loss: 1.6096 - acc: 0.59 - ETA: 27s - loss: 1.6096 - acc: 0.59 - ETA: 26s - loss: 1.6119 - acc: 0.59 - ETA: 26s - loss: 1.6119 - acc: 0.59 - ETA: 26s - loss: 1.6124 - acc: 0.59 - ETA: 26s - loss: 1.6159 - acc: 0.59 - ETA: 26s - loss: 1.6164 - acc: 0.59 - ETA: 26s - loss: 1.6146 - acc: 0.59 - ETA: 25s - loss: 1.6217 - acc: 0.58 - ETA: 25s - loss: 1.6194 - acc: 0.59 - ETA: 25s - loss: 1.6185 - acc: 0.59 - ETA: 25s - loss: 1.6205 - acc: 0.58 - ETA: 25s - loss: 1.6198 - acc: 0.58 - ETA: 25s - loss: 1.6210 - acc: 0.58 - ETA: 24s - loss: 1.6185 - acc: 0.59 - ETA: 24s - loss: 1.6177 - acc: 0.59 - ETA: 24s - loss: 1.6176 - acc: 0.59 - ETA: 24s - loss: 1.6193 - acc: 0.59 - ETA: 24s - loss: 1.6209 - acc: 0.58 - ETA: 24s - loss: 1.6195 - acc: 0.58 - ETA: 23s - loss: 1.6196 - acc: 0.58 - ETA: 23s - loss: 1.6227 - acc: 0.58 - ETA: 23s - loss: 1.6227 - acc: 0.58 - ETA: 23s - loss: 1.6259 - acc: 0.58 - ETA: 23s - loss: 1.6270 - acc: 0.58 - ETA: 23s - loss: 1.6289 - acc: 0.58 - ETA: 22s - loss: 1.6280 - acc: 0.58 - ETA: 22s - loss: 1.6279 - acc: 0.58 - ETA: 22s - loss: 1.6288 - acc: 0.58 - ETA: 22s - loss: 1.6295 - acc: 0.58 - ETA: 22s - loss: 1.6318 - acc: 0.58 - ETA: 22s - loss: 1.6313 - acc: 0.58 - ETA: 21s - loss: 1.6283 - acc: 0.58 - ETA: 21s - loss: 1.6317 - acc: 0.58 - ETA: 21s - loss: 1.6341 - acc: 0.58 - ETA: 21s - loss: 1.6353 - acc: 0.58 - ETA: 21s - loss: 1.6363 - acc: 0.58 - ETA: 21s - loss: 1.6350 - acc: 0.58 - ETA: 20s - loss: 1.6339 - acc: 0.58 - ETA: 20s - loss: 1.6337 - acc: 0.58 - ETA: 20s - loss: 1.6328 - acc: 0.58 - ETA: 20s - loss: 1.6351 - acc: 0.58 - ETA: 20s - loss: 1.6366 - acc: 0.58586660/6680 [============================>.] - ETA: 20s - loss: 1.6422 - acc: 0.58 - ETA: 19s - loss: 1.6426 - acc: 0.58 - ETA: 19s - loss: 1.6405 - acc: 0.58 - ETA: 19s - loss: 1.6455 - acc: 0.58 - ETA: 19s - loss: 1.6441 - acc: 0.58 - ETA: 19s - loss: 1.6428 - acc: 0.58 - ETA: 19s - loss: 1.6429 - acc: 0.58 - ETA: 18s - loss: 1.6407 - acc: 0.58 - ETA: 18s - loss: 1.6417 - acc: 0.58 - ETA: 18s - loss: 1.6451 - acc: 0.58 - ETA: 18s - loss: 1.6450 - acc: 0.58 - ETA: 18s - loss: 1.6422 - acc: 0.58 - ETA: 18s - loss: 1.6414 - acc: 0.58 - ETA: 17s - loss: 1.6416 - acc: 0.58 - ETA: 17s - loss: 1.6417 - acc: 0.58 - ETA: 17s - loss: 1.6409 - acc: 0.58 - ETA: 17s - loss: 1.6405 - acc: 0.58 - ETA: 17s - loss: 1.6391 - acc: 0.58 - ETA: 16s - loss: 1.6416 - acc: 0.58 - ETA: 16s - loss: 1.6427 - acc: 0.58 - ETA: 16s - loss: 1.6423 - acc: 0.58 - ETA: 16s - loss: 1.6441 - acc: 0.58 - ETA: 16s - loss: 1.6442 - acc: 0.58 - ETA: 16s - loss: 1.6454 - acc: 0.58 - ETA: 15s - loss: 1.6453 - acc: 0.58 - ETA: 15s - loss: 1.6442 - acc: 0.58 - ETA: 15s - loss: 1.6420 - acc: 0.58 - ETA: 15s - loss: 1.6426 - acc: 0.58 - ETA: 15s - loss: 1.6420 - acc: 0.58 - ETA: 15s - loss: 1.6440 - acc: 0.58 - ETA: 14s - loss: 1.6460 - acc: 0.58 - ETA: 14s - loss: 1.6477 - acc: 0.58 - ETA: 14s - loss: 1.6470 - acc: 0.58 - ETA: 14s - loss: 1.6475 - acc: 0.58 - ETA: 14s - loss: 1.6504 - acc: 0.58 - ETA: 14s - loss: 1.6509 - acc: 0.58 - ETA: 13s - loss: 1.6500 - acc: 0.58 - ETA: 13s - loss: 1.6509 - acc: 0.58 - ETA: 13s - loss: 1.6511 - acc: 0.58 - ETA: 13s - loss: 1.6515 - acc: 0.58 - ETA: 13s - loss: 1.6494 - acc: 0.58 - ETA: 13s - loss: 1.6483 - acc: 0.58 - ETA: 12s - loss: 1.6468 - acc: 0.58 - ETA: 12s - loss: 1.6471 - acc: 0.58 - ETA: 12s - loss: 1.6469 - acc: 0.58 - ETA: 12s - loss: 1.6458 - acc: 0.58 - ETA: 12s - loss: 1.6466 - acc: 0.58 - ETA: 12s - loss: 1.6467 - acc: 0.58 - ETA: 11s - loss: 1.6459 - acc: 0.58 - ETA: 11s - loss: 1.6456 - acc: 0.58 - ETA: 11s - loss: 1.6455 - acc: 0.58 - ETA: 11s - loss: 1.6436 - acc: 0.58 - ETA: 11s - loss: 1.6452 - acc: 0.58 - ETA: 11s - loss: 1.6464 - acc: 0.58 - ETA: 10s - loss: 1.6464 - acc: 0.58 - ETA: 10s - loss: 1.6465 - acc: 0.58 - ETA: 10s - loss: 1.6483 - acc: 0.58 - ETA: 10s - loss: 1.6512 - acc: 0.58 - ETA: 10s - loss: 1.6509 - acc: 0.58 - ETA: 10s - loss: 1.6509 - acc: 0.58 - ETA: 9s - loss: 1.6513 - acc: 0.5832 - ETA: 9s - loss: 1.6524 - acc: 0.582 - ETA: 9s - loss: 1.6539 - acc: 0.582 - ETA: 9s - loss: 1.6542 - acc: 0.583 - ETA: 9s - loss: 1.6542 - acc: 0.583 - ETA: 8s - loss: 1.6540 - acc: 0.583 - ETA: 8s - loss: 1.6523 - acc: 0.583 - ETA: 8s - loss: 1.6532 - acc: 0.583 - ETA: 8s - loss: 1.6542 - acc: 0.583 - ETA: 8s - loss: 1.6547 - acc: 0.583 - ETA: 8s - loss: 1.6560 - acc: 0.583 - ETA: 7s - loss: 1.6567 - acc: 0.582 - ETA: 7s - loss: 1.6558 - acc: 0.582 - ETA: 7s - loss: 1.6576 - acc: 0.582 - ETA: 7s - loss: 1.6602 - acc: 0.582 - ETA: 7s - loss: 1.6605 - acc: 0.582 - ETA: 7s - loss: 1.6598 - acc: 0.581 - ETA: 6s - loss: 1.6602 - acc: 0.581 - ETA: 6s - loss: 1.6609 - acc: 0.581 - ETA: 6s - loss: 1.6613 - acc: 0.580 - ETA: 6s - loss: 1.6629 - acc: 0.580 - ETA: 6s - loss: 1.6647 - acc: 0.580 - ETA: 6s - loss: 1.6647 - acc: 0.580 - ETA: 5s - loss: 1.6635 - acc: 0.580 - ETA: 5s - loss: 1.6638 - acc: 0.580 - ETA: 5s - loss: 1.6680 - acc: 0.579 - ETA: 5s - loss: 1.6673 - acc: 0.579 - ETA: 5s - loss: 1.6667 - acc: 0.579 - ETA: 5s - loss: 1.6669 - acc: 0.579 - ETA: 4s - loss: 1.6665 - acc: 0.579 - ETA: 4s - loss: 1.6661 - acc: 0.580 - ETA: 4s - loss: 1.6678 - acc: 0.580 - ETA: 4s - loss: 1.6662 - acc: 0.580 - ETA: 4s - loss: 1.6648 - acc: 0.580 - ETA: 4s - loss: 1.6650 - acc: 0.580 - ETA: 3s - loss: 1.6650 - acc: 0.580 - ETA: 3s - loss: 1.6659 - acc: 0.580 - ETA: 3s - loss: 1.6673 - acc: 0.580 - ETA: 3s - loss: 1.6677 - acc: 0.579 - ETA: 3s - loss: 1.6661 - acc: 0.580 - ETA: 3s - loss: 1.6670 - acc: 0.580 - ETA: 2s - loss: 1.6652 - acc: 0.580 - ETA: 2s - loss: 1.6670 - acc: 0.580 - ETA: 2s - loss: 1.6651 - acc: 0.580 - ETA: 2s - loss: 1.6646 - acc: 0.580 - ETA: 2s - loss: 1.6663 - acc: 0.580 - ETA: 2s - loss: 1.6681 - acc: 0.580 - ETA: 1s - loss: 1.6691 - acc: 0.580 - ETA: 1s - loss: 1.6723 - acc: 0.579 - ETA: 1s - loss: 1.6737 - acc: 0.579 - ETA: 1s - loss: 1.6759 - acc: 0.579 - ETA: 1s - loss: 1.6743 - acc: 0.579 - ETA: 1s - loss: 1.6729 - acc: 0.579 - ETA: 0s - loss: 1.6746 - acc: 0.579 - ETA: 0s - loss: 1.6735 - acc: 0.579 - ETA: 0s - loss: 1.6745 - acc: 0.579 - ETA: 0s - loss: 1.6747 - acc: 0.579 - ETA: 0s - loss: 1.6742 - acc: 0.5793Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 59s 9ms/step - loss: 1.6754 - acc: 0.5793 - val_loss: 4.7848 - val_acc: 0.0850
    Epoch 8/10
    4300/6680 [==================>...........] - ETA: 55s - loss: 1.1066 - acc: 0.70 - ETA: 56s - loss: 1.0419 - acc: 0.70 - ETA: 57s - loss: 1.0036 - acc: 0.71 - ETA: 58s - loss: 1.0978 - acc: 0.70 - ETA: 58s - loss: 1.0395 - acc: 0.72 - ETA: 57s - loss: 1.2191 - acc: 0.68 - ETA: 57s - loss: 1.2275 - acc: 0.68 - ETA: 57s - loss: 1.2103 - acc: 0.68 - ETA: 57s - loss: 1.2269 - acc: 0.66 - ETA: 56s - loss: 1.1877 - acc: 0.66 - ETA: 56s - loss: 1.1555 - acc: 0.67 - ETA: 56s - loss: 1.1317 - acc: 0.68 - ETA: 56s - loss: 1.1129 - acc: 0.69 - ETA: 56s - loss: 1.1155 - acc: 0.68 - ETA: 55s - loss: 1.0935 - acc: 0.69 - ETA: 55s - loss: 1.0797 - acc: 0.70 - ETA: 55s - loss: 1.0537 - acc: 0.70 - ETA: 55s - loss: 1.0455 - acc: 0.71 - ETA: 54s - loss: 1.0501 - acc: 0.70 - ETA: 54s - loss: 1.0401 - acc: 0.71 - ETA: 54s - loss: 1.0259 - acc: 0.71 - ETA: 54s - loss: 1.0248 - acc: 0.71 - ETA: 54s - loss: 1.0326 - acc: 0.71 - ETA: 54s - loss: 1.0288 - acc: 0.71 - ETA: 53s - loss: 1.0393 - acc: 0.71 - ETA: 53s - loss: 1.0555 - acc: 0.71 - ETA: 53s - loss: 1.0541 - acc: 0.71 - ETA: 53s - loss: 1.0583 - acc: 0.71 - ETA: 53s - loss: 1.0677 - acc: 0.70 - ETA: 52s - loss: 1.0852 - acc: 0.70 - ETA: 52s - loss: 1.1058 - acc: 0.70 - ETA: 52s - loss: 1.1005 - acc: 0.70 - ETA: 52s - loss: 1.0939 - acc: 0.70 - ETA: 51s - loss: 1.0902 - acc: 0.70 - ETA: 51s - loss: 1.0895 - acc: 0.70 - ETA: 51s - loss: 1.0856 - acc: 0.70 - ETA: 51s - loss: 1.0950 - acc: 0.70 - ETA: 51s - loss: 1.0990 - acc: 0.70 - ETA: 50s - loss: 1.0968 - acc: 0.70 - ETA: 50s - loss: 1.0904 - acc: 0.70 - ETA: 50s - loss: 1.1003 - acc: 0.70 - ETA: 50s - loss: 1.0965 - acc: 0.70 - ETA: 50s - loss: 1.0866 - acc: 0.70 - ETA: 50s - loss: 1.0734 - acc: 0.71 - ETA: 49s - loss: 1.0957 - acc: 0.71 - ETA: 49s - loss: 1.1039 - acc: 0.70 - ETA: 49s - loss: 1.1056 - acc: 0.70 - ETA: 49s - loss: 1.1120 - acc: 0.70 - ETA: 49s - loss: 1.1110 - acc: 0.70 - ETA: 49s - loss: 1.1126 - acc: 0.70 - ETA: 48s - loss: 1.1116 - acc: 0.70 - ETA: 48s - loss: 1.1008 - acc: 0.71 - ETA: 48s - loss: 1.1060 - acc: 0.71 - ETA: 48s - loss: 1.1095 - acc: 0.71 - ETA: 48s - loss: 1.1137 - acc: 0.71 - ETA: 48s - loss: 1.1133 - acc: 0.71 - ETA: 47s - loss: 1.1123 - acc: 0.71 - ETA: 47s - loss: 1.1342 - acc: 0.70 - ETA: 47s - loss: 1.1398 - acc: 0.70 - ETA: 47s - loss: 1.1420 - acc: 0.70 - ETA: 47s - loss: 1.1498 - acc: 0.70 - ETA: 46s - loss: 1.1449 - acc: 0.70 - ETA: 46s - loss: 1.1519 - acc: 0.70 - ETA: 46s - loss: 1.1529 - acc: 0.70 - ETA: 46s - loss: 1.1501 - acc: 0.70 - ETA: 46s - loss: 1.1495 - acc: 0.70 - ETA: 46s - loss: 1.1472 - acc: 0.70 - ETA: 45s - loss: 1.1450 - acc: 0.70 - ETA: 45s - loss: 1.1447 - acc: 0.71 - ETA: 45s - loss: 1.1486 - acc: 0.70 - ETA: 45s - loss: 1.1434 - acc: 0.71 - ETA: 45s - loss: 1.1545 - acc: 0.70 - ETA: 44s - loss: 1.1493 - acc: 0.70 - ETA: 44s - loss: 1.1477 - acc: 0.71 - ETA: 44s - loss: 1.1517 - acc: 0.70 - ETA: 44s - loss: 1.1470 - acc: 0.71 - ETA: 44s - loss: 1.1467 - acc: 0.70 - ETA: 44s - loss: 1.1462 - acc: 0.70 - ETA: 43s - loss: 1.1399 - acc: 0.71 - ETA: 43s - loss: 1.1461 - acc: 0.71 - ETA: 43s - loss: 1.1431 - acc: 0.71 - ETA: 43s - loss: 1.1487 - acc: 0.71 - ETA: 43s - loss: 1.1463 - acc: 0.71 - ETA: 43s - loss: 1.1468 - acc: 0.71 - ETA: 42s - loss: 1.1453 - acc: 0.71 - ETA: 42s - loss: 1.1429 - acc: 0.71 - ETA: 42s - loss: 1.1432 - acc: 0.71 - ETA: 42s - loss: 1.1480 - acc: 0.71 - ETA: 42s - loss: 1.1451 - acc: 0.71 - ETA: 41s - loss: 1.1450 - acc: 0.71 - ETA: 41s - loss: 1.1463 - acc: 0.70 - ETA: 41s - loss: 1.1488 - acc: 0.70 - ETA: 41s - loss: 1.1515 - acc: 0.70 - ETA: 41s - loss: 1.1592 - acc: 0.70 - ETA: 41s - loss: 1.1543 - acc: 0.70 - ETA: 40s - loss: 1.1496 - acc: 0.70 - ETA: 40s - loss: 1.1511 - acc: 0.70 - ETA: 40s - loss: 1.1468 - acc: 0.70 - ETA: 40s - loss: 1.1463 - acc: 0.70 - ETA: 40s - loss: 1.1420 - acc: 0.71 - ETA: 40s - loss: 1.1442 - acc: 0.71 - ETA: 39s - loss: 1.1446 - acc: 0.71 - ETA: 39s - loss: 1.1410 - acc: 0.71 - ETA: 39s - loss: 1.1381 - acc: 0.71 - ETA: 39s - loss: 1.1392 - acc: 0.71 - ETA: 39s - loss: 1.1428 - acc: 0.71 - ETA: 39s - loss: 1.1459 - acc: 0.70 - ETA: 38s - loss: 1.1423 - acc: 0.70 - ETA: 38s - loss: 1.1398 - acc: 0.70 - ETA: 38s - loss: 1.1392 - acc: 0.70 - ETA: 38s - loss: 1.1467 - acc: 0.70 - ETA: 38s - loss: 1.1490 - acc: 0.70 - ETA: 38s - loss: 1.1500 - acc: 0.70 - ETA: 37s - loss: 1.1473 - acc: 0.70 - ETA: 37s - loss: 1.1468 - acc: 0.70 - ETA: 37s - loss: 1.1462 - acc: 0.70 - ETA: 37s - loss: 1.1500 - acc: 0.70 - ETA: 37s - loss: 1.1538 - acc: 0.70 - ETA: 36s - loss: 1.1529 - acc: 0.70 - ETA: 36s - loss: 1.1579 - acc: 0.70 - ETA: 36s - loss: 1.1570 - acc: 0.70 - ETA: 36s - loss: 1.1558 - acc: 0.70 - ETA: 36s - loss: 1.1512 - acc: 0.70 - ETA: 36s - loss: 1.1490 - acc: 0.70 - ETA: 36s - loss: 1.1470 - acc: 0.70 - ETA: 35s - loss: 1.1477 - acc: 0.70 - ETA: 35s - loss: 1.1450 - acc: 0.70 - ETA: 35s - loss: 1.1440 - acc: 0.70 - ETA: 35s - loss: 1.1428 - acc: 0.70 - ETA: 35s - loss: 1.1425 - acc: 0.70 - ETA: 34s - loss: 1.1419 - acc: 0.70 - ETA: 34s - loss: 1.1404 - acc: 0.70 - ETA: 34s - loss: 1.1405 - acc: 0.70 - ETA: 34s - loss: 1.1477 - acc: 0.70 - ETA: 34s - loss: 1.1486 - acc: 0.70 - ETA: 34s - loss: 1.1507 - acc: 0.70 - ETA: 33s - loss: 1.1511 - acc: 0.70 - ETA: 33s - loss: 1.1477 - acc: 0.70 - ETA: 33s - loss: 1.1492 - acc: 0.70 - ETA: 33s - loss: 1.1465 - acc: 0.70 - ETA: 33s - loss: 1.1495 - acc: 0.70 - ETA: 32s - loss: 1.1475 - acc: 0.70 - ETA: 32s - loss: 1.1453 - acc: 0.70 - ETA: 32s - loss: 1.1446 - acc: 0.70 - ETA: 32s - loss: 1.1450 - acc: 0.70 - ETA: 32s - loss: 1.1490 - acc: 0.70 - ETA: 32s - loss: 1.1533 - acc: 0.70 - ETA: 31s - loss: 1.1545 - acc: 0.70 - ETA: 31s - loss: 1.1597 - acc: 0.70 - ETA: 31s - loss: 1.1602 - acc: 0.70 - ETA: 31s - loss: 1.1591 - acc: 0.70 - ETA: 31s - loss: 1.1599 - acc: 0.70 - ETA: 31s - loss: 1.1639 - acc: 0.70 - ETA: 30s - loss: 1.1613 - acc: 0.70 - ETA: 30s - loss: 1.1605 - acc: 0.70 - ETA: 30s - loss: 1.1582 - acc: 0.70 - ETA: 30s - loss: 1.1562 - acc: 0.70 - ETA: 30s - loss: 1.1566 - acc: 0.70 - ETA: 29s - loss: 1.1560 - acc: 0.70 - ETA: 29s - loss: 1.1569 - acc: 0.70 - ETA: 29s - loss: 1.1546 - acc: 0.70 - ETA: 29s - loss: 1.1580 - acc: 0.70 - ETA: 29s - loss: 1.1630 - acc: 0.70 - ETA: 29s - loss: 1.1627 - acc: 0.70 - ETA: 28s - loss: 1.1651 - acc: 0.70 - ETA: 28s - loss: 1.1642 - acc: 0.70 - ETA: 28s - loss: 1.1629 - acc: 0.70 - ETA: 28s - loss: 1.1604 - acc: 0.70 - ETA: 28s - loss: 1.1617 - acc: 0.70 - ETA: 28s - loss: 1.1610 - acc: 0.70 - ETA: 27s - loss: 1.1581 - acc: 0.70 - ETA: 27s - loss: 1.1580 - acc: 0.70 - ETA: 27s - loss: 1.1559 - acc: 0.70 - ETA: 27s - loss: 1.1542 - acc: 0.70 - ETA: 27s - loss: 1.1569 - acc: 0.70 - ETA: 26s - loss: 1.1596 - acc: 0.70 - ETA: 26s - loss: 1.1587 - acc: 0.70 - ETA: 26s - loss: 1.1551 - acc: 0.70 - ETA: 26s - loss: 1.1570 - acc: 0.70 - ETA: 26s - loss: 1.1590 - acc: 0.70 - ETA: 26s - loss: 1.1614 - acc: 0.69 - ETA: 25s - loss: 1.1635 - acc: 0.69 - ETA: 25s - loss: 1.1608 - acc: 0.70 - ETA: 25s - loss: 1.1595 - acc: 0.70 - ETA: 25s - loss: 1.1613 - acc: 0.69 - ETA: 25s - loss: 1.1593 - acc: 0.70 - ETA: 25s - loss: 1.1563 - acc: 0.70 - ETA: 24s - loss: 1.1535 - acc: 0.70 - ETA: 24s - loss: 1.1563 - acc: 0.70 - ETA: 24s - loss: 1.1565 - acc: 0.70 - ETA: 24s - loss: 1.1564 - acc: 0.69 - ETA: 24s - loss: 1.1609 - acc: 0.69 - ETA: 24s - loss: 1.1598 - acc: 0.69 - ETA: 23s - loss: 1.1629 - acc: 0.69 - ETA: 23s - loss: 1.1628 - acc: 0.69 - ETA: 23s - loss: 1.1637 - acc: 0.69 - ETA: 23s - loss: 1.1663 - acc: 0.69 - ETA: 23s - loss: 1.1686 - acc: 0.69 - ETA: 22s - loss: 1.1669 - acc: 0.69 - ETA: 22s - loss: 1.1643 - acc: 0.69 - ETA: 22s - loss: 1.1648 - acc: 0.69 - ETA: 22s - loss: 1.1663 - acc: 0.69 - ETA: 22s - loss: 1.1710 - acc: 0.69 - ETA: 22s - loss: 1.1740 - acc: 0.69 - ETA: 21s - loss: 1.1710 - acc: 0.69 - ETA: 21s - loss: 1.1712 - acc: 0.69 - ETA: 21s - loss: 1.1714 - acc: 0.69 - ETA: 21s - loss: 1.1676 - acc: 0.69 - ETA: 21s - loss: 1.1676 - acc: 0.69 - ETA: 21s - loss: 1.1666 - acc: 0.69 - ETA: 20s - loss: 1.1674 - acc: 0.69 - ETA: 20s - loss: 1.1658 - acc: 0.69 - ETA: 20s - loss: 1.1653 - acc: 0.69 - ETA: 20s - loss: 1.1648 - acc: 0.69 - ETA: 20s - loss: 1.1662 - acc: 0.69886660/6680 [============================>.] - ETA: 20s - loss: 1.1692 - acc: 0.69 - ETA: 19s - loss: 1.1729 - acc: 0.69 - ETA: 19s - loss: 1.1716 - acc: 0.69 - ETA: 19s - loss: 1.1689 - acc: 0.69 - ETA: 19s - loss: 1.1686 - acc: 0.69 - ETA: 19s - loss: 1.1695 - acc: 0.69 - ETA: 19s - loss: 1.1715 - acc: 0.69 - ETA: 18s - loss: 1.1719 - acc: 0.69 - ETA: 18s - loss: 1.1717 - acc: 0.69 - ETA: 18s - loss: 1.1726 - acc: 0.69 - ETA: 18s - loss: 1.1732 - acc: 0.69 - ETA: 18s - loss: 1.1707 - acc: 0.69 - ETA: 18s - loss: 1.1708 - acc: 0.69 - ETA: 17s - loss: 1.1691 - acc: 0.69 - ETA: 17s - loss: 1.1720 - acc: 0.69 - ETA: 17s - loss: 1.1739 - acc: 0.69 - ETA: 17s - loss: 1.1733 - acc: 0.69 - ETA: 17s - loss: 1.1751 - acc: 0.69 - ETA: 16s - loss: 1.1744 - acc: 0.69 - ETA: 16s - loss: 1.1751 - acc: 0.69 - ETA: 16s - loss: 1.1750 - acc: 0.69 - ETA: 16s - loss: 1.1763 - acc: 0.69 - ETA: 16s - loss: 1.1756 - acc: 0.69 - ETA: 16s - loss: 1.1743 - acc: 0.69 - ETA: 15s - loss: 1.1771 - acc: 0.69 - ETA: 15s - loss: 1.1789 - acc: 0.69 - ETA: 15s - loss: 1.1788 - acc: 0.69 - ETA: 15s - loss: 1.1811 - acc: 0.69 - ETA: 15s - loss: 1.1826 - acc: 0.69 - ETA: 15s - loss: 1.1816 - acc: 0.69 - ETA: 14s - loss: 1.1812 - acc: 0.69 - ETA: 14s - loss: 1.1828 - acc: 0.69 - ETA: 14s - loss: 1.1843 - acc: 0.69 - ETA: 14s - loss: 1.1844 - acc: 0.69 - ETA: 14s - loss: 1.1834 - acc: 0.69 - ETA: 14s - loss: 1.1835 - acc: 0.69 - ETA: 13s - loss: 1.1839 - acc: 0.69 - ETA: 13s - loss: 1.1841 - acc: 0.69 - ETA: 13s - loss: 1.1849 - acc: 0.69 - ETA: 13s - loss: 1.1868 - acc: 0.69 - ETA: 13s - loss: 1.1892 - acc: 0.69 - ETA: 13s - loss: 1.1897 - acc: 0.69 - ETA: 12s - loss: 1.1902 - acc: 0.69 - ETA: 12s - loss: 1.1915 - acc: 0.69 - ETA: 12s - loss: 1.1901 - acc: 0.69 - ETA: 12s - loss: 1.1920 - acc: 0.69 - ETA: 12s - loss: 1.1904 - acc: 0.69 - ETA: 12s - loss: 1.1913 - acc: 0.69 - ETA: 11s - loss: 1.1901 - acc: 0.69 - ETA: 11s - loss: 1.1901 - acc: 0.69 - ETA: 11s - loss: 1.1895 - acc: 0.69 - ETA: 11s - loss: 1.1893 - acc: 0.69 - ETA: 11s - loss: 1.1886 - acc: 0.69 - ETA: 11s - loss: 1.1934 - acc: 0.69 - ETA: 10s - loss: 1.1951 - acc: 0.69 - ETA: 10s - loss: 1.1954 - acc: 0.69 - ETA: 10s - loss: 1.1967 - acc: 0.69 - ETA: 10s - loss: 1.1946 - acc: 0.69 - ETA: 10s - loss: 1.1932 - acc: 0.69 - ETA: 10s - loss: 1.1927 - acc: 0.69 - ETA: 9s - loss: 1.1937 - acc: 0.6926 - ETA: 9s - loss: 1.1914 - acc: 0.693 - ETA: 9s - loss: 1.1938 - acc: 0.692 - ETA: 9s - loss: 1.1924 - acc: 0.693 - ETA: 9s - loss: 1.1921 - acc: 0.693 - ETA: 9s - loss: 1.1914 - acc: 0.693 - ETA: 8s - loss: 1.1909 - acc: 0.693 - ETA: 8s - loss: 1.1924 - acc: 0.693 - ETA: 8s - loss: 1.1907 - acc: 0.693 - ETA: 8s - loss: 1.1893 - acc: 0.694 - ETA: 8s - loss: 1.1904 - acc: 0.693 - ETA: 7s - loss: 1.1915 - acc: 0.693 - ETA: 7s - loss: 1.1916 - acc: 0.693 - ETA: 7s - loss: 1.1921 - acc: 0.693 - ETA: 7s - loss: 1.1918 - acc: 0.693 - ETA: 7s - loss: 1.1944 - acc: 0.692 - ETA: 7s - loss: 1.1942 - acc: 0.692 - ETA: 6s - loss: 1.1949 - acc: 0.692 - ETA: 6s - loss: 1.1944 - acc: 0.692 - ETA: 6s - loss: 1.1984 - acc: 0.691 - ETA: 6s - loss: 1.1994 - acc: 0.691 - ETA: 6s - loss: 1.2014 - acc: 0.690 - ETA: 6s - loss: 1.2010 - acc: 0.690 - ETA: 5s - loss: 1.2024 - acc: 0.690 - ETA: 5s - loss: 1.2015 - acc: 0.690 - ETA: 5s - loss: 1.2023 - acc: 0.690 - ETA: 5s - loss: 1.2016 - acc: 0.690 - ETA: 5s - loss: 1.2036 - acc: 0.689 - ETA: 5s - loss: 1.2041 - acc: 0.688 - ETA: 4s - loss: 1.2040 - acc: 0.688 - ETA: 4s - loss: 1.2046 - acc: 0.688 - ETA: 4s - loss: 1.2053 - acc: 0.688 - ETA: 4s - loss: 1.2079 - acc: 0.687 - ETA: 4s - loss: 1.2116 - acc: 0.687 - ETA: 4s - loss: 1.2111 - acc: 0.687 - ETA: 3s - loss: 1.2099 - acc: 0.687 - ETA: 3s - loss: 1.2111 - acc: 0.687 - ETA: 3s - loss: 1.2120 - acc: 0.686 - ETA: 3s - loss: 1.2117 - acc: 0.686 - ETA: 3s - loss: 1.2118 - acc: 0.687 - ETA: 3s - loss: 1.2134 - acc: 0.686 - ETA: 2s - loss: 1.2140 - acc: 0.686 - ETA: 2s - loss: 1.2144 - acc: 0.686 - ETA: 2s - loss: 1.2151 - acc: 0.685 - ETA: 2s - loss: 1.2173 - acc: 0.685 - ETA: 2s - loss: 1.2173 - acc: 0.685 - ETA: 2s - loss: 1.2166 - acc: 0.685 - ETA: 1s - loss: 1.2161 - acc: 0.685 - ETA: 1s - loss: 1.2154 - acc: 0.685 - ETA: 1s - loss: 1.2172 - acc: 0.684 - ETA: 1s - loss: 1.2195 - acc: 0.684 - ETA: 1s - loss: 1.2200 - acc: 0.683 - ETA: 1s - loss: 1.2226 - acc: 0.683 - ETA: 0s - loss: 1.2226 - acc: 0.683 - ETA: 0s - loss: 1.2230 - acc: 0.683 - ETA: 0s - loss: 1.2226 - acc: 0.683 - ETA: 0s - loss: 1.2211 - acc: 0.683 - ETA: 0s - loss: 1.2211 - acc: 0.6838Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 59s 9ms/step - loss: 1.2234 - acc: 0.6832 - val_loss: 5.4611 - val_acc: 0.0790
    Epoch 9/10
    4300/6680 [==================>...........] - ETA: 53s - loss: 1.1241 - acc: 0.70 - ETA: 54s - loss: 0.8451 - acc: 0.77 - ETA: 54s - loss: 0.9130 - acc: 0.78 - ETA: 54s - loss: 0.9136 - acc: 0.77 - ETA: 54s - loss: 0.9124 - acc: 0.75 - ETA: 54s - loss: 0.9465 - acc: 0.74 - ETA: 54s - loss: 0.9675 - acc: 0.74 - ETA: 53s - loss: 1.0028 - acc: 0.74 - ETA: 54s - loss: 0.9532 - acc: 0.75 - ETA: 54s - loss: 0.9498 - acc: 0.75 - ETA: 54s - loss: 0.9368 - acc: 0.75 - ETA: 54s - loss: 0.9119 - acc: 0.76 - ETA: 54s - loss: 0.9224 - acc: 0.76 - ETA: 54s - loss: 0.8957 - acc: 0.76 - ETA: 53s - loss: 0.8802 - acc: 0.77 - ETA: 53s - loss: 0.8839 - acc: 0.77 - ETA: 53s - loss: 0.8435 - acc: 0.78 - ETA: 53s - loss: 0.8233 - acc: 0.78 - ETA: 53s - loss: 0.8538 - acc: 0.77 - ETA: 52s - loss: 0.8428 - acc: 0.78 - ETA: 52s - loss: 0.8388 - acc: 0.77 - ETA: 52s - loss: 0.8449 - acc: 0.78 - ETA: 52s - loss: 0.8409 - acc: 0.78 - ETA: 52s - loss: 0.8268 - acc: 0.78 - ETA: 52s - loss: 0.8154 - acc: 0.78 - ETA: 51s - loss: 0.8243 - acc: 0.78 - ETA: 51s - loss: 0.8286 - acc: 0.78 - ETA: 51s - loss: 0.8123 - acc: 0.78 - ETA: 51s - loss: 0.8079 - acc: 0.79 - ETA: 51s - loss: 0.8203 - acc: 0.78 - ETA: 51s - loss: 0.8204 - acc: 0.78 - ETA: 50s - loss: 0.8219 - acc: 0.78 - ETA: 50s - loss: 0.8420 - acc: 0.77 - ETA: 50s - loss: 0.8348 - acc: 0.77 - ETA: 50s - loss: 0.8358 - acc: 0.78 - ETA: 50s - loss: 0.8322 - acc: 0.78 - ETA: 50s - loss: 0.8417 - acc: 0.78 - ETA: 50s - loss: 0.8570 - acc: 0.77 - ETA: 50s - loss: 0.8527 - acc: 0.77 - ETA: 49s - loss: 0.8528 - acc: 0.77 - ETA: 49s - loss: 0.8441 - acc: 0.77 - ETA: 49s - loss: 0.8330 - acc: 0.78 - ETA: 49s - loss: 0.8242 - acc: 0.78 - ETA: 49s - loss: 0.8178 - acc: 0.78 - ETA: 48s - loss: 0.8194 - acc: 0.78 - ETA: 48s - loss: 0.8295 - acc: 0.78 - ETA: 48s - loss: 0.8208 - acc: 0.78 - ETA: 48s - loss: 0.8228 - acc: 0.78 - ETA: 48s - loss: 0.8255 - acc: 0.78 - ETA: 48s - loss: 0.8247 - acc: 0.78 - ETA: 47s - loss: 0.8206 - acc: 0.78 - ETA: 47s - loss: 0.8214 - acc: 0.78 - ETA: 47s - loss: 0.8198 - acc: 0.78 - ETA: 47s - loss: 0.8174 - acc: 0.78 - ETA: 47s - loss: 0.8247 - acc: 0.78 - ETA: 46s - loss: 0.8284 - acc: 0.78 - ETA: 46s - loss: 0.8309 - acc: 0.78 - ETA: 46s - loss: 0.8299 - acc: 0.78 - ETA: 46s - loss: 0.8317 - acc: 0.78 - ETA: 46s - loss: 0.8347 - acc: 0.78 - ETA: 46s - loss: 0.8374 - acc: 0.78 - ETA: 45s - loss: 0.8420 - acc: 0.78 - ETA: 45s - loss: 0.8410 - acc: 0.78 - ETA: 45s - loss: 0.8413 - acc: 0.78 - ETA: 45s - loss: 0.8431 - acc: 0.78 - ETA: 45s - loss: 0.8506 - acc: 0.77 - ETA: 44s - loss: 0.8500 - acc: 0.77 - ETA: 44s - loss: 0.8511 - acc: 0.77 - ETA: 44s - loss: 0.8621 - acc: 0.77 - ETA: 44s - loss: 0.8617 - acc: 0.77 - ETA: 44s - loss: 0.8599 - acc: 0.77 - ETA: 44s - loss: 0.8553 - acc: 0.77 - ETA: 44s - loss: 0.8613 - acc: 0.77 - ETA: 43s - loss: 0.8587 - acc: 0.77 - ETA: 43s - loss: 0.8557 - acc: 0.77 - ETA: 43s - loss: 0.8556 - acc: 0.77 - ETA: 43s - loss: 0.8528 - acc: 0.77 - ETA: 43s - loss: 0.8476 - acc: 0.77 - ETA: 43s - loss: 0.8441 - acc: 0.77 - ETA: 42s - loss: 0.8484 - acc: 0.77 - ETA: 42s - loss: 0.8555 - acc: 0.77 - ETA: 42s - loss: 0.8542 - acc: 0.77 - ETA: 42s - loss: 0.8483 - acc: 0.77 - ETA: 42s - loss: 0.8507 - acc: 0.77 - ETA: 42s - loss: 0.8520 - acc: 0.77 - ETA: 41s - loss: 0.8567 - acc: 0.77 - ETA: 41s - loss: 0.8593 - acc: 0.77 - ETA: 41s - loss: 0.8560 - acc: 0.77 - ETA: 41s - loss: 0.8557 - acc: 0.77 - ETA: 41s - loss: 0.8589 - acc: 0.77 - ETA: 41s - loss: 0.8580 - acc: 0.77 - ETA: 40s - loss: 0.8590 - acc: 0.77 - ETA: 40s - loss: 0.8581 - acc: 0.77 - ETA: 40s - loss: 0.8564 - acc: 0.77 - ETA: 40s - loss: 0.8579 - acc: 0.77 - ETA: 40s - loss: 0.8597 - acc: 0.77 - ETA: 40s - loss: 0.8559 - acc: 0.77 - ETA: 39s - loss: 0.8565 - acc: 0.77 - ETA: 39s - loss: 0.8551 - acc: 0.77 - ETA: 39s - loss: 0.8552 - acc: 0.78 - ETA: 39s - loss: 0.8523 - acc: 0.78 - ETA: 39s - loss: 0.8486 - acc: 0.78 - ETA: 38s - loss: 0.8471 - acc: 0.78 - ETA: 38s - loss: 0.8441 - acc: 0.78 - ETA: 38s - loss: 0.8465 - acc: 0.78 - ETA: 38s - loss: 0.8454 - acc: 0.78 - ETA: 38s - loss: 0.8445 - acc: 0.78 - ETA: 38s - loss: 0.8404 - acc: 0.78 - ETA: 37s - loss: 0.8402 - acc: 0.78 - ETA: 37s - loss: 0.8379 - acc: 0.78 - ETA: 37s - loss: 0.8394 - acc: 0.78 - ETA: 37s - loss: 0.8419 - acc: 0.78 - ETA: 37s - loss: 0.8441 - acc: 0.78 - ETA: 37s - loss: 0.8443 - acc: 0.78 - ETA: 36s - loss: 0.8460 - acc: 0.78 - ETA: 36s - loss: 0.8423 - acc: 0.78 - ETA: 36s - loss: 0.8444 - acc: 0.78 - ETA: 36s - loss: 0.8444 - acc: 0.78 - ETA: 36s - loss: 0.8454 - acc: 0.78 - ETA: 36s - loss: 0.8489 - acc: 0.78 - ETA: 35s - loss: 0.8489 - acc: 0.78 - ETA: 35s - loss: 0.8497 - acc: 0.78 - ETA: 35s - loss: 0.8504 - acc: 0.78 - ETA: 35s - loss: 0.8616 - acc: 0.78 - ETA: 35s - loss: 0.8690 - acc: 0.77 - ETA: 35s - loss: 0.8639 - acc: 0.77 - ETA: 34s - loss: 0.8628 - acc: 0.77 - ETA: 34s - loss: 0.8692 - acc: 0.77 - ETA: 34s - loss: 0.8727 - acc: 0.77 - ETA: 34s - loss: 0.8746 - acc: 0.77 - ETA: 34s - loss: 0.8765 - acc: 0.77 - ETA: 34s - loss: 0.8727 - acc: 0.77 - ETA: 33s - loss: 0.8740 - acc: 0.77 - ETA: 33s - loss: 0.8785 - acc: 0.77 - ETA: 33s - loss: 0.8771 - acc: 0.77 - ETA: 33s - loss: 0.8746 - acc: 0.77 - ETA: 33s - loss: 0.8749 - acc: 0.77 - ETA: 33s - loss: 0.8743 - acc: 0.77 - ETA: 32s - loss: 0.8760 - acc: 0.77 - ETA: 32s - loss: 0.8739 - acc: 0.77 - ETA: 32s - loss: 0.8743 - acc: 0.77 - ETA: 32s - loss: 0.8744 - acc: 0.77 - ETA: 32s - loss: 0.8729 - acc: 0.77 - ETA: 32s - loss: 0.8720 - acc: 0.77 - ETA: 31s - loss: 0.8734 - acc: 0.77 - ETA: 31s - loss: 0.8749 - acc: 0.77 - ETA: 31s - loss: 0.8765 - acc: 0.77 - ETA: 31s - loss: 0.8791 - acc: 0.77 - ETA: 31s - loss: 0.8773 - acc: 0.77 - ETA: 31s - loss: 0.8777 - acc: 0.77 - ETA: 30s - loss: 0.8765 - acc: 0.77 - ETA: 30s - loss: 0.8747 - acc: 0.77 - ETA: 30s - loss: 0.8743 - acc: 0.77 - ETA: 30s - loss: 0.8744 - acc: 0.77 - ETA: 30s - loss: 0.8766 - acc: 0.77 - ETA: 30s - loss: 0.8757 - acc: 0.77 - ETA: 29s - loss: 0.8785 - acc: 0.77 - ETA: 29s - loss: 0.8835 - acc: 0.77 - ETA: 29s - loss: 0.8870 - acc: 0.77 - ETA: 29s - loss: 0.8907 - acc: 0.76 - ETA: 29s - loss: 0.8930 - acc: 0.76 - ETA: 29s - loss: 0.8923 - acc: 0.76 - ETA: 28s - loss: 0.8921 - acc: 0.76 - ETA: 28s - loss: 0.8931 - acc: 0.76 - ETA: 28s - loss: 0.8936 - acc: 0.76 - ETA: 28s - loss: 0.8924 - acc: 0.76 - ETA: 28s - loss: 0.8937 - acc: 0.76 - ETA: 28s - loss: 0.8941 - acc: 0.76 - ETA: 27s - loss: 0.8944 - acc: 0.76 - ETA: 27s - loss: 0.8940 - acc: 0.76 - ETA: 27s - loss: 0.8914 - acc: 0.76 - ETA: 27s - loss: 0.8895 - acc: 0.76 - ETA: 27s - loss: 0.8910 - acc: 0.76 - ETA: 27s - loss: 0.8937 - acc: 0.76 - ETA: 26s - loss: 0.8967 - acc: 0.76 - ETA: 26s - loss: 0.8962 - acc: 0.76 - ETA: 26s - loss: 0.8966 - acc: 0.76 - ETA: 26s - loss: 0.9007 - acc: 0.76 - ETA: 26s - loss: 0.8994 - acc: 0.76 - ETA: 26s - loss: 0.9009 - acc: 0.76 - ETA: 25s - loss: 0.9046 - acc: 0.76 - ETA: 25s - loss: 0.9027 - acc: 0.76 - ETA: 25s - loss: 0.9018 - acc: 0.76 - ETA: 25s - loss: 0.9057 - acc: 0.76 - ETA: 25s - loss: 0.9050 - acc: 0.76 - ETA: 25s - loss: 0.9036 - acc: 0.76 - ETA: 24s - loss: 0.9025 - acc: 0.76 - ETA: 24s - loss: 0.9018 - acc: 0.76 - ETA: 24s - loss: 0.9003 - acc: 0.76 - ETA: 24s - loss: 0.8983 - acc: 0.76 - ETA: 24s - loss: 0.8980 - acc: 0.76 - ETA: 24s - loss: 0.9011 - acc: 0.76 - ETA: 23s - loss: 0.9020 - acc: 0.76 - ETA: 23s - loss: 0.9036 - acc: 0.76 - ETA: 23s - loss: 0.9062 - acc: 0.76 - ETA: 23s - loss: 0.9094 - acc: 0.76 - ETA: 23s - loss: 0.9070 - acc: 0.76 - ETA: 23s - loss: 0.9066 - acc: 0.76 - ETA: 22s - loss: 0.9051 - acc: 0.76 - ETA: 22s - loss: 0.9052 - acc: 0.76 - ETA: 22s - loss: 0.9053 - acc: 0.76 - ETA: 22s - loss: 0.9043 - acc: 0.76 - ETA: 22s - loss: 0.9050 - acc: 0.76 - ETA: 22s - loss: 0.9034 - acc: 0.76 - ETA: 21s - loss: 0.9077 - acc: 0.76 - ETA: 21s - loss: 0.9104 - acc: 0.76 - ETA: 21s - loss: 0.9082 - acc: 0.76 - ETA: 21s - loss: 0.9085 - acc: 0.76 - ETA: 21s - loss: 0.9085 - acc: 0.76 - ETA: 21s - loss: 0.9088 - acc: 0.76 - ETA: 20s - loss: 0.9095 - acc: 0.76 - ETA: 20s - loss: 0.9090 - acc: 0.76 - ETA: 20s - loss: 0.9133 - acc: 0.76 - ETA: 20s - loss: 0.9129 - acc: 0.76 - ETA: 20s - loss: 0.9123 - acc: 0.76376660/6680 [============================>.] - ETA: 19s - loss: 0.9114 - acc: 0.76 - ETA: 19s - loss: 0.9094 - acc: 0.76 - ETA: 19s - loss: 0.9092 - acc: 0.76 - ETA: 19s - loss: 0.9097 - acc: 0.76 - ETA: 19s - loss: 0.9083 - acc: 0.76 - ETA: 19s - loss: 0.9066 - acc: 0.76 - ETA: 18s - loss: 0.9071 - acc: 0.76 - ETA: 18s - loss: 0.9089 - acc: 0.76 - ETA: 18s - loss: 0.9092 - acc: 0.76 - ETA: 18s - loss: 0.9092 - acc: 0.76 - ETA: 18s - loss: 0.9069 - acc: 0.76 - ETA: 18s - loss: 0.9082 - acc: 0.76 - ETA: 17s - loss: 0.9057 - acc: 0.76 - ETA: 17s - loss: 0.9082 - acc: 0.76 - ETA: 17s - loss: 0.9078 - acc: 0.76 - ETA: 17s - loss: 0.9084 - acc: 0.76 - ETA: 17s - loss: 0.9084 - acc: 0.76 - ETA: 17s - loss: 0.9104 - acc: 0.76 - ETA: 16s - loss: 0.9109 - acc: 0.76 - ETA: 16s - loss: 0.9107 - acc: 0.76 - ETA: 16s - loss: 0.9122 - acc: 0.76 - ETA: 16s - loss: 0.9132 - acc: 0.76 - ETA: 16s - loss: 0.9129 - acc: 0.76 - ETA: 16s - loss: 0.9157 - acc: 0.75 - ETA: 15s - loss: 0.9172 - acc: 0.75 - ETA: 15s - loss: 0.9193 - acc: 0.75 - ETA: 15s - loss: 0.9201 - acc: 0.75 - ETA: 15s - loss: 0.9218 - acc: 0.75 - ETA: 15s - loss: 0.9201 - acc: 0.75 - ETA: 15s - loss: 0.9207 - acc: 0.75 - ETA: 14s - loss: 0.9220 - acc: 0.75 - ETA: 14s - loss: 0.9228 - acc: 0.75 - ETA: 14s - loss: 0.9225 - acc: 0.75 - ETA: 14s - loss: 0.9217 - acc: 0.75 - ETA: 14s - loss: 0.9224 - acc: 0.75 - ETA: 14s - loss: 0.9255 - acc: 0.75 - ETA: 13s - loss: 0.9271 - acc: 0.75 - ETA: 13s - loss: 0.9271 - acc: 0.75 - ETA: 13s - loss: 0.9270 - acc: 0.75 - ETA: 13s - loss: 0.9255 - acc: 0.75 - ETA: 13s - loss: 0.9246 - acc: 0.75 - ETA: 13s - loss: 0.9237 - acc: 0.75 - ETA: 12s - loss: 0.9262 - acc: 0.75 - ETA: 12s - loss: 0.9270 - acc: 0.75 - ETA: 12s - loss: 0.9277 - acc: 0.75 - ETA: 12s - loss: 0.9277 - acc: 0.75 - ETA: 12s - loss: 0.9303 - acc: 0.75 - ETA: 12s - loss: 0.9310 - acc: 0.75 - ETA: 11s - loss: 0.9322 - acc: 0.75 - ETA: 11s - loss: 0.9317 - acc: 0.75 - ETA: 11s - loss: 0.9334 - acc: 0.75 - ETA: 11s - loss: 0.9336 - acc: 0.75 - ETA: 11s - loss: 0.9343 - acc: 0.75 - ETA: 11s - loss: 0.9352 - acc: 0.75 - ETA: 10s - loss: 0.9359 - acc: 0.75 - ETA: 10s - loss: 0.9359 - acc: 0.75 - ETA: 10s - loss: 0.9370 - acc: 0.75 - ETA: 10s - loss: 0.9364 - acc: 0.75 - ETA: 10s - loss: 0.9380 - acc: 0.75 - ETA: 10s - loss: 0.9396 - acc: 0.75 - ETA: 9s - loss: 0.9387 - acc: 0.7529 - ETA: 9s - loss: 0.9388 - acc: 0.752 - ETA: 9s - loss: 0.9395 - acc: 0.752 - ETA: 9s - loss: 0.9398 - acc: 0.752 - ETA: 9s - loss: 0.9392 - acc: 0.752 - ETA: 8s - loss: 0.9395 - acc: 0.752 - ETA: 8s - loss: 0.9392 - acc: 0.752 - ETA: 8s - loss: 0.9376 - acc: 0.752 - ETA: 8s - loss: 0.9372 - acc: 0.752 - ETA: 8s - loss: 0.9358 - acc: 0.753 - ETA: 8s - loss: 0.9351 - acc: 0.753 - ETA: 7s - loss: 0.9342 - acc: 0.753 - ETA: 7s - loss: 0.9340 - acc: 0.753 - ETA: 7s - loss: 0.9353 - acc: 0.752 - ETA: 7s - loss: 0.9353 - acc: 0.752 - ETA: 7s - loss: 0.9354 - acc: 0.752 - ETA: 7s - loss: 0.9368 - acc: 0.752 - ETA: 6s - loss: 0.9367 - acc: 0.752 - ETA: 6s - loss: 0.9372 - acc: 0.752 - ETA: 6s - loss: 0.9368 - acc: 0.752 - ETA: 6s - loss: 0.9361 - acc: 0.752 - ETA: 6s - loss: 0.9371 - acc: 0.752 - ETA: 6s - loss: 0.9378 - acc: 0.751 - ETA: 5s - loss: 0.9394 - acc: 0.751 - ETA: 5s - loss: 0.9397 - acc: 0.751 - ETA: 5s - loss: 0.9405 - acc: 0.751 - ETA: 5s - loss: 0.9397 - acc: 0.751 - ETA: 5s - loss: 0.9387 - acc: 0.751 - ETA: 5s - loss: 0.9402 - acc: 0.751 - ETA: 4s - loss: 0.9397 - acc: 0.751 - ETA: 4s - loss: 0.9397 - acc: 0.751 - ETA: 4s - loss: 0.9394 - acc: 0.751 - ETA: 4s - loss: 0.9378 - acc: 0.751 - ETA: 4s - loss: 0.9387 - acc: 0.751 - ETA: 4s - loss: 0.9386 - acc: 0.751 - ETA: 3s - loss: 0.9377 - acc: 0.751 - ETA: 3s - loss: 0.9386 - acc: 0.751 - ETA: 3s - loss: 0.9413 - acc: 0.750 - ETA: 3s - loss: 0.9426 - acc: 0.750 - ETA: 3s - loss: 0.9424 - acc: 0.750 - ETA: 3s - loss: 0.9420 - acc: 0.750 - ETA: 2s - loss: 0.9432 - acc: 0.750 - ETA: 2s - loss: 0.9443 - acc: 0.750 - ETA: 2s - loss: 0.9425 - acc: 0.750 - ETA: 2s - loss: 0.9423 - acc: 0.750 - ETA: 2s - loss: 0.9438 - acc: 0.750 - ETA: 2s - loss: 0.9445 - acc: 0.750 - ETA: 1s - loss: 0.9445 - acc: 0.750 - ETA: 1s - loss: 0.9437 - acc: 0.750 - ETA: 1s - loss: 0.9447 - acc: 0.750 - ETA: 1s - loss: 0.9438 - acc: 0.751 - ETA: 1s - loss: 0.9428 - acc: 0.751 - ETA: 1s - loss: 0.9436 - acc: 0.751 - ETA: 0s - loss: 0.9431 - acc: 0.751 - ETA: 0s - loss: 0.9432 - acc: 0.751 - ETA: 0s - loss: 0.9438 - acc: 0.751 - ETA: 0s - loss: 0.9447 - acc: 0.751 - ETA: 0s - loss: 0.9449 - acc: 0.7511Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 59s 9ms/step - loss: 0.9448 - acc: 0.7512 - val_loss: 5.4748 - val_acc: 0.0946
    Epoch 10/10
    4300/6680 [==================>...........] - ETA: 54s - loss: 0.7621 - acc: 0.80 - ETA: 57s - loss: 0.6122 - acc: 0.82 - ETA: 58s - loss: 0.7243 - acc: 0.80 - ETA: 57s - loss: 0.7778 - acc: 0.77 - ETA: 57s - loss: 0.7548 - acc: 0.79 - ETA: 57s - loss: 0.8398 - acc: 0.77 - ETA: 57s - loss: 0.7858 - acc: 0.79 - ETA: 57s - loss: 0.7424 - acc: 0.80 - ETA: 57s - loss: 0.6969 - acc: 0.81 - ETA: 56s - loss: 0.6759 - acc: 0.82 - ETA: 56s - loss: 0.6543 - acc: 0.83 - ETA: 55s - loss: 0.6294 - acc: 0.83 - ETA: 56s - loss: 0.6194 - acc: 0.83 - ETA: 55s - loss: 0.6120 - acc: 0.83 - ETA: 55s - loss: 0.6409 - acc: 0.83 - ETA: 55s - loss: 0.6442 - acc: 0.83 - ETA: 54s - loss: 0.6647 - acc: 0.82 - ETA: 54s - loss: 0.6584 - acc: 0.82 - ETA: 54s - loss: 0.6460 - acc: 0.83 - ETA: 53s - loss: 0.6409 - acc: 0.83 - ETA: 53s - loss: 0.6470 - acc: 0.83 - ETA: 53s - loss: 0.6472 - acc: 0.83 - ETA: 53s - loss: 0.6490 - acc: 0.83 - ETA: 53s - loss: 0.6640 - acc: 0.82 - ETA: 52s - loss: 0.6600 - acc: 0.82 - ETA: 52s - loss: 0.6456 - acc: 0.83 - ETA: 52s - loss: 0.6347 - acc: 0.83 - ETA: 52s - loss: 0.6324 - acc: 0.83 - ETA: 51s - loss: 0.6260 - acc: 0.83 - ETA: 51s - loss: 0.6294 - acc: 0.83 - ETA: 51s - loss: 0.6145 - acc: 0.84 - ETA: 51s - loss: 0.6281 - acc: 0.83 - ETA: 51s - loss: 0.6250 - acc: 0.83 - ETA: 51s - loss: 0.6263 - acc: 0.83 - ETA: 51s - loss: 0.6207 - acc: 0.83 - ETA: 51s - loss: 0.6167 - acc: 0.83 - ETA: 50s - loss: 0.6136 - acc: 0.83 - ETA: 50s - loss: 0.6033 - acc: 0.83 - ETA: 50s - loss: 0.6114 - acc: 0.83 - ETA: 50s - loss: 0.6007 - acc: 0.84 - ETA: 50s - loss: 0.6103 - acc: 0.83 - ETA: 50s - loss: 0.6102 - acc: 0.83 - ETA: 50s - loss: 0.6088 - acc: 0.83 - ETA: 49s - loss: 0.6056 - acc: 0.83 - ETA: 49s - loss: 0.6014 - acc: 0.83 - ETA: 49s - loss: 0.6106 - acc: 0.83 - ETA: 49s - loss: 0.6013 - acc: 0.83 - ETA: 49s - loss: 0.6093 - acc: 0.83 - ETA: 48s - loss: 0.6113 - acc: 0.83 - ETA: 48s - loss: 0.6169 - acc: 0.83 - ETA: 48s - loss: 0.6262 - acc: 0.83 - ETA: 48s - loss: 0.6410 - acc: 0.83 - ETA: 48s - loss: 0.6377 - acc: 0.83 - ETA: 48s - loss: 0.6392 - acc: 0.83 - ETA: 47s - loss: 0.6468 - acc: 0.83 - ETA: 47s - loss: 0.6443 - acc: 0.83 - ETA: 47s - loss: 0.6368 - acc: 0.83 - ETA: 47s - loss: 0.6364 - acc: 0.83 - ETA: 47s - loss: 0.6340 - acc: 0.83 - ETA: 47s - loss: 0.6360 - acc: 0.83 - ETA: 46s - loss: 0.6496 - acc: 0.83 - ETA: 46s - loss: 0.6559 - acc: 0.83 - ETA: 46s - loss: 0.6540 - acc: 0.83 - ETA: 46s - loss: 0.6633 - acc: 0.83 - ETA: 46s - loss: 0.6720 - acc: 0.82 - ETA: 46s - loss: 0.6736 - acc: 0.82 - ETA: 45s - loss: 0.6682 - acc: 0.82 - ETA: 45s - loss: 0.6674 - acc: 0.82 - ETA: 45s - loss: 0.6748 - acc: 0.82 - ETA: 45s - loss: 0.6744 - acc: 0.82 - ETA: 45s - loss: 0.6753 - acc: 0.82 - ETA: 44s - loss: 0.6732 - acc: 0.82 - ETA: 44s - loss: 0.6711 - acc: 0.82 - ETA: 44s - loss: 0.6660 - acc: 0.82 - ETA: 44s - loss: 0.6650 - acc: 0.82 - ETA: 44s - loss: 0.6655 - acc: 0.82 - ETA: 44s - loss: 0.6614 - acc: 0.82 - ETA: 43s - loss: 0.6597 - acc: 0.82 - ETA: 43s - loss: 0.6561 - acc: 0.83 - ETA: 43s - loss: 0.6583 - acc: 0.83 - ETA: 43s - loss: 0.6588 - acc: 0.83 - ETA: 43s - loss: 0.6659 - acc: 0.82 - ETA: 42s - loss: 0.6645 - acc: 0.83 - ETA: 42s - loss: 0.6702 - acc: 0.83 - ETA: 42s - loss: 0.6726 - acc: 0.82 - ETA: 42s - loss: 0.6709 - acc: 0.83 - ETA: 42s - loss: 0.6697 - acc: 0.82 - ETA: 42s - loss: 0.6642 - acc: 0.83 - ETA: 41s - loss: 0.6605 - acc: 0.83 - ETA: 41s - loss: 0.6644 - acc: 0.83 - ETA: 41s - loss: 0.6624 - acc: 0.83 - ETA: 41s - loss: 0.6608 - acc: 0.83 - ETA: 41s - loss: 0.6590 - acc: 0.83 - ETA: 41s - loss: 0.6621 - acc: 0.83 - ETA: 40s - loss: 0.6702 - acc: 0.83 - ETA: 40s - loss: 0.6746 - acc: 0.83 - ETA: 40s - loss: 0.6773 - acc: 0.83 - ETA: 40s - loss: 0.6752 - acc: 0.83 - ETA: 40s - loss: 0.6714 - acc: 0.83 - ETA: 39s - loss: 0.6697 - acc: 0.83 - ETA: 39s - loss: 0.6711 - acc: 0.83 - ETA: 39s - loss: 0.6800 - acc: 0.83 - ETA: 39s - loss: 0.6787 - acc: 0.83 - ETA: 39s - loss: 0.6766 - acc: 0.83 - ETA: 39s - loss: 0.6842 - acc: 0.83 - ETA: 39s - loss: 0.6857 - acc: 0.83 - ETA: 38s - loss: 0.6859 - acc: 0.82 - ETA: 38s - loss: 0.6824 - acc: 0.83 - ETA: 38s - loss: 0.6882 - acc: 0.83 - ETA: 38s - loss: 0.6881 - acc: 0.83 - ETA: 38s - loss: 0.6935 - acc: 0.82 - ETA: 38s - loss: 0.6937 - acc: 0.82 - ETA: 37s - loss: 0.6932 - acc: 0.82 - ETA: 37s - loss: 0.6949 - acc: 0.82 - ETA: 37s - loss: 0.6934 - acc: 0.82 - ETA: 37s - loss: 0.6890 - acc: 0.83 - ETA: 37s - loss: 0.6893 - acc: 0.82 - ETA: 37s - loss: 0.6885 - acc: 0.82 - ETA: 36s - loss: 0.6867 - acc: 0.82 - ETA: 36s - loss: 0.6876 - acc: 0.82 - ETA: 36s - loss: 0.6875 - acc: 0.83 - ETA: 36s - loss: 0.6860 - acc: 0.83 - ETA: 36s - loss: 0.6890 - acc: 0.83 - ETA: 36s - loss: 0.6870 - acc: 0.83 - ETA: 35s - loss: 0.6875 - acc: 0.83 - ETA: 35s - loss: 0.6903 - acc: 0.83 - ETA: 35s - loss: 0.6899 - acc: 0.82 - ETA: 35s - loss: 0.6882 - acc: 0.83 - ETA: 35s - loss: 0.6890 - acc: 0.83 - ETA: 35s - loss: 0.6901 - acc: 0.83 - ETA: 34s - loss: 0.6886 - acc: 0.83 - ETA: 34s - loss: 0.6890 - acc: 0.82 - ETA: 34s - loss: 0.6880 - acc: 0.83 - ETA: 34s - loss: 0.6877 - acc: 0.83 - ETA: 34s - loss: 0.6889 - acc: 0.83 - ETA: 34s - loss: 0.6872 - acc: 0.83 - ETA: 33s - loss: 0.6892 - acc: 0.82 - ETA: 33s - loss: 0.6939 - acc: 0.82 - ETA: 33s - loss: 0.6941 - acc: 0.82 - ETA: 33s - loss: 0.6949 - acc: 0.82 - ETA: 33s - loss: 0.6963 - acc: 0.82 - ETA: 33s - loss: 0.7033 - acc: 0.82 - ETA: 32s - loss: 0.7008 - acc: 0.82 - ETA: 32s - loss: 0.7011 - acc: 0.82 - ETA: 32s - loss: 0.7033 - acc: 0.82 - ETA: 32s - loss: 0.7026 - acc: 0.82 - ETA: 32s - loss: 0.7034 - acc: 0.82 - ETA: 32s - loss: 0.7024 - acc: 0.82 - ETA: 31s - loss: 0.7003 - acc: 0.82 - ETA: 31s - loss: 0.6971 - acc: 0.82 - ETA: 31s - loss: 0.6965 - acc: 0.82 - ETA: 31s - loss: 0.6974 - acc: 0.82 - ETA: 31s - loss: 0.6997 - acc: 0.82 - ETA: 30s - loss: 0.6986 - acc: 0.82 - ETA: 30s - loss: 0.6983 - acc: 0.82 - ETA: 30s - loss: 0.6999 - acc: 0.82 - ETA: 30s - loss: 0.7019 - acc: 0.82 - ETA: 30s - loss: 0.7016 - acc: 0.82 - ETA: 30s - loss: 0.7000 - acc: 0.82 - ETA: 29s - loss: 0.7003 - acc: 0.82 - ETA: 29s - loss: 0.6979 - acc: 0.82 - ETA: 29s - loss: 0.6981 - acc: 0.82 - ETA: 29s - loss: 0.7001 - acc: 0.82 - ETA: 29s - loss: 0.7006 - acc: 0.82 - ETA: 29s - loss: 0.6992 - acc: 0.82 - ETA: 28s - loss: 0.6976 - acc: 0.82 - ETA: 28s - loss: 0.7019 - acc: 0.82 - ETA: 28s - loss: 0.7045 - acc: 0.82 - ETA: 28s - loss: 0.7041 - acc: 0.82 - ETA: 28s - loss: 0.7040 - acc: 0.82 - ETA: 28s - loss: 0.7069 - acc: 0.82 - ETA: 27s - loss: 0.7060 - acc: 0.81 - ETA: 27s - loss: 0.7050 - acc: 0.81 - ETA: 27s - loss: 0.7054 - acc: 0.82 - ETA: 27s - loss: 0.7046 - acc: 0.82 - ETA: 27s - loss: 0.7056 - acc: 0.82 - ETA: 27s - loss: 0.7040 - acc: 0.82 - ETA: 26s - loss: 0.7036 - acc: 0.82 - ETA: 26s - loss: 0.7050 - acc: 0.82 - ETA: 26s - loss: 0.7040 - acc: 0.82 - ETA: 26s - loss: 0.7027 - acc: 0.82 - ETA: 26s - loss: 0.7073 - acc: 0.81 - ETA: 26s - loss: 0.7084 - acc: 0.81 - ETA: 25s - loss: 0.7112 - acc: 0.81 - ETA: 25s - loss: 0.7100 - acc: 0.81 - ETA: 25s - loss: 0.7119 - acc: 0.81 - ETA: 25s - loss: 0.7119 - acc: 0.81 - ETA: 25s - loss: 0.7135 - acc: 0.81 - ETA: 24s - loss: 0.7124 - acc: 0.81 - ETA: 24s - loss: 0.7122 - acc: 0.81 - ETA: 24s - loss: 0.7103 - acc: 0.81 - ETA: 24s - loss: 0.7100 - acc: 0.81 - ETA: 24s - loss: 0.7109 - acc: 0.81 - ETA: 24s - loss: 0.7088 - acc: 0.81 - ETA: 23s - loss: 0.7066 - acc: 0.81 - ETA: 23s - loss: 0.7042 - acc: 0.81 - ETA: 23s - loss: 0.7050 - acc: 0.81 - ETA: 23s - loss: 0.7046 - acc: 0.81 - ETA: 23s - loss: 0.7031 - acc: 0.81 - ETA: 23s - loss: 0.7052 - acc: 0.82 - ETA: 22s - loss: 0.7088 - acc: 0.81 - ETA: 22s - loss: 0.7078 - acc: 0.81 - ETA: 22s - loss: 0.7083 - acc: 0.81 - ETA: 22s - loss: 0.7081 - acc: 0.81 - ETA: 22s - loss: 0.7067 - acc: 0.81 - ETA: 22s - loss: 0.7044 - acc: 0.82 - ETA: 21s - loss: 0.7059 - acc: 0.81 - ETA: 21s - loss: 0.7083 - acc: 0.81 - ETA: 21s - loss: 0.7077 - acc: 0.81 - ETA: 21s - loss: 0.7084 - acc: 0.81 - ETA: 21s - loss: 0.7071 - acc: 0.81 - ETA: 20s - loss: 0.7060 - acc: 0.81 - ETA: 20s - loss: 0.7074 - acc: 0.81 - ETA: 20s - loss: 0.7053 - acc: 0.81 - ETA: 20s - loss: 0.7068 - acc: 0.81956660/6680 [============================>.] - ETA: 20s - loss: 0.7092 - acc: 0.81 - ETA: 20s - loss: 0.7077 - acc: 0.81 - ETA: 19s - loss: 0.7071 - acc: 0.81 - ETA: 19s - loss: 0.7066 - acc: 0.81 - ETA: 19s - loss: 0.7091 - acc: 0.81 - ETA: 19s - loss: 0.7079 - acc: 0.81 - ETA: 19s - loss: 0.7071 - acc: 0.81 - ETA: 19s - loss: 0.7101 - acc: 0.81 - ETA: 18s - loss: 0.7113 - acc: 0.81 - ETA: 18s - loss: 0.7091 - acc: 0.81 - ETA: 18s - loss: 0.7099 - acc: 0.81 - ETA: 18s - loss: 0.7133 - acc: 0.81 - ETA: 18s - loss: 0.7125 - acc: 0.81 - ETA: 18s - loss: 0.7148 - acc: 0.81 - ETA: 17s - loss: 0.7147 - acc: 0.81 - ETA: 17s - loss: 0.7155 - acc: 0.81 - ETA: 17s - loss: 0.7159 - acc: 0.81 - ETA: 17s - loss: 0.7164 - acc: 0.81 - ETA: 17s - loss: 0.7166 - acc: 0.81 - ETA: 17s - loss: 0.7163 - acc: 0.81 - ETA: 16s - loss: 0.7163 - acc: 0.81 - ETA: 16s - loss: 0.7160 - acc: 0.81 - ETA: 16s - loss: 0.7184 - acc: 0.81 - ETA: 16s - loss: 0.7190 - acc: 0.81 - ETA: 16s - loss: 0.7193 - acc: 0.81 - ETA: 16s - loss: 0.7186 - acc: 0.81 - ETA: 15s - loss: 0.7210 - acc: 0.81 - ETA: 15s - loss: 0.7221 - acc: 0.81 - ETA: 15s - loss: 0.7229 - acc: 0.81 - ETA: 15s - loss: 0.7248 - acc: 0.81 - ETA: 15s - loss: 0.7257 - acc: 0.81 - ETA: 14s - loss: 0.7234 - acc: 0.81 - ETA: 14s - loss: 0.7257 - acc: 0.81 - ETA: 14s - loss: 0.7259 - acc: 0.81 - ETA: 14s - loss: 0.7252 - acc: 0.81 - ETA: 14s - loss: 0.7266 - acc: 0.81 - ETA: 14s - loss: 0.7272 - acc: 0.81 - ETA: 13s - loss: 0.7287 - acc: 0.81 - ETA: 13s - loss: 0.7286 - acc: 0.81 - ETA: 13s - loss: 0.7322 - acc: 0.81 - ETA: 13s - loss: 0.7319 - acc: 0.81 - ETA: 13s - loss: 0.7306 - acc: 0.81 - ETA: 13s - loss: 0.7349 - acc: 0.80 - ETA: 12s - loss: 0.7337 - acc: 0.81 - ETA: 12s - loss: 0.7357 - acc: 0.80 - ETA: 12s - loss: 0.7352 - acc: 0.80 - ETA: 12s - loss: 0.7367 - acc: 0.80 - ETA: 12s - loss: 0.7374 - acc: 0.80 - ETA: 12s - loss: 0.7372 - acc: 0.80 - ETA: 11s - loss: 0.7377 - acc: 0.80 - ETA: 11s - loss: 0.7378 - acc: 0.80 - ETA: 11s - loss: 0.7391 - acc: 0.80 - ETA: 11s - loss: 0.7384 - acc: 0.80 - ETA: 11s - loss: 0.7376 - acc: 0.80 - ETA: 11s - loss: 0.7372 - acc: 0.80 - ETA: 10s - loss: 0.7380 - acc: 0.80 - ETA: 10s - loss: 0.7375 - acc: 0.80 - ETA: 10s - loss: 0.7375 - acc: 0.80 - ETA: 10s - loss: 0.7370 - acc: 0.80 - ETA: 10s - loss: 0.7396 - acc: 0.80 - ETA: 10s - loss: 0.7407 - acc: 0.80 - ETA: 9s - loss: 0.7394 - acc: 0.8072 - ETA: 9s - loss: 0.7409 - acc: 0.807 - ETA: 9s - loss: 0.7413 - acc: 0.806 - ETA: 9s - loss: 0.7400 - acc: 0.807 - ETA: 9s - loss: 0.7396 - acc: 0.807 - ETA: 9s - loss: 0.7415 - acc: 0.806 - ETA: 8s - loss: 0.7434 - acc: 0.805 - ETA: 8s - loss: 0.7430 - acc: 0.806 - ETA: 8s - loss: 0.7429 - acc: 0.806 - ETA: 8s - loss: 0.7426 - acc: 0.806 - ETA: 8s - loss: 0.7425 - acc: 0.806 - ETA: 7s - loss: 0.7412 - acc: 0.806 - ETA: 7s - loss: 0.7420 - acc: 0.806 - ETA: 7s - loss: 0.7419 - acc: 0.806 - ETA: 7s - loss: 0.7430 - acc: 0.806 - ETA: 7s - loss: 0.7434 - acc: 0.806 - ETA: 7s - loss: 0.7434 - acc: 0.806 - ETA: 6s - loss: 0.7430 - acc: 0.806 - ETA: 6s - loss: 0.7470 - acc: 0.806 - ETA: 6s - loss: 0.7499 - acc: 0.805 - ETA: 6s - loss: 0.7509 - acc: 0.804 - ETA: 6s - loss: 0.7516 - acc: 0.804 - ETA: 6s - loss: 0.7532 - acc: 0.803 - ETA: 5s - loss: 0.7537 - acc: 0.803 - ETA: 5s - loss: 0.7546 - acc: 0.803 - ETA: 5s - loss: 0.7558 - acc: 0.803 - ETA: 5s - loss: 0.7553 - acc: 0.803 - ETA: 5s - loss: 0.7565 - acc: 0.803 - ETA: 5s - loss: 0.7592 - acc: 0.802 - ETA: 4s - loss: 0.7589 - acc: 0.802 - ETA: 4s - loss: 0.7591 - acc: 0.802 - ETA: 4s - loss: 0.7575 - acc: 0.802 - ETA: 4s - loss: 0.7588 - acc: 0.802 - ETA: 4s - loss: 0.7585 - acc: 0.802 - ETA: 3s - loss: 0.7591 - acc: 0.802 - ETA: 3s - loss: 0.7603 - acc: 0.802 - ETA: 3s - loss: 0.7603 - acc: 0.802 - ETA: 3s - loss: 0.7604 - acc: 0.802 - ETA: 3s - loss: 0.7609 - acc: 0.801 - ETA: 3s - loss: 0.7610 - acc: 0.801 - ETA: 2s - loss: 0.7603 - acc: 0.802 - ETA: 2s - loss: 0.7614 - acc: 0.802 - ETA: 2s - loss: 0.7615 - acc: 0.801 - ETA: 2s - loss: 0.7607 - acc: 0.802 - ETA: 2s - loss: 0.7595 - acc: 0.802 - ETA: 2s - loss: 0.7599 - acc: 0.801 - ETA: 1s - loss: 0.7594 - acc: 0.801 - ETA: 1s - loss: 0.7604 - acc: 0.801 - ETA: 1s - loss: 0.7605 - acc: 0.801 - ETA: 1s - loss: 0.7626 - acc: 0.801 - ETA: 1s - loss: 0.7635 - acc: 0.801 - ETA: 1s - loss: 0.7637 - acc: 0.801 - ETA: 0s - loss: 0.7624 - acc: 0.801 - ETA: 0s - loss: 0.7621 - acc: 0.801 - ETA: 0s - loss: 0.7645 - acc: 0.801 - ETA: 0s - loss: 0.7628 - acc: 0.801 - ETA: 0s - loss: 0.7636 - acc: 0.8014Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 60s 9ms/step - loss: 0.7633 - acc: 0.8013 - val_loss: 5.7316 - val_acc: 0.0838
    




    <keras.callbacks.History at 0x22d5ae1e9b0>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 10.5263%
    

---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=2)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    Epoch 00001: val_loss improved from inf to 11.19254, saving model to saved_models/weights.best.VGG16.hdf5
     - 6s - loss: 12.4060 - acc: 0.1219 - val_loss: 11.1925 - val_acc: 0.1988
    Epoch 2/20
    Epoch 00002: val_loss improved from 11.19254 to 10.48439, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 10.5046 - acc: 0.2656 - val_loss: 10.4844 - val_acc: 0.2766
    Epoch 3/20
    Epoch 00003: val_loss improved from 10.48439 to 10.20181, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.9738 - acc: 0.3278 - val_loss: 10.2018 - val_acc: 0.2874
    Epoch 4/20
    Epoch 00004: val_loss improved from 10.20181 to 10.02010, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.7736 - acc: 0.3536 - val_loss: 10.0201 - val_acc: 0.3090
    Epoch 5/20
    Epoch 00005: val_loss improved from 10.02010 to 9.88408, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.5226 - acc: 0.3778 - val_loss: 9.8841 - val_acc: 0.3281
    Epoch 6/20
    Epoch 00006: val_loss improved from 9.88408 to 9.78742, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.3814 - acc: 0.3910 - val_loss: 9.7874 - val_acc: 0.3293
    Epoch 7/20
    Epoch 00007: val_loss improved from 9.78742 to 9.70188, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.3099 - acc: 0.4052 - val_loss: 9.7019 - val_acc: 0.3437
    Epoch 8/20
    Epoch 00008: val_loss improved from 9.70188 to 9.55102, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.2428 - acc: 0.4106 - val_loss: 9.5510 - val_acc: 0.3497
    Epoch 9/20
    Epoch 00009: val_loss improved from 9.55102 to 9.39986, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 9.0334 - acc: 0.4235 - val_loss: 9.3999 - val_acc: 0.3605
    Epoch 10/20
    Epoch 00010: val_loss improved from 9.39986 to 9.36582, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.9251 - acc: 0.4362 - val_loss: 9.3658 - val_acc: 0.3545
    Epoch 11/20
    Epoch 00011: val_loss improved from 9.36582 to 9.32486, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.8388 - acc: 0.4413 - val_loss: 9.3249 - val_acc: 0.3605
    Epoch 12/20
    Epoch 00012: val_loss improved from 9.32486 to 9.24512, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.7848 - acc: 0.4464 - val_loss: 9.2451 - val_acc: 0.3749
    Epoch 13/20
    Epoch 00013: val_loss did not improve
     - 3s - loss: 8.6950 - acc: 0.4507 - val_loss: 9.2675 - val_acc: 0.3701
    Epoch 14/20
    Epoch 00014: val_loss improved from 9.24512 to 9.17720, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.6762 - acc: 0.4554 - val_loss: 9.1772 - val_acc: 0.3749
    Epoch 15/20
    Epoch 00015: val_loss improved from 9.17720 to 9.12936, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.6072 - acc: 0.4566 - val_loss: 9.1294 - val_acc: 0.3677
    Epoch 16/20
    Epoch 00016: val_loss improved from 9.12936 to 8.85784, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.3333 - acc: 0.4669 - val_loss: 8.8578 - val_acc: 0.3820
    Epoch 17/20
    Epoch 00017: val_loss improved from 8.85784 to 8.58872, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 8.1667 - acc: 0.4768 - val_loss: 8.5887 - val_acc: 0.3952
    Epoch 18/20
    Epoch 00018: val_loss improved from 8.58872 to 8.32880, saving model to saved_models/weights.best.VGG16.hdf5
     - 2s - loss: 7.7866 - acc: 0.4981 - val_loss: 8.3288 - val_acc: 0.4156
    Epoch 19/20
    Epoch 00019: val_loss did not improve
     - 3s - loss: 7.6708 - acc: 0.5100 - val_loss: 8.3532 - val_acc: 0.4132
    Epoch 20/20
    Epoch 00020: val_loss improved from 8.32880 to 8.28671, saving model to saved_models/weights.best.VGG16.hdf5
     - 3s - loss: 7.5475 - acc: 0.5168 - val_loss: 8.2867 - val_acc: 0.4024
    




    <keras.callbacks.History at 0x22d50453a58>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 42.7033%
    

### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 




```python
### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dropout(0.5))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_4 ( (None, 2048)              0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________
    

### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=60, batch_size=20, callbacks=[checkpointer], verbose=2)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/60
    Epoch 00001: val_loss improved from inf to 0.90694, saving model to saved_models/weights.best.Resnet50.hdf5
     - 5s - loss: 2.3642 - acc: 0.4386 - val_loss: 0.9069 - val_acc: 0.7281
    Epoch 2/60
    Epoch 00002: val_loss improved from 0.90694 to 0.67576, saving model to saved_models/weights.best.Resnet50.hdf5
     - 3s - loss: 0.8094 - acc: 0.7496 - val_loss: 0.6758 - val_acc: 0.8024
    Epoch 3/60
    Epoch 00003: val_loss improved from 0.67576 to 0.66672, saving model to saved_models/weights.best.Resnet50.hdf5
     - 3s - loss: 0.6101 - acc: 0.8141 - val_loss: 0.6667 - val_acc: 0.7880
    Epoch 4/60
    Epoch 00004: val_loss improved from 0.66672 to 0.65406, saving model to saved_models/weights.best.Resnet50.hdf5
     - 3s - loss: 0.5104 - acc: 0.8448 - val_loss: 0.6541 - val_acc: 0.8072
    Epoch 5/60
    Epoch 00005: val_loss improved from 0.65406 to 0.64572, saving model to saved_models/weights.best.Resnet50.hdf5
     - 3s - loss: 0.4495 - acc: 0.8657 - val_loss: 0.6457 - val_acc: 0.8024
    Epoch 6/60
    Epoch 00006: val_loss did not improve
     - 3s - loss: 0.3853 - acc: 0.8816 - val_loss: 0.6830 - val_acc: 0.8120
    Epoch 7/60
    Epoch 00007: val_loss improved from 0.64572 to 0.63086, saving model to saved_models/weights.best.Resnet50.hdf5
     - 3s - loss: 0.3577 - acc: 0.8880 - val_loss: 0.6309 - val_acc: 0.8144
    Epoch 8/60
    Epoch 00008: val_loss did not improve
     - 3s - loss: 0.3406 - acc: 0.9003 - val_loss: 0.6684 - val_acc: 0.8204
    Epoch 9/60
    Epoch 00009: val_loss did not improve
     - 3s - loss: 0.3124 - acc: 0.9054 - val_loss: 0.6645 - val_acc: 0.8240
    Epoch 10/60
    Epoch 00010: val_loss did not improve
     - 3s - loss: 0.2928 - acc: 0.9105 - val_loss: 0.6719 - val_acc: 0.8180
    Epoch 11/60
    Epoch 00011: val_loss did not improve
     - 3s - loss: 0.2684 - acc: 0.9216 - val_loss: 0.6705 - val_acc: 0.8180
    Epoch 12/60
    Epoch 00012: val_loss did not improve
     - 3s - loss: 0.2665 - acc: 0.9207 - val_loss: 0.6651 - val_acc: 0.8287
    Epoch 13/60
    Epoch 00013: val_loss did not improve
     - 3s - loss: 0.2616 - acc: 0.9271 - val_loss: 0.7055 - val_acc: 0.8108
    Epoch 14/60
    Epoch 00014: val_loss did not improve
     - 3s - loss: 0.2343 - acc: 0.9281 - val_loss: 0.7638 - val_acc: 0.8144
    Epoch 15/60
    Epoch 00015: val_loss did not improve
     - 3s - loss: 0.2303 - acc: 0.9277 - val_loss: 0.7271 - val_acc: 0.8216
    Epoch 16/60
    Epoch 00016: val_loss did not improve
     - 3s - loss: 0.2342 - acc: 0.9308 - val_loss: 0.7691 - val_acc: 0.8216
    Epoch 17/60
    Epoch 00017: val_loss did not improve
     - 3s - loss: 0.2191 - acc: 0.9359 - val_loss: 0.7244 - val_acc: 0.8251
    Epoch 18/60
    Epoch 00018: val_loss did not improve
     - 3s - loss: 0.2101 - acc: 0.9403 - val_loss: 0.7306 - val_acc: 0.8156
    Epoch 19/60
    Epoch 00019: val_loss did not improve
     - 3s - loss: 0.1993 - acc: 0.9463 - val_loss: 0.8055 - val_acc: 0.8216
    Epoch 20/60
    Epoch 00020: val_loss did not improve
     - 3s - loss: 0.1970 - acc: 0.9443 - val_loss: 0.7578 - val_acc: 0.8263
    Epoch 21/60
    Epoch 00021: val_loss did not improve
     - 3s - loss: 0.1975 - acc: 0.9422 - val_loss: 0.7633 - val_acc: 0.8228
    Epoch 22/60
    Epoch 00022: val_loss did not improve
     - 3s - loss: 0.1937 - acc: 0.9445 - val_loss: 0.7771 - val_acc: 0.8299
    Epoch 23/60
    Epoch 00023: val_loss did not improve
     - 3s - loss: 0.1830 - acc: 0.9464 - val_loss: 0.7879 - val_acc: 0.8192
    Epoch 24/60
    Epoch 00024: val_loss did not improve
     - 3s - loss: 0.1751 - acc: 0.9494 - val_loss: 0.7948 - val_acc: 0.8275
    Epoch 25/60
    Epoch 00025: val_loss did not improve
     - 3s - loss: 0.1660 - acc: 0.9515 - val_loss: 0.8235 - val_acc: 0.8156
    Epoch 26/60
    Epoch 00026: val_loss did not improve
     - 3s - loss: 0.1677 - acc: 0.9524 - val_loss: 0.8728 - val_acc: 0.8323
    Epoch 27/60
    Epoch 00027: val_loss did not improve
     - 3s - loss: 0.1720 - acc: 0.9543 - val_loss: 0.8067 - val_acc: 0.8263
    Epoch 28/60
    Epoch 00028: val_loss did not improve
     - 3s - loss: 0.1592 - acc: 0.9563 - val_loss: 0.7872 - val_acc: 0.8323
    Epoch 29/60
    Epoch 00029: val_loss did not improve
     - 3s - loss: 0.1687 - acc: 0.9551 - val_loss: 0.8156 - val_acc: 0.8192
    Epoch 30/60
    Epoch 00030: val_loss did not improve
     - 3s - loss: 0.1617 - acc: 0.9534 - val_loss: 0.8279 - val_acc: 0.8216
    Epoch 31/60
    Epoch 00031: val_loss did not improve
     - 3s - loss: 0.1545 - acc: 0.9575 - val_loss: 0.8349 - val_acc: 0.8192
    Epoch 32/60
    Epoch 00032: val_loss did not improve
     - 3s - loss: 0.1648 - acc: 0.9531 - val_loss: 0.8055 - val_acc: 0.8287
    Epoch 33/60
    Epoch 00033: val_loss did not improve
     - 3s - loss: 0.1549 - acc: 0.9578 - val_loss: 0.8563 - val_acc: 0.8263
    Epoch 34/60
    Epoch 00034: val_loss did not improve
     - 3s - loss: 0.1520 - acc: 0.9581 - val_loss: 0.8612 - val_acc: 0.8311
    Epoch 35/60
    Epoch 00035: val_loss did not improve
     - 3s - loss: 0.1446 - acc: 0.9612 - val_loss: 0.8810 - val_acc: 0.8311
    Epoch 36/60
    Epoch 00036: val_loss did not improve
     - 3s - loss: 0.1396 - acc: 0.9624 - val_loss: 0.8442 - val_acc: 0.8287
    Epoch 37/60
    Epoch 00037: val_loss did not improve
     - 3s - loss: 0.1356 - acc: 0.9609 - val_loss: 0.8662 - val_acc: 0.8275
    Epoch 38/60
    Epoch 00038: val_loss did not improve
     - 2s - loss: 0.1495 - acc: 0.9558 - val_loss: 0.8631 - val_acc: 0.8407
    Epoch 39/60
    Epoch 00039: val_loss did not improve
     - 2s - loss: 0.1294 - acc: 0.9627 - val_loss: 0.8480 - val_acc: 0.8359
    Epoch 40/60
    Epoch 00040: val_loss did not improve
     - 3s - loss: 0.1520 - acc: 0.9593 - val_loss: 0.8724 - val_acc: 0.8335
    Epoch 41/60
    Epoch 00041: val_loss did not improve
     - 3s - loss: 0.1271 - acc: 0.9659 - val_loss: 0.8824 - val_acc: 0.8299
    Epoch 42/60
    Epoch 00042: val_loss did not improve
     - 3s - loss: 0.1336 - acc: 0.9618 - val_loss: 0.8799 - val_acc: 0.8287
    Epoch 43/60
    Epoch 00043: val_loss did not improve
     - 3s - loss: 0.1384 - acc: 0.9623 - val_loss: 0.8577 - val_acc: 0.8323
    Epoch 44/60
    Epoch 00044: val_loss did not improve
     - 3s - loss: 0.1272 - acc: 0.9659 - val_loss: 0.8881 - val_acc: 0.8240
    Epoch 45/60
    Epoch 00045: val_loss did not improve
     - 3s - loss: 0.1201 - acc: 0.9677 - val_loss: 0.8994 - val_acc: 0.8228
    Epoch 46/60
    Epoch 00046: val_loss did not improve
     - 3s - loss: 0.1145 - acc: 0.9692 - val_loss: 0.8965 - val_acc: 0.8383
    Epoch 47/60
    Epoch 00047: val_loss did not improve
     - 3s - loss: 0.1212 - acc: 0.9665 - val_loss: 0.8931 - val_acc: 0.8287
    Epoch 48/60
    Epoch 00048: val_loss did not improve
     - 3s - loss: 0.1241 - acc: 0.9662 - val_loss: 0.8886 - val_acc: 0.8335
    Epoch 49/60
    Epoch 00049: val_loss did not improve
     - 3s - loss: 0.1193 - acc: 0.9660 - val_loss: 0.9032 - val_acc: 0.8251
    Epoch 50/60
    Epoch 00050: val_loss did not improve
     - 3s - loss: 0.1194 - acc: 0.9666 - val_loss: 0.9237 - val_acc: 0.8347
    Epoch 51/60
    Epoch 00051: val_loss did not improve
     - 3s - loss: 0.1062 - acc: 0.9684 - val_loss: 0.8976 - val_acc: 0.8347
    Epoch 52/60
    Epoch 00052: val_loss did not improve
     - 3s - loss: 0.1136 - acc: 0.9689 - val_loss: 0.8952 - val_acc: 0.8299
    Epoch 53/60
    Epoch 00053: val_loss did not improve
     - 3s - loss: 0.1061 - acc: 0.9699 - val_loss: 0.9276 - val_acc: 0.8275
    Epoch 54/60
    Epoch 00054: val_loss did not improve
     - 3s - loss: 0.1259 - acc: 0.9654 - val_loss: 0.9177 - val_acc: 0.8359
    Epoch 55/60
    Epoch 00055: val_loss did not improve
     - 2s - loss: 0.1194 - acc: 0.9693 - val_loss: 0.9342 - val_acc: 0.8299
    Epoch 56/60
    Epoch 00056: val_loss did not improve
     - 3s - loss: 0.1243 - acc: 0.9692 - val_loss: 0.9780 - val_acc: 0.8251
    Epoch 57/60
    Epoch 00057: val_loss did not improve
     - 3s - loss: 0.1069 - acc: 0.9719 - val_loss: 0.9373 - val_acc: 0.8228
    Epoch 58/60
    Epoch 00058: val_loss did not improve
     - 3s - loss: 0.1060 - acc: 0.9708 - val_loss: 1.0020 - val_acc: 0.8251
    Epoch 59/60
    Epoch 00059: val_loss did not improve
     - 3s - loss: 0.1176 - acc: 0.9698 - val_loss: 0.9439 - val_acc: 0.8311
    Epoch 60/60
    Epoch 00060: val_loss did not improve
     - 3s - loss: 0.0991 - acc: 0.9699 - val_loss: 0.9836 - val_acc: 0.8299
    




    <keras.callbacks.History at 0x2303f020da0>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 80.7416%
    

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def image_detector(im_path):
    dog_breed = Resnet50_predict_breed(im_path)
    
    image = cv2.imread(im_path)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
    
    if dog_detector(im_path):
        print("Hello dog, I think you are a " + str(dog_breed))
    elif face_detector(im_path):
        print("Hello human, you look like a " + str(dog_breed))
    else:
        print("I can't recognize you !!!")
    
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
image_detector('test_image\\dog1.jpg')
```


![png](output_64_0.png)


    Hello dog, I think you are a Alaskan_malamute
    


```python
image_detector('test_image\\dog2.jpg')
```


![png](output_65_0.png)


    Hello dog, I think you are a Labrador_retriever
    


```python
image_detector('test_image\\human1.jpg')
```


![png](output_66_0.png)


    Hello human, you look like a Portuguese_water_dog
    


```python
image_detector('test_image\\human2.jpg')
```


![png](output_67_0.png)


    Hello human, you look like a Pomeranian
    


```python
image_detector('test_image\\other1.jpg')
```


![png](output_68_0.png)


    I can't recognize you !!!
    


```python
image_detector('test_image\\other2.jpg')
```


![png](output_69_0.png)


    I can't recognize you !!!
    

I think the output is amazing as least for those pictures. But for the improvement i have some propositions:
* We can choose a more complicated architecture such as Xception with a power machine, maybe we can reach 90% of accuracy.
* Instead of take weights directly, we can augment our data by a transformation to avoid the variant feature or we can use the spatial transformer to get the contour of our image.
* We can try the ELU activation function, who is converge more quikly then RELU, like we saw in the MLP, each layer send a bias to the next layer and the bias get important when we have Not 0 mean of all the activation unite, so maybe the ELU is more suitable the RELU because of negtive values that it product.
