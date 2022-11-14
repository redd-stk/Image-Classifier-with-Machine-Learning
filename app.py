import os
import pickle

import streamlit as st
import matplotlib.pyplot as plt  # for displaying the images
import numpy as np  # for numerical operations
from PIL import Image
from sklearn import svm
from skimage.io import imread      #for reading the images downloaded/ image processing
from skimage.transform import resize  #for resizing all the downloaded images into one size
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

target = []
images = []
flat_data = [] #for storing the flattened data

# folder where images are stored
data_directory = 'images'
IMG_categories = ['Bicycle', 'Cruise ship', 'Motorcycle']

for category in IMG_categories:
    class_num = IMG_categories.index(category)  # label encoding of the values
    # combines path of category iterated with image folder directory
    path = os.path.join(data_directory, category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        # changing the size of the images. Normalizes value from 0 to 1
        resized_image = resize(img_array, (150, 150, 3))
        # flattenning the images and adding to the flat_data array
        flat_data.append(resized_image.flatten())
        # saving the resized images into a new array 'imagees'
        images.append(resized_image)
        target.append(class_num)


# making sure they are converted into arrays
flat_data = np.array(flat_data)
images = np.array(images)
target = np.array(target)


#adding streamlit ui
st.title('Image Classifier with Machine learning')

classifier_name = st.sidebar.selectbox('Select Classifier', ('SVM', 'KNN', 'Random Forest'))

def add_param_ui(clsf_name):
    params = dict()
    if clsf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clsf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        no_estimators = st.sidebar.slider('no_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['no_estimators'] = no_estimators
        
    return params

params = add_param_ui(classifier_name)

def get_classifier(clsf_name, params):
    if clsf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clsf_name == 'SVM':
        clf = SVC(C=params['C'], probability=True)
    else:
        clf = RandomForestClassifier(no_estimators=params['no_estimators'], max_depth=params['max_depth'], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)


# splitting the data into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.2, random_state=1234)


clf.fit(x_train, y_train)
#checking the accuracy of the prediction
y_pred = clf.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


#saving the model using pickle library
pickle.dump(clf, open('imgclsf_model.p', 'wb'))  #saving model into a pickle file

model = pickle.load(open('imgclsf_model.p', 'rb'))


st.text('Choose an image to upload')

#uploading the image
uploaded_file = st.file_uploader('Choose image', type = 'jpg')
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption= 'Uploaded Image')

    #predicting the image type from trained model above
    if st.button('PREDICT'):
        st.write('Result')
        flat_data = []
        img = np.array(img)
        resized_image = resize(img, (150, 150, 3))
        flat_data.append(resized_image.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = IMG_categories[y_out[0]]
        accuracy_percentage = accuracy*100
        st.write(f"Model Prediction Accuracy = {accuracy_percentage}%")
        st.title(f'PREDICTED OUTPUT: {y_out}')  #displaying the predicted out
        
         
        
        q = model.predict_proba(flat_data) #getting the probability score
        for index, item in enumerate(IMG_categories):
            st.write(f'{item} : {q[0][index] * 100}%')
