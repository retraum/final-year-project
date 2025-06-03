# training the RandomForest on TA image

# --------------------------------------------------
# video 63 - image segmentation in python using traditional machine learning
# STEP 1 - feature extraction

import numpy as np
import cv2
import pandas as pd
import time

start_time = time.time()

# reading the file
file = 'sample.png'
directory = 'D:/kuliah/TA/ML magnesium coating/images/practice materials/fix/'
path = directory + file

img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grayscale

# creating empty dataframme
df = pd.DataFrame()

# unwrap the img so it can be 1D
# it has to be 1D so it can be added to the dataframe
img2 = img.reshape(-1)

# adding features into dataframe
# feature 1: original pixel values
df['Original Image'] = img2

# feature 2: Gabor features
# kernel properties
phi = 0
num = 1
for theta in range(4):
    theta = theta * np.pi/4
    for sigma in (3,5):
        for lamda in np.arange(0,np.pi,np.pi/4):
            for gamma in (.05,.5):
# define gabor label for the dataframe
                gabor_label = 'Gabor' + str(num)
                kernel = cv2.getGaborKernel((5,5), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
# filtered image (2D)
                fimg = cv2.filter2D(img,cv2.CV_8UC3,kernel)
# reshape the filtered image to 1D
                fimg = fimg.reshape(-1)
                df[gabor_label] = fimg
                print(f'{gabor_label} - theta: {theta:.2f}, sigma: {sigma}, lambda: {lamda:.2f}, gamma: {gamma:.2f}, phi: {phi}')
                num += 1

# ##########
# feature 3: canny edge
edges = cv2.Canny(img,100,200)
edges = edges.reshape(-1)
df['Canny edge'] = edges

from skimage.filters import roberts, sobel, scharr, prewitt
# feature 4: roberts
edge_roberts = roberts(img)
edge_roberts = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts

# feature 5: sobel
edge_sobel = sobel(img)
edge_sobel = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel

# feature 6: scharr
edge_scharr = scharr(img)
edge_scharr = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr

# feature 7: prewitt
edge_prewitt = prewitt(img)
edge_prewitt = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt

from scipy import ndimage as nd

# feature 8: gaussian
for s in range(1,11):
    name = 'Gaussian '+str(s)
    gaussian = nd.gaussian_filter(img,sigma=s)
    gaussian = gaussian.reshape(-1)
    df[name] = gaussian

# feature 9: median
for m in range(1,11):
    name = 'Median '+str(m)
    median = nd.median_filter(img,size=m)
    median = median.reshape(-1)
    df[name] = median

# feature 10: variance
for v in range(1,11):
    name = 'Variance '+str(v)
    variance = nd.generic_filter(img,np.var,size=v)
    variance = variance.reshape(-1)
    df[name] = variance

# feature 11: labeled (masked) image
masked_image = 'tracing_ver2.png'
path = directory + masked_image
labeled_img = cv2.imread(path)
labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_BGR2GRAY) # convert to grayscale
labeled_img = labeled_img.reshape(-1)
df['Labeled'] = labeled_img

# allocating fragmented dataframe
df = df.copy()
if len(df._data.blocks) > 2:
    print('dataframe is fragmented')
else:
    print('dataframe is not fragmented')

print(df.head())

# storing to csv
csv_path = directory+'filters.csv' 
df.to_csv(csv_path)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Time to store CSV: {elapsed_time//60} minutes {elapsed_time%60:.1f} seconds')


# --------------------------------------------------
# video 64 - image segmentation in python using traditional machine learning
# STEP 2 - training random forest (classifier)
import pandas as pd

# reading the file
df = pd.read_csv(csv_path)

# dropping the 'Unnamed: 0' column
df = df.drop(columns=df.columns[0])
print(df.head())

# printing list of columns
col_list = df.columns.tolist()
print('\ncolumns:')
for i,col in enumerate(col_list):
    print(i,col)
   
# independent variables: columns 0-99
x = df.drop(columns=df.columns[100])
print('\nx:\n',x)

# dependent variable: column 100 ('Labeled')
y = df.iloc[:,100].values
print('y: ',y)

# splitting data into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=20)

# import and train the random forest model
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
model = RandomForestClassifier(n_estimators=10, random_state=25)
model.fit(x_train,y_train)

# predict the accuracy on the training data
y_predict = model.predict(x_test)

# calculating the model accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_predict)*100

# --------------------------------------------------
# video 65 - image segmentation in python using traditional machine learning
# STEP 3 - feature ranking (Random Forest Classifier)
importances = list(model.feature_importances_)
features_list = list(x.columns)
features_imp = pd.Series(model.feature_importances_,index=features_list).sort_values(ascending=False)

# features now only consist of features with non-zero importances
features_imp = features_imp[features_imp > 0]

#for feature,importance in zip(features_imp.index,features_imp.values):
#    print(f'{feature}: {importance}')

# features now are only the top 10 of highest importances
features_imp = features_imp.nlargest(10)
print('\nimportant features:')
print(features_imp)

# --------------------------------------------------
# video 66 - image segmentation in python using traditional machine learning
# STEP 4 - pickling model
# storing model we just trained for future uses
import pickle
filename = directory + file[:-4] +'_model'

# wb = write binary
pickle.dump(model,open(filename,'wb'))

# load the model
# rb = read binary
load_model = pickle.load(open(filename,'rb'))

# predicting the new image
# for now, the testing image is the whole x data
result = load_model.predict(x)

# the result only consists of one column
# we need to make it the same size as our image
segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt
plt.imshow(segmented,cmap='binary')
plt.imsave(directory+'segmented_sample.png',segmented,cmap='binary')
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time

print(f'accuracy (RandomForest) = {accuracy:.2f}%')
print(f'Time to train Random Forest: {elapsed_time//60} minutes {elapsed_time%60:.1f} seconds')


####

'''
# --------------------------------------------------
# video 67 - image segmentation in python using traditional machine learning
# STEP 5 - segmenting images using a trained model
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

def feature_extraction(img):
# creating empty dataframe
    df = pd.DataFrame()

# unwrap the img so it can be 1D
# it has to be 1D so it can be added to the dataframe
    img2 = img.reshape(-1)

# adding features into dataframe
# feature 1: original pixel values
    df['Original Image'] = img2

# feature 2: Gabor features
# kernel properties
    phi = 0
    num = 1
    for theta in range(4):
        theta = theta * np.pi/4
        for sigma in (3,5):
            for lamda in np.arange(0,np.pi,np.pi/4):
                for gamma in (.05,.5):
# define gabor label for the dataframe
                    gabor_label = 'Gabor' + str(num)
                    kernel = cv2.getGaborKernel((5,5), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
# filtered image (2D)
                    fimg = cv2.filter2D(img,cv2.CV_8UC3,kernel)
# reshape the filtered image to 1D
                    fimg = fimg.reshape(-1)
                    df[gabor_label] = fimg
                    #print(f'{gabor_label} - theta: {theta:.2f}, sigma: {sigma}, lambda: {lamda:.2f}, gamma: {gamma:.2f}, phi: {phi}')
                    num += 1

# ##########
# feature 3: canny edge
    edges = cv2.Canny(img,100,200)
    edges = edges.reshape(-1)
    df['Canny edge'] = edges

# feature 4: roberts
    edge_roberts = roberts(img)
    edge_roberts = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts

# feature 5: sobel
    edge_sobel = sobel(img)
    edge_sobel = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel

# feature 6: scharr
    edge_scharr = scharr(img)
    edge_scharr = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr

# feature 7: prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt

# feature 8: gaussian
    for s in range(1,11):
        name = 'Gaussian '+str(s)
        gaussian = nd.gaussian_filter(img,sigma=s)
        gaussian = gaussian.reshape(-1)
        df[name] = gaussian

# feature 9: median
    for m in range(1,11):
        name = 'Median '+str(m)
        median = nd.median_filter(img,size=m)
        median = median.reshape(-1)
        df[name] = median

# feature 10: variance
    for v in range(1,11):
        name = 'Variance '+str(v)
        variance = nd.generic_filter(img,np.var,size=v)
        variance = variance.reshape(-1)
        df[name] = variance

# allocating fragmented dataframe
    if len(df._data.blocks) > 2:
        print('dataframe is fragmented')
        df = df.copy()
    else:
        print('dataframe is not fragmented')
    
# return dataframe
    return df

# ---------------
# importing important libraries
import glob
import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# reading the RandomForest model already saved previously
model_name = 'D:/kuliah/TA/ML magnesium coating/images/practice materials/fix/sample_model'

# loading the model
load_model = pickle.load(open(model_name,'rb'))

# reading the path, the folder only contains .tif files
path = 'D:/kuliah/TA/ML magnesium coating/images/practice materials/training/*'
save_path = 'D:/kuliah/TA/ML magnesium coating/images/practice materials/segmented/'

import time
start_time = time.time()

for file in glob.glob(path):
    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
# feature extraction → independent variables as a dataframe
    features = feature_extraction(img)

# dependent variables (predicted image)
    result = load_model.predict(features)
    
# segment the result
    segmented = result.reshape((img.shape))
    
# saving the image
    name = file[len(path)-1:-4] + '_segmented.png'
    plt.imsave(save_path + name,segmented,cmap='binary')
    print(f'{name} is saved')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total time elapsed: {elapsed_time//60} minutes {elapsed_time%60:.1f} seconds')
'''

# --------------------------------------------------
# video 68b - support vector machine (SVM) vs RandomForest for image segmentation
'''
from sklearn.svm import LinearSVC

# train the model
start_time = time.time()
iterate = 1000
model = LinearSVC(max_iter=iterate) # max iteration is 100 by default
model.fit(x_train,y_train)

# prediction based on model
y_predict = model.predict(x_test)

# accuracy
accuracy = metrics.accuracy_score(y_test,y_predict) * 100

# saving the file
filename = filename[:-5] + 'SVM_model'

# wb = write binary
pickle.dump(model,open(filename,'wb'))

# load the model
# rb = read binary
load_model = pickle.load(open(filename,'rb'))

# predicting the new image
# for now, the testing image is the whole x data
result = load_model.predict(x)

# the result only consists of one column
# we need to make it the same size as our image
segmented = result.reshape((img.shape))

plt.imshow(segmented,cmap='binary')
plt.imsave(filename[:-5]+'segmented.png',segmented,cmap='binary')
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time

print(f'accuracy (SVM) SVM ({iterate} iterations): {accuracy:.2f}%')
print(f'Time to train SVM ({iterate} iterations): {elapsed_time//60} minutes {elapsed_time%60:.1f} seconds')
'''