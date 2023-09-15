import itertools
import os
import time
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  # shuffling the data improves the model
from matplotlib import style
style.use("ggplot")
from sklearn import metrics
from sklearn import svm, metrics, datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
print(tf.__version__)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
ZeroPadding2D = tf.keras.layers.ZeroPadding2D
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
load_model = tf.keras.models.load_model
TensorBoard = tf.keras.callbacks.TensorBoard
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def decupare_contur(image, plot=False):
    # imaginea pe nivele de gri

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # bluram putin imaginea ( ajuta la eliminarea zgomotului Gaussian )

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Facem operatie de threshold pt a face imaginea din gray scale in binary
    # Operatiile de eroziune si dilatare sunt folosite pt eliminarea regiunilor mici de zgomot .

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Functie pt preluat conturul exterior al imaginii ( CHAIN_APPROX_NONE ia tot conturul ,
    # CHAIN_APPROX_SIMPLE ia doar 4 puncte exterioare)
    # se lucreaza pe imagini binare , deci vom crea o copie a thresh .
    # RETR_EXTERNAL = ce tip de contur returnam
    # CHAIN_APPROX = metoda prin care il returnam .
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # imutils este utilizat pt detectarea de margini
    # grab_countours
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    # conturul este un array NumPy de coordonate (x,y)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])  # argmin() preia cea mai mica valoare de pe axa X
    extRight = tuple(c[c[:, :, 0].argmax()][0])  # argmax() preia cea mai mare valoare de pe axa X
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Imaginea initiala')

        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Imaginea Segmentata')
        plt.show()

    return new_image


ex_img = cv2.imread('yes/Y3.jpg')
ex_new_img = decupare_contur(ex_img, True)


def load_data(dir_list, image_size):
    """
            Citim imaginea , facem resize si o normalizam
        Intrare:
        dir_list: lista cu directoarele
        image_size = dimensiunea imaginii

    Returnam :
        X:  array de forma = (nr_img, image_width, image_height, nr_canale)
        y: array de forma = (nr_img, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size

    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '\\' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = decupare_contur(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y


augmented_path = 'augmented_data/'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + 'yes'
augmented_no = augmented_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))


def plot_sample_images(X, y, n=50):
    """
    Plots n sample images for both values of y (labels).

"""
    for label in [0, 1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n / columns_n)

        plt.figure(figsize=(20, 10))

        i = 1  # current plot
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # stergem axele si valorile de pe axe
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()


plot_sample_images(X, y)


def split_data(X, y, test_size=0.3):
    # Functia train_test_split imparte imaginile in seturi de training si test

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    # prima operatie o facem pentru a atribui 30% din imagini pt test si validation
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    # a doua operatie o facem pentru a imparti datele de test si validation in parti egale

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

# in momentul actual avem 70% training , 15% test , 15%validation
y = dict()
y[0] = []
y[1] = []

for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

labels = ['Date antrenare', 'Date validare', 'Date testare']
colors = ['#FFFF00', '#FA8072']

fig, ax = plt.subplots()
ax.bar(labels, y[0], label='Img.non-tumoroase', color=colors[0], alpha=0.7)
ax.bar(labels, y[1], bottom=y[0], label='Img. tumoroase', color=colors[1], alpha=0.7)
ax.set_title('Numarul imaginilor din fiecare set')
ax.set_xlabel('Set de date')
ax.set_ylabel('Numar imagini')
ax.legend()

plt.show()

##################################


print("Nr exemple training = " + str(X_train.shape[0]))
print("Nr exemple validation = " + str(X_val.shape[0]))
print("Nr exemple test = " + str(X_test.shape[0]))
print("X_train : " + str(X_train.shape))
print("Y_train : " + str(y_train.shape))
print("X_val  : " + str(X_val.shape))
print("Y_val  : " + str(y_val.shape))
print("X_test : " + str(X_test.shape))
print("Y_test : " + str(y_test.shape))
print("Y_test : " + str(y_test.shape))
print("Y_test : " + str(y_test.shape))


# Nicely formatted time string

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"


def build_model(input_shape):

    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model.
        shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    X_input = Input(input_shape)  # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X1 = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)

    # CONV -> BN -> RELU Block applied to X
    X1 = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X1)
    X1 = BatchNormalization(axis=3, name='bn0')(X1)
    X1 = Activation('relu')(X1)  # shape=(?, 238, 238, 32)

    # MAXPOOL
    X1 = MaxPooling2D((4, 4), name='max_pool0')(X1)  # shape=(?, 59, 59, 32)

    # MAXPOOL
    X1 = MaxPooling2D((4, 4), name='max_pool1')(X1)  # shape=(?, 14, 14, 32)

    # FLATTEN X
    X1 = Flatten()(X1)  # shape=(?, 6272)
    # FULLYCONNECTED
    X1 = Dense(2, activation='sigmoid', name='fc')(X1)  # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    # model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

    # model 2   - seamana cu  vgg16 23layers
    # X_input = Input(input_shape)  # shape=(?, 240, 240, 3)
    #
    # x1 = Conv2D(64, (22, 22), strides=2)(X_input)
    # x1 = MaxPooling2D((4, 4))(x1)
    # x1 = BatchNormalization()(x1)
    #
    # x2 = Conv2D(128, (11, 11), strides=2, padding="same")(x1)
    # x2 = MaxPooling2D((2, 2))(x2)
    # x2 = BatchNormalization()(x2)
    #
    # x3 = Conv2D(256, (7, 7), strides=2, padding="same")(x2)
    # x3 = MaxPooling2D((2, 2))(x3)
    # x3 = BatchNormalization()(x3)
    #
    # # x4 = Conv2D(512, (3, 3), strides=2, padding="same")(x3)
    # # x4 = MaxPooling2D((2, 2))(x4)
    # # x4 = BatchNormalization()(x4)
    #
    # x5 = GlobalAveragePooling2D()(x3)
    # x5 = Activation("relu")(x5)
    #
    # x6 = Dense(1024, "relu")(x5)
    # x6 = BatchNormalization()(x6)
    # x7 = Dense(512, "relu")(x6)
    # x7 = BatchNormalization()(x7)
    # x8 = Dense(256, "relu")(x7)
    # x8 = BatchNormalization()(x8)
    # x8 = Dropout(.2)(x8)
    # x9 = Dense(1)(x8)
    # pred = Activation("softmax")(x9)

    # model 3

    # X_input = Input(input_shape)  # shape=(?, 240, 240, 3)
    #
    # x1 = Conv2D(32, (3,3), activation='relu' )(X_input)
    # x1 = MaxPooling2D((2, 2))(x1)
    #
    # x2 = Conv2D(64, (3,3),activation='relu')(x1)
    # x2 = MaxPooling2D((2, 2))(x2)
    #
    # x3 = Conv2D(128, (3, 3), activation='relu')(x2)
    # x3 = MaxPooling2D((2, 2))(x3)
    #
    # x4 = Conv2D(128, (3, 3),  activation='relu')(x3)
    # x4 = MaxPooling2D((2, 2), padding='same')(x4)
    #
    # x5 = Flatten()(x4)
    # x5 = Dense(512,activation='relu')(x5)
    # X = Dense(1, activation='sigmoid', name='fc')(x5)  # shape=(?, 1)

    # Create model. This creates my Keras model instance, i will use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X1, name='BrainDetectionModel')

    return model


IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
model = build_model(IMG_SHAPE)
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}', histogram_freq=1)

# tensorboard --logdir=logs    command for tensorboard interface

sess = tf.compat.v1.Session()
file_writer = tf.summary.create_file_writer(f'logs/{log_file_name}')
sess.close()

# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath = "cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}"

# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(
    "models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

# Train the model
#
# # # The smaller the batch the less accurate the estimate of the gradient will be
# start_time = time.time()
#
# model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

#
# end_time = time.time()
# execution_time = (end_time - start_time)
# print(f"Elapsed time: {hms_string(execution_time)}")
# # #
# # #
# #
# print(X_train.shape[0])
# print(y_train.shape[0])
# y_train_svm = y_train.reshape(y_train.shape[0], )
# X_train_svm = X_train.reshape(X_train.shape[0], 172800)
# print('Training data and target sizes: \n{}, {}'.format(X_train_svm.shape, y_train_svm.shape))
# X_val_svm = X_val.reshape(X_val.shape[0], 172800)
# y_val_svm = y_val.reshape(y_val.shape[0], )
# X_test_svm = X_test.reshape(X_test.shape[0], 172800)
# y_test_svm = y_test.reshape(y_test.shape[0], )
# print('Test data and target sizes: \n{}, {}'.format(X_test_svm.shape, y_test_svm.shape))


#SUPPORT VECTOR MACHINE ALGORITHM

## Create a classifier: a support vector classifier
# param_grid ={'C': [1], 'kernel': ['linear']}
# svc = svm.SVC()
# classifier = GridSearchCV(svc, param_grid, verbose=3)
# classifier.fit(X_train_svm, y_train_svm)
#
# y_pred_svm = classifier.predict(X_test_svm)
# # %%
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(y_test_svm, y_pred_svm)))

# scores = cross_val_score(svc, X_train_svm, y_train_svm, cv=5)
#
# # Calcularea scorului mediu
# mean_score = scores.mean()
#
# # Afișarea scorului mediu
# print("Media acuratetii :", mean_score)
# # #saving svm model
#
# joblib.dump(classifier, 'svm_2nd_model.pkl')

# # later loading
#
# joblib.load("model_file_name.pkl")
#

# RANDOM FOREST ALGORITHM


# Reshape X_train to (n_samples, n_features)  random forest functioneaza pe 2d  array
# si facem reshape de la datele noastre de 3d array ca sa extragem n_features

# X_train_reshaped = X_train[..., 0]
# X_test_reshaped = X_test[..., 0]
#
# X_train_rf = []
# X_test_rf = []
# for image in X_train_reshaped:
#     features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
#     X_train_rf.append(features)
# X_train_rf = np.array(X_train_rf)
# for image in X_test_reshaped:
#     features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
#     X_test_rf.append(features)
# X_test_rf = np.array(X_test_rf)
#
# y_train_rf = y_train.ravel()
# y_test_rf = y_test.ravel()
#
# print("X_train_rf : " + str(X_train_rf.shape))
# print("y_train_rf : " + str(y_train_rf.shape))
# print("X_test_rf : " + str(X_test_rf.shape))
# print("y_test_rf : " + str(y_test_rf.shape))
#
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# # Train the model  # am obtinut 82% precizie
# rf_model.fit(X_train_rf, y_train_rf)
#
# # Step 4: Model Evaluation
# # Make predictions on the test set
# y_pred = rf_model.predict(X_test_rf)
#
# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test_rf, y_pred)
# precision = precision_score(y_test_rf, y_pred)
# recall = recall_score(y_test_rf, y_pred)
# f1 = f1_score(y_test_rf, y_pred)
#
# # Print the evaluation metrics
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)
# #

# joblib.dump(rf_model, 'random_forest_model.pkl')
#
#
# # Load the trained model from disk
# loaded_model = joblib.load('random_forest_model.pkl')

#knn model

# X_train_reshaped = X_train[..., 0]
# X_test_reshaped = X_test[..., 0]
# #
# y_train_knn = y_train.ravel()
# y_test_knn = y_test.ravel()
# X_train_knn = []
# X_test_knn = []
# for image in X_train_reshaped:
#
#     features = hog(image, orientations= 9 ,pixels_per_cell = (16, 16) ,cells_per_block = (2, 2), block_norm = 'L2-Hys')
#     X_train_knn.append(features)
#
# X_train_knn = np.array(X_train_knn)
#
# for image in X_test_reshaped:
#     features = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
#     X_test_knn.append(features)
#
# X_test_knn = np.array(X_test_knn)
# print(X_test_knn.shape)
# #
# k = 3
# knn = KNeighborsClassifier(n_neighbors=k)
#
# # Training the KNN classifier
# knn.fit(X_train_knn, y_train_knn)
#
# # Predicting on the test set
# y_pred_knn = knn.predict(X_test_knn)
#
# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test_knn, y_pred_knn)
# precision = precision_score(y_test_knn, y_pred_knn)
# recall = recall_score(y_test_knn, y_pred_knn)
# f1 = f1_score(y_test_knn, y_pred_knn)
#
# # Print the evaluation metrics
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

# joblib.dump(knn, 'knn_model.pkl')

best_model = load_model(filepath='models2/cnn-parameters-improvement-18-0.91.model')
var = best_model.metrics_names

loss, acc = best_model.evaluate(x=X_test, y=y_test)
loss1, acc1 = best_model.evaluate(x=X_val, y=y_val)


# Accuracy of the best model on the testing data :
print(f"Test Loss = {loss}")
print(f"Test Accuracy = {acc}")
print(f"Vall Loss = {loss1}")
print(f"Vall Accuracy = {acc1}")

y_test_prob = best_model.predict(X_test)
y_test_prob = np.argmax(y_test_prob, axis=1)

print(f"y_test= {y_test}")
print(f"y_tesT_prob= {y_test_prob}")

print("Classification report for best model on the testing data :\n%s\n"
      % (metrics.classification_report(y_test, y_test_prob, zero_division=0)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_test_prob))

labels = ['yes', 'no']
confusion_mtx_test = metrics.confusion_matrix(y_test, y_test_prob)
plot_confusion_matrix(confusion_mtx_test, title='Matrice confuzie pentru img test', classes=labels,
                      normalize=False)

y_val_prob = best_model.predict(X_val)
y_val_prob = np.argmax(y_val_prob, axis=1)

print("Classification report for best model on the validation data :\n%s\n"
      % (metrics.classification_report(y_val, y_val_prob, zero_division=0)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_val, y_val_prob))

labels = ['yes', 'no']
confusion_mtx_val = metrics.confusion_matrix(y_val, y_val_prob)
plot_confusion_matrix(confusion_mtx_val, title='Matrice confuzie pentru img validare', classes=labels,
                      normalize=False)
