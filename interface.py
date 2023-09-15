import cv2
import imutils
import tkinter as tk
import joblib
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow import keras
from skimage.feature import hog
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_cnn = keras.models.load_model(filepath='models2/cnn-parameters-improvement-18-0.91.model')
model_svm = joblib.load("model_file_name.pkl")
model_rf = joblib.load("random_forest_model.pkl")
model_knn = joblib.load("knn_model.pkl")


def crop_brain_contour(image):
    # imaginea pe nivele de gri

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # bluram putin imaginea ( ajuta la eliminarea zgomotului Gaussian )

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Facem operatie de threshold pt a face imaginea din gray scale in binary
    # Operatiile de eroziune si dilatare sunt folosite pt eliminarea regiunilor mici de zgomot .

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Functie pt preluat conturul exterior al imaginii
    # ( CHAIN_APPROX_NONE ia tot conturul , CHAIN_APPROX_SIMPLE ia doar 4 puncte exterioare)
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

    return new_image


def apply_threshold(prediction_value):
    threshold = 0.80

    if prediction_value >= threshold:
        return True
    else:
        return False


IMG_WIDTH, IMG_HEIGHT = (240, 240)


def detect_tumor_cnn():
    # Get the selected image file
    # file_path = filedialog.askopenfilename(initialdir="yes", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    global selected_file_path

    if selected_file_path:

        # if file_path:

        # Load the image using PIL  adasdadsa  a
        image = cv2.imread(selected_file_path)
        # Display the image on the interface

        image = crop_brain_contour(image)
        # resize image
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(image)

        # Display the image on the interface
        image_tk = ImageTk.PhotoImage(pil_image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk

        image = np.array(image)
        # Expand dimensions to add the batch dimension
        image = np.expand_dims(image, axis=0)
        # normalizam valorile pentru computare mai rapida
        image = image / 255.

        prediction = model_cnn.predict(image)
        prediction = np.argmax(prediction, axis=1)
        print(prediction)

        predicted_value = float(prediction)

        tumor_present = apply_threshold(predicted_value)

        if tumor_present:
            result_text_cnn.set(f"CNN Algorithm : Tumor detected !")
        else:
            result_text_cnn.set(f"CNN Algorithm : No tumor detected.")


def detect_tumor_svm():
    # Get the selected image file
    global selected_file_path

    if selected_file_path:
        # Load the image using PIL  adasdadsa  a
        image = cv2.imread(selected_file_path)
        # Display the image on the interface
        image = crop_brain_contour(image)
        # resize image
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(image)
        # Display the image on the interface
        image_tk = ImageTk.PhotoImage(pil_image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk

        image = np.array(image)
        # Expand dimensions to add the batch dimension
        image = np.expand_dims(image, axis=0)
        # normalizam valorile pentru computare mai rapida
        image = image / 255.
        flattened_image = image.flatten().reshape(1, -1)
        prediction = model_svm.predict(flattened_image)

        print(prediction)
        predicted_value = float(prediction)

        tumor_present = apply_threshold(predicted_value)

        if tumor_present:
            result_text_svm.set(f"SVM Algorithm : Tumor detected !")
        else:
            result_text_svm.set(f"SVM Algorithm : No tumor detected.")


def detect_tumor_rf():
    # Get the selected image file
    global selected_file_path

    if selected_file_path:

        # Load the image using PIL
        image = cv2.imread(selected_file_path)
        # Display the image on the interface
        image = crop_brain_contour(image)
        # resize image
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(image)
        # Display the image on the interface
        image_tk = ImageTk.PhotoImage(pil_image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk

        image = image / 255.

        image_reshaped = image[..., 0]
        features = hog(image_reshaped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        image_reshaped = np.vstack(features)

        image_reshaped = np.array(image_reshaped)

        image_reshaped = np.expand_dims(image_reshaped, axis=0)
        image_reshaped = np.squeeze(image_reshaped, axis=-1)

        prediction = model_rf.predict(image_reshaped)

        print(prediction)
        predicted_value = float(prediction)

        tumor_present = apply_threshold(predicted_value)

        if tumor_present:
            result_text_rf.set(f"Random Forest Alg.: Tumor detected!")
        else:
            result_text_rf.set(f"Random Forest Alg.: No tumor detected")


def detect_tumor_knn():
    # Get the selected image file
    global selected_file_path

    if selected_file_path:
        # Load the image using PIL
        image = cv2.imread(selected_file_path)
        # Display the image on the interface
        image = crop_brain_contour(image)
        # resize image
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(image)
        # Display the image on the interface
        image_tk = ImageTk.PhotoImage(pil_image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk

        image = image / 255.

        image_reshaped = image[..., 0]
        features = hog(image_reshaped, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       block_norm='L2-Hys')
        image_reshaped = np.vstack(features)

        image_reshaped = np.array(image_reshaped)

        image_reshaped = np.expand_dims(image_reshaped, axis=0)
        image_reshaped = np.squeeze(image_reshaped, axis=-1)

        prediction = model_knn.predict(image_reshaped)

        print(prediction)
        predicted_value = float(prediction)

        tumor_present = apply_threshold(predicted_value)

        if tumor_present:
            result_text_knn.set(f"KNN Algorithm: Tumor detected!")
        else:
            result_text_knn.set(f"KNN Algorithm: No tumor detected")


window = tk.Tk()
window.title("Brain Tumor Detection")
window.geometry("800x600")
window.configure(bg="#f2f2f2")

result_texts = []  # List to store result_text variables

# Global variable to store the file path
selected_file_path = ""


def detect_tumor_all():
    global selected_file_path

    # Get the selected image file
    selected_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if selected_file_path:
        # Load the image using PIL
        image = Image.open(selected_file_path)
        image = image.resize((300, 300))

        # Display the image on the interface
        image_tk = ImageTk.PhotoImage(image)
        image_display.configure(image=image_tk)
        image_display.image = image_tk

        # Call each detect_tumor function and print the result_texts
        detect_tumor_knn()
        detect_tumor_rf()
        detect_tumor_svm()
        detect_tumor_cnn()


# Create a button to browse and select an image
browse_button = tk.Button(window, text="Select Image", command=detect_tumor_all, font=("Helvetica", 12), bg="#ffffff",
                          fg="#333333", relief="raised")
browse_button.pack(pady=10)

# Create a frame for image display
image_frame = tk.Frame(window, bg="#f2f2f2")
image_frame.pack(pady=10)

# Create an image label
image_display = tk.Label(image_frame, bg="#ffffff", relief="solid", width=300, height=300)
image_display.pack()

# Create labels for the algorithm results
result_text_knn = tk.StringVar()
result_text_rf = tk.StringVar()
result_text_svm = tk.StringVar()
result_text_cnn = tk.StringVar()

result_label_knn = tk.Label(window, textvariable=result_text_knn, font=("Helvetica", 14), bg="#f2f2f2")
result_label_knn.pack(pady=5)

result_label_rf = tk.Label(window, textvariable=result_text_rf, font=("Helvetica", 14), bg="#f2f2f2")
result_label_rf.pack(pady=5)

result_label_svm = tk.Label(window, textvariable=result_text_svm, font=("Helvetica", 14), bg="#f2f2f2")
result_label_svm.pack(pady=5)

result_label_cnn = tk.Label(window, textvariable=result_text_cnn, font=("Helvetica", 14), bg="#f2f2f2")
result_label_cnn.pack(pady=5)

result_texts.extend([result_text_knn, result_text_rf, result_text_svm, result_text_cnn])

# Run the Tkinter event loop
window.mainloop()
