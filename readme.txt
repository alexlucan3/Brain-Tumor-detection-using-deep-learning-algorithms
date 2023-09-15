Brain tumor detection is a very complex process. Machine learning algorithms were used for better diagnosis of these tumors. The Python programming language was used to implement the solution. Our solution uses libraries such as TensorFlow, Scikit-learn and OpenCV to be able to work with MRI images. This paper proposes the implementation and comparison of four algorithms that have gained prominence in tumor detection, namely: Convolutional Neural Networks (CNN), SVM, Random Forest and K-NN. In order to be able to train the models in the most optimal way, we used a database comprising 250 MRI images and used a data augmentation function to increase the size of the data set to more than 2000 MRI images. After training the models, we achieved 91% accuracy for CNN, 69% accuracy for SVM, 86% accuracy for Random Forest, and 93.54% accuracy for K-NN. We also created a minimalist interface that helped us compare the performance of these algorithms side by side. As results from this paper, we can say that CNN and K-NN algorithms are the most suitable for brain tumor detection.

Building a detection model using a convolutional neural network in Tensorflow & Keras.
Used a brain MRI images data founded on Kaggle. You can find it here : https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in taking the data imbalance issue in the data.

Before data augmentation, the dataset consisted of:
155 positive and 98 negative examples, resulting in 253 example images.

After data augmentation, now the dataset consists of:
1085 positive and 980 examples, resulting in 2065 example images.

The original data is in the folders named 'yes' and 'no'. And, the augmented data in the folder named 'augmented data'.

Hope this project may help other peoples trying to learn about using of deep learning in real life problems like brain tumor detection.

If you find difficulties in understanding the code , contact me up on instagram : alex.lucan.31
