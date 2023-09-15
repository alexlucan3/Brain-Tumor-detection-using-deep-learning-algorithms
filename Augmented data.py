import cv2
from os import listdir
from keras.preprocessing.image import ImageDataGenerator


def augment_data(file_dir, n_generated_samples, save_to_dir):
    """
    Arguments:
        file_dir: A string representing the directory where images that we want to augment are found.
        n_generated_samples: A string representing the number of generated samples using the given image.
        save_to_dir: A string representing the directory in which the generated images will be saved.
    """

    # from keras.preprocessing.image import ImageDataGenerator
    # from os import listdir

    data_gen = ImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.1,
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest'
                                  )

    for filename in listdir(file_dir):
        # load the image
        image = cv2.imread(file_dir + '\\' + filename)
        # am definite imread(param) in init.py

        # reshape the image
        image = image.reshape((1,) + image.shape)
        # prefix of the names for the generated samples.
        save_prefix = 'aug_' + filename[:-4]
        # generate 'n_generated_samples' sample images
        i = 0
        # for batch
        for _ in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,
                               save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break


augmented_data_path = 'augmented_data/'

# augment data for the examples with label equal to 'yes' representing tumorous examples
augment_data(file_dir='yes', n_generated_samples=6, save_to_dir=augmented_data_path + 'yes')
# augment data for the examples with label equal to 'no' representing non-tumorous examples
augment_data(file_dir='no', n_generated_samples=9, save_to_dir=augmented_data_path + 'no')


def data_summary(main_path):
    yes_path = main_path + 'yes'
    no_path = main_path + 'no'

    # number of files (images) that are in the folder named 'yes' that represent tumorous (positive) examples
    m_pos = len(listdir(yes_path))
    # number of files (images) that are in  the folder named 'no' that represent non-tumorous (negative) examples
    m_neg = len(listdir(no_path))
    # number of all examples
    m = (m_pos + m_neg)

    pos_perc = (m_pos * 100.0) / m
    neg_perc = (m_neg * 100.0) / m

    print(f"Numarul de exemple: {m}")
    print(f"Procent exemple pozitive: {pos_perc}%, nr exemple pozitive: {m_pos}")
    print(f"Procent exemple negative: {neg_perc}%, nr exemple negative: {m_neg}")


data_summary(augmented_data_path)
