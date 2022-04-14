import os
import numpy as np
import time
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

X_train = np.load("train/x_128_train.npy")
y_vacuole_train = np.load("train/y_vacuole_train.npy")

X_test = np.load("test/x_128_test.npy")
y_vacuole_test = np.load("test/y_vacuole_test.npy")

valid = np.load("validation/x_128_valid.npy")
y_vacuole_valid = np.load("validation/y_vacuole_valid.npy")

full_X = np.concatenate([X_train, X_test, valid], axis = 0)
full_y_vacuole = np.concatenate([y_vacuole_train, y_vacuole_test, y_vacuole_valid], axis = 0)

def np_to_image(data, path):
    for i in range(len(data)):

        x = data[i]
        im = Image.fromarray(x)
        newpath = os.path.join(path,"x_%s"%(i)+".jpg")
        im.save(path+"x_%s"%(i)+".jpg")

bool_full_y_vacuole = full_y_vacuole != 0

pos_x_train = full_X[bool_full_y_vacuole]
neg_x_train = full_X[~bool_full_y_vacuole]

pos_labels = full_y_vacuole[bool_full_y_vacuole]
neg_labels = full_y_vacuole[~bool_full_y_vacuole]

image_gen = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = [0.1, 1]
)

path = 'data_images/pos_X_train/'

start_time = time.time()

for i in os.listdir(path):
    full_path = path + i
    img = load_img(full_path)
    img = img_to_array(img)
    array = img.reshape((1,) + img.shape)

    iterator = 0

    for batch in image_gen.flow(array,
                                save_to_dir = 'data_images/pos_X_train_oversamples/',
                                save_prefix = f"{iterator}",
                                save_format = "jpg"):
        iterator += 1
        if iterator > 1:
            break

end_time = time.time()

oversample_path = 'data_images/pos_X_train_oversamples'

print(f"{len(os.listdir(oversample_path))} images saved to {path} in {end_time-start_time} seconds")

### convert back to np arrays
# X_train_new = []
# files = './splits/altered/'
# for myFile in os.listdir(files):
#     image = imread(files + myFile)
#     X_train_new.append(image)
#
# X_train_new = np.array(X_train_train)
