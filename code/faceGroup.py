import os
import shutil
import cv2 as cv
from sklearn.cluster import KMeans

current_path = 'assets\\images\\output\\faces'
home_path = 'assets\\images\\output'
out_path = 'assets\\images\\output\\resized'

os.makedirs(out_path, exist_ok=True)


# image resizing
def resizing(input_dir, output_dir):
    for file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        try:
            img = cv.imread(input_path)
            resized_img = cv.resize(img, (100, 100))
            cv.imwrite(output_path, resized_img)
        except (IOError, OSError):
            print(f"Error resizing image: {input_path}")

resizing(current_path, out_path)

img_features = []
dataset_path = os.path.join(out_path)
num_of_images = len(os.listdir(dataset_path))

for i in range(1, num_of_images):
    image_name = "face-" + str(i) + ".png"
    image_path = os.path.join(dataset_path, image_name)
    rawimg = cv.imread(image_path)
    features = rawimg.flatten()
    img_features.append(features)

print('Image featuring done')

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(img_features)
kmeans.labels_

person0_path = os.path.join(home_path, 'person0')
person1_path = os.path.join(home_path, 'person1')
os.makedirs(person0_path, exist_ok=True)
os.makedirs(person1_path, exist_ok=True)

image_count = 0
person0_count = 0
person1_count = 0

for i in range(1, num_of_images):
    image_name = "face-" + str(i) + ".png"
    image_path = os.path.join(dataset_path, image_name)
    rawimg = cv.imread(image_path)
    if kmeans.labels_[image_count] == 0:
        image_name = "person_0." + str(person0_count) + ".png"
        image_path = os.path.join(person0_path, image_name)
        cv.imwrite(image_path, rawimg)
        person0_count += 1
    elif kmeans.labels_[image_count] == 1:
        image_name = "person_1." + str(person1_count) + ".png"
        image_path = os.path.join(person1_path, image_name)
        cv.imwrite(image_path, rawimg)
        person1_count += 1
    else:
        pass
    image_count += 1

print(f'''Image sorting done---
Total number of images: {num_of_images}
Number of detected images of person-0: {person0_count}
Number of detected images of person-1: {person1_count}''')
shutil.rmtree(out_path)
