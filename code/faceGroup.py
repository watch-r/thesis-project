import os
import shutil
import cv2 as cv
from sklearn.cluster import KMeans
import imageSharpening as ishrpn


class faceGroup:
    def __init__(self, current_path, home_path):
        self.current_path = current_path
        self.home_path = home_path
        out_path = 'resized'
        self.out_path = os.path.join(home_path, out_path)
        os.makedirs(self.out_path, exist_ok=True)

    def create_output_directories(self):
        person0_path = os.path.join(self.home_path, 'person0')
        person1_path = os.path.join(self.home_path, 'person1')
        os.makedirs(person0_path, exist_ok=True)
        os.makedirs(person1_path, exist_ok=True)
        return person0_path, person1_path

    def resize_images(self, input_dir, output_dir):
        sharpen = ishrpn.imageSharpening()
        for file in os.listdir(input_dir):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            try:
                img = cv.imread(input_path)
                resized_img = cv.resize(img, (100, 100))
                resized_img = sharpen.imgOnly(resized_img)
                cv.imwrite(output_path, resized_img)
            except (IOError, OSError):
                print(f"Error resizing image: {input_path}")
        print('Image Sharpening Done (4x)')

    def process(self):
        os.makedirs(self.out_path, exist_ok=True)
        self.resize_images(self.current_path, self.out_path)

        img_features = []
        dataset_path = os.path.join(self.out_path)
        num_of_images = len(os.listdir(dataset_path))

        for i in range(1, num_of_images):
            image_name = "face-" + str(i) + ".png"
            image_path = os.path.join(dataset_path, image_name)
            rawimg = cv.imread(image_path)
            features = rawimg.flatten()
            img_features.append(features)

        print('Image featuring done')

        kmeans = KMeans(n_clusters=2, random_state=42,
                        n_init=10).fit(img_features)
        kmeans.labels_

        person0_path = os.path.join(self.home_path, 'person0')
        person1_path = os.path.join(self.home_path, 'person1')
        os.makedirs(person0_path, exist_ok=True)
        os.makedirs(person1_path, exist_ok=True)

        image_count = 0
        person0_count = 0
        person1_count = 0

        for i in range(1, num_of_images):
            image_name = "face-" + str(i) + ".png"
            image_path = os.path.join(dataset_path, image_name)
            rawimg = cv.imread(image_path)
            # rawimg = sharpen.imgOnly(rawimg)
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
        print(f'Image sorting done---\nTotal number of images: {num_of_images}\nNumber of detected images of person-0: {person0_count}\nNumber of detected images of person-1: {person1_count}')

        shutil.rmtree(self.out_path)


# * Test Purpose Only
current_path = 'assets\\images\\output\\faces'
home_path = 'assets\\images\\output'
processor = faceGroup(current_path, home_path)
processor.process()
