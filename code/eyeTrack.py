import os
import math
import cv2 as cv
import numpy as np
import mediapipe as mp

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]


class eyeTrack:

    def __init__(self):
        pass

    def landmarksDetection(self, img, results):
        img_height, img_width = img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height))
                      for point in results.multi_face_landmarks[0].landmark]

        # returning the list of tuples for each landmarks
        return mesh_coord

    def euclaideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    def eyesExtractor(self, img, right_eye_coords, left_eye_coords):
        # converting color image to  scale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # getting the dimension of image
        dim = gray.shape

        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)

        # drawing Eyes Shape on mask with white color
        cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

        # draw eyes image on mask, where white shape is
        eyes = cv.bitwise_and(gray, gray, mask=mask)
        eyes[mask == 0] = 155

        # getting minium and maximum x and y  for right and left eyes
        # For Right Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        # croping the eyes from mask
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

        # returning the cropped eyes
        return cropped_right, cropped_left

    def positionEstimator(self, cropped_eye):
        # getting height and width of eye
        h, w = cropped_eye.shape

        # remove the noise from images
        gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
        median_blur = cv.medianBlur(gaussain_blur, 3)

        # applying thrsholding to convert binary_image
        ret, threshed_eye = cv.threshold(
            median_blur, 130, 255, cv.THRESH_BINARY)

        # create fixd part for eye with
        piece = int(w/3)

        # slicing the eyes into three parts
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece + piece:w]

        # calling pixel counter function
        eye_position = self.pixelCounter(right_piece, center_piece, left_piece)

        return eye_position

    def pixelCounter(self, first_piece, second_piece, third_piece):
        # counting black pixel in each part
        right_part = np.sum(first_piece == 0)
        center_part = np.sum(second_piece == 0)
        left_part = np.sum(third_piece == 0)
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]

        # getting the index of max values in the list
        max_index = eye_parts.index(max(eye_parts))
        pos_eye = ''
        if max_index == 0:
            pos_eye = "r"
        elif max_index == 1:
            pos_eye = 'c'
        elif max_index == 2:
            pos_eye = 'l'
        else:
            pos_eye = "n"

        return pos_eye

    def process(self, path):
        map_face_mesh = mp.solutions.face_mesh
        with map_face_mesh.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.4) as face_mesh:
            # for 1 person
            eye_data = []
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                try:
                    frame = cv.imread(img_path)
                    # frame_height, frame_width = frame.shape[:2]
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        mesh_coords = self.landmarksDetection(
                            frame, results)

                        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                        left_coords = [mesh_coords[p] for p in LEFT_EYE]
                        crop_right, crop_left = self.eyesExtractor(
                            frame, right_coords, left_coords)
                        # Resizing
                        crop_right = cv.resize(crop_right, (200, 100))
                        crop_left = cv.resize(crop_left, (200, 100))

                        eye_position_right = self.positionEstimator(crop_right)

                        eye_position_left = self.positionEstimator(crop_left)

                        # building eye array
                        if (eye_position_left == 'n' or eye_position_right == 'n'):
                            eye_data.append('n')
                        elif (eye_position_left == eye_position_right):
                            eye_data.append(eye_position_right)
                        elif (eye_position_left != eye_position_right):
                            # L = [eye_position_left, eye_position_right]
                            # L = random.choice(L)
                            # print('Random: '+L)
                            # eye_data1.append(L)
                            eye_data.append('n')
                        else:
                            eye_data.append('c')
                except (IOError, OSError):
                    print(f"Error resizing image: {img_path}")
            return eye_data


# path = 'assets/images/output/person1'
# # # sinImg = cv.imread('assets\\images\\output\\person1\\person_1.34.png')
# eyetrack = eyeTrack()
# personOneEyeData = eyetrack.process(path=path)
# # # dataImg = headOrient.singleImage(sinImg)
# # print(f'Person 1: {personOneHeadData}\nTotal Scanned Count: {len(personOneHeadData)}')
# # # print(f'Data: {dataImg}')
# print(personOneEyeData)
