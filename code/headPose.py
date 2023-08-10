from sixdrepnet import SixDRepNet
import cv2 as cv

model = SixDRepNet()
img = cv.imread('assets\images\output\person0\person_0.3.png')

pitch, yaw, roll = model.predict(img)
print(f'Pitch:{pitch}\nYaw:{yaw}\nRoll:{roll}')

# import torch
# print(torch.cuda.is_available())

# from nvidia.tao.gaze_estimation import GazeNet

# # Instantiate a GazeNet model with pretrained weights
# model = GazeNet(pretrained=True)

# # Load an image of a face
# face_image = load_image('face.jpg')

# # Estimate the gaze
# gaze = model.estimate_gaze(face_image)

# # Print the estimated gaze
# print(gaze)
