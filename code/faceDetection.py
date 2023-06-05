import cv2 as cv
import os
import face_recognition as fcrecgntn

frames=[]
frame_counter = 0
current_time_ms =0

img_cnt = 1
face_location=[]

video_path = 'assets\\videos\\only-faces.mp4'

directory='faces'
path = 'assets\\images\\output'
output_dir = os.path.join(path, directory)
os.makedirs(output_dir, exist_ok=True)


cap = cv.VideoCapture(video_path)

while True:
    cap.set(cv.CAP_PROP_POS_MSEC, current_time_ms)
    success, frame = cap.read()
    if success is False:
        if frame_counter == 0:
            print("unSucessfully Read")
        else:
            print(f"All Frames read Sucessfully\nTotal Frames: {frame_counter}\nTotal Images: {img_cnt} ")
        break

    scale_percent = 100 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    frame_re = cv.resize(frame, dim, fx=0, fy=0,
                        interpolation=cv.INTER_NEAREST_EXACT)
    # frame_0 = cv.resize(frame_re, dim, fx=0, fy=0,
    #                     interpolation=cv.INTER_LANCZOS4)
    frame_0 = frame_re
    
    rgb_frame = cv.cvtColor(frame_0, cv.COLOR_BGR2RGB)
    face_locations = fcrecgntn.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv.rectangle(frame_0, (left, top), (right, bottom), (0, 0, 255), 2)
        # Display the resulting image
        faces = frame_0[top:bottom, left:right]
        faces = cv.resize(faces, (100, 100))
        
        # Scaling the images
        scale_percent = 100 # percent of original size
        width = int(faces.shape[1] * scale_percent / 100)
        height = int(faces.shape[0] * scale_percent / 100)
        dim = (width, height)
        faces = cv.resize(faces, dim, fx=0, fy=0,
                        interpolation=cv.INTER_NEAREST_EXACT)
        
        cv.imwrite(os.path.join(output_dir , f"face-{img_cnt}.png"), faces)
        img_cnt += 1
        
    frame_counter += 1
    current_time_ms += (1000/.5)





# class faceDetect:
#     def __init__(self):
#         self.farmes = []
    
        
#     def video_read(self, path):
#         cap = cv.VideoCapture(path)
#         frame_counter = 0
#         while True:
#             success, frame = cap.read()
#             if success is False:
#                 if frame_counter == 0:
#                     print("unSucessfully Read")
#                 else:
#                     print("All Frames read Sucessfully")
#                 break

#             frame_counter += 1

#             frame = cv.resize(frame, (1080, 720), fx=0, fy=0,
#                               interpolation=cv.INTER_CUBIC)
#             self.frames.append(frame)
            
#         return self.frames
#     def face_read(self):
#         frames = self.video_read(video_path)
#         cv.imshow('Fcae Wondow', frames[12])
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#         # face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
