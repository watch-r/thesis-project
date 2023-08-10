import cv2 as cv
import os
import face_recognition as fcrecgntn
import numpy as np

class faceDetect:
    def __init__(self):
        self.frames = []
        self.frame_counter = 0
        self.current_time_ms = 0

        self.img_cnt = 1
        self.face_location = []

        self.directory = 'faces'
        self.path = 'assets\\images\\output'
        self.output_dir = os.path.join(self.path, self.directory)
        os.makedirs(self.output_dir, exist_ok=True)
        

    def face_read(self, face_list):
        
        for frame in face_list:
            
            frame_0 = frame
            rgb_frame = cv.cvtColor(frame_0, cv.COLOR_BGR2RGB)
            face_locations = fcrecgntn.face_locations(rgb_frame)

            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                # cv.rectangle(frame_0, (left, top), (right, bottom), (0, 0, 255), 2)
                # Display the resulting image
                faces = frame_0[top:bottom, left:right]
                faces = cv.resize(faces, (100, 100))
                cv.imwrite(os.path.join(
                    self.output_dir, f"face-{self.img_cnt}.png"), faces)
                self.img_cnt += 1
                
        print(f'Face Detection Sucessful\nTotal Detected faces: {self.img_cnt}')
        return self.output_dir, self.path
