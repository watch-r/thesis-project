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
        
        # For Scaling
        # self.sr = cv.dnn_superres.DnnSuperResImpl_create() 
        # self.sr.readModel("models\LapSRN_x8.pb")
        # self.sr.setModel("lapsrn", 8)

    def face_read(self, face_video):
        
        for frame in face_video:
            # scale_percent = 110  # percent of original size
            # width = int(frame.shape[1] * scale_percent / 100)
            # height = int(frame.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # frame_0 = cv.resize(frame, dim, fx=0, fy=0,
            #                     interpolation=cv.INTER_NEAREST_EXACT)
            frame_0 = frame
            rgb_frame = cv.cvtColor(frame_0, cv.COLOR_BGR2RGB)
            face_locations = fcrecgntn.face_locations(rgb_frame)

            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                # cv.rectangle(frame_0, (left, top), (right, bottom), (0, 0, 255), 2)
                # Display the resulting image
                faces = frame_0[top:bottom, left:right]
                faces = cv.resize(faces, (100, 100))

                # Scaling the images
                # result = self.sr.upsample(faces)
                # sharpen_kernel = np.array([[-1, -1, -1],
                #                            [-1, 9, -1],
                #                            [-1, -1, -1]])
                # faces = cv.filter2D(result, -1, sharpen_kernel)

                cv.imwrite(os.path.join(
                    self.output_dir, f"face-{self.img_cnt}.png"), faces)
                self.img_cnt += 1
                
        print(f'Face Detection Sucessful\nTotal Detected faces: {self.img_cnt}')
