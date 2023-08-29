import cv2 as cv
import numpy as np

# cap = cv.VideoCapture("assets\\videos\\2-cam-1-screen.mp4")


class preprocess:

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)

        self.constant = 1000 # constant for time conversion
        self.fps_counter = float(input("How Many Frames you want to Read per Second: ")) # how many FPS we need to read

        # formula: milisecond = second * 1000
        self.current_time_ms = 0 * self.constant
        self.end_time_ms = 45 * self.constant

        self.face_list = [] # webcam faces from the video
        self.screen_list = [] # screen 
        self.frame_counter = 0

    def video_process(self,screenShare,screenShareTime):
        
        screenShareTime = screenShareTime*self.constant
        
        while True:
            # to start the video at a specific time in milliseconds
            self.cap.set(cv.CAP_PROP_POS_MSEC, self.current_time_ms)

            success, frame = self.cap.read()
            if success is False:
                if self.frame_counter == 0:
                    print("unSucessfully Read")
                else:
                    print("All Frames read Sucessfully")
                break

            self.frame_counter += 1

            frame = cv.resize(frame, (1080, 720), fx=0, fy=0,
                              interpolation=cv.INTER_CUBIC)

            # Changable module
            # first type
            # must make type wise function
            
            if screenShare == 'no':
                self.face_list.append(frame)
            elif screenShare == 'yes':
                
                if self.current_time_ms < screenShareTime: 
                    self.face_list.append(frame)
                else:
                    hight, width, _ = frame.shape

                    frameScreen = frame[0:hight, 0:930]
                    frameFace = frame[0:512, 930:width]

                    self.face_list.append(frameFace)
                    self.screen_list.append(frameScreen)

            self.current_time_ms += (self.constant/self.fps_counter)

            # # debugging blocks
            # if self.current_time_ms >= self.end_time_ms:  # loop breaking condition for specific chunk
            #     break
        print(
            f'Total Frames Read:{self.frame_counter}')
        if screenShare == 'no':
                return self.face_list
        elif screenShare == 'yes':
            return self.face_list, self.screen_list
