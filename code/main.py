import cv2 as cv
import preprocess as pps
import screenShare as sShare
import faceDetection as fd
import faceGroup as fg
import numpy as np

var = None

video_path = "assets\\videos\\2-cam-1-screen.mp4"
video_processor = pps.preprocess(video_path)


while var == None:
    var = input("Is there any Screen Sharing in the video?(y/n):").lower()
    if var == 'y' or var == 'yes' or var == 'ye':
        print('Screen sharing present = True, Proceeding...')
        screenShareTime = int(input(
            'At which time the screen sharing is present?(""in Seconds""): '))
        face_list, screen_list = video_processor.video_process(
            screenShare='yes',
            screenShareTime=screenShareTime+1)
        
        # Debugging block
        # cv.imshow('Fcae Wondow', face_list[12])
        # cv.imshow('Screen Wondow', screen_list[12])
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        
        screenShare = sShare.screenShare()
        screen_sharing = screenShare.screenShareDetection(screen_list)
        print('Screen Sharing Read Sucessfully\nGoing to next Step: Face Detection: Proceeding...')
        faceDetect = fd.faceDetect()
        curr_path,home_path = faceDetect.face_read(face_list)
        print(f'Faces Path: {curr_path}\nMain Path: {home_path}')
        facegroup = fg.faceGroup(current_path=curr_path,home_path= home_path)
        facegroup.process()
    elif var == 'n' or var == 'no':
        print('No Screen Sharing!! \nProceeding to next phase...')
        face_list = video_processor.video_process(
            screenShare='no', screenShareTime=-1)
    else:
        print('Please answer the following question again... ')
        var = None


# print(np.array(screen_list))
# print(np.array(screen_sharing))
