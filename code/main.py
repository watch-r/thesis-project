import preprocess as pps
import screenShare as sShare
import numpy as np

video_path = "assets\\videos\\2-cam-1-screen.mp4"
video_processor = pps.preprocess(video_path)
face_list, screen_list = video_processor.video_process()
screenShare = sShare.screenShare()

screen_sharing = screenShare.screenShareDetection(screen_list)

# print(np.array(screen_list))
# print(np.array(screen_sharing))
