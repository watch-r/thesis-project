import cv2 as cv
import numpy as np

cap = cv.VideoCapture("assets\\videos\\2-cam-1-screen.mp4")


constant = 1000
fps_counter = .5

current_time_ms = 0 * constant  # formula: milisecond = second * 1000
end_time_ms = 45 * constant

screen_share = False
screenShare_count = 0
frame_counter = 0
screen_sharing = []


while True:
    # to start the video at a specific time in milliseconds
    cap.set(cv.CAP_PROP_POS_MSEC, current_time_ms)

    current_time = int(current_time_ms / constant)

    success, frame = cap.read()
    if success is False:
        if frame_counter == 0:
            print("unSucessfully Read")
        else:
            print("All Frames read Sucessfully")
        break

    frame_counter += 1

    frame = cv.resize(frame, (1080, 720), fx=0, fy=0,
                      interpolation=cv.INTER_CUBIC)

    # converting the video to a processable format
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(image_gray, 100, 230, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
    contour_count = len(contours)

    if not contour_count > 100:
        screen_share = 0
    else:
        screen_share = 1
        screenShare_count += 1

    screen_sharing.append(screen_share)

    current_time_ms += (constant/fps_counter)

    # debugging blocks
    # if current_time_ms >= end_time_ms:  # loop breaking condition for specific chunk
    #     break

print(
    f'Total Frames Read:{frame_counter}\nTotal Screen Sharing Time: {screenShare_count}')
