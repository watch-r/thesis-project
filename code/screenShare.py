import cv2 as cv


class screenShare:
    def __init__(self):
        self.screen_share = False
        self.screenShare_count = 0
        self.screen_sharing = []

    def screenShareDetection(self, screen_list):
        # converting the video to a processable format
        # Screen sharing detection
        for frame in screen_list:
            image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(image_gray, 100, 230, 0)
            contours, hierarchy = cv.findContours(
                thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
            contour_count = len(contours)
            if not contour_count > 100:
                self.screen_share = 0
            else:
                self.screen_share = 1
                self.screenShare_count += 1

            self.screen_sharing.append(self.screen_share)
        print(
            f'Screen Sharing Detection was Sucessful \nScreen Shared(frames): {self.screenShare_count}')
        return self.screen_sharing
