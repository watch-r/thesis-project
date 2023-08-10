import numpy as np
import cv2 as cv
import os


class imageSharpening:

    def upscale(self, img):
        result = self.sr.upsample(img)
        faces = cv.filter2D(result, -1, self.sharpen_kernel)
        return faces

    def scale(self, scalePercent, frame):
        width = int(frame.shape[1] * scalePercent / 100)
        height = int(frame.shape[0] * scalePercent / 100)
        dim = (width, height)
        frame = cv.resize(frame, dim, fx=0, fy=0,
                          interpolation=cv.INTER_NEAREST_EXACT)
        return frame
    def imgOnly(self, img):
        resized_img = imageSharpening().upscale(img)
        return resized_img
        
    def fromPath(self,path):
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            try:
                img = cv.imread(img_path)
                resized_img = imageSharpening().upscale(img)
                cv.imwrite(img_path, resized_img)
            except (IOError, OSError):
                print(f"Error resizing image: {img_path}")

    def __init__(self):
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                        [-1, 9, -1],
                                        [-1, -1, -1]])
        self.sr = cv.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("models\LapSRN_x4.pb")
        self.sr.setModel("lapsrn", 4)
