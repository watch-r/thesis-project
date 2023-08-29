from sixdrepnet import SixDRepNet
import cv2 as cv
import os


class headPose:
    
    def dataOfHead(self,path):
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            try:
                img = cv.imread(img_path)
                
                rotation = self.singleImage(img)
                self.data.append(rotation)
            except (IOError, OSError):
                print(f"Error Finding Picture: {img_path}")
        return self.data
    
    def singleImage(self,img):
        pitch, yaw, roll = self.model.predict(img)
        if pitch > -10 and pitch < 10:
            if yaw > -10 and yaw < 10:
                    # print('F')
                rotation='f'
            elif yaw < -10:
                # print('L')
                rotation='l'
            elif yaw > 10:
                # print('R')
                rotation='r'
        elif pitch < -10:
            rotation='d'
        elif pitch > 10:
            rotation='u'
        else:
            rotation='N'
        return rotation
   
    def __init__(self):
        self.model = SixDRepNet()
        self.data =[]
        
 
# path =  'assets\images\output\person1' 
# # sinImg = cv.imread('assets\\images\\output\\person1\\person_1.34.png')
# headOrient = headPose()
# personOneHeadData=headOrient.dataOfHead(path)
# # dataImg = headOrient.singleImage(sinImg)
# print(f'Person 1: {personOneHeadData}\nTotal Scanned Count: {len(personOneHeadData)}')
# # print(f'Data: {dataImg}')

# import torch
# print(torch.cuda.is_available())
