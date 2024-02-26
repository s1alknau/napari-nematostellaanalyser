import napari
import numpy as np
import cv2 

# Open the video file
video_path = '/Users/bene/Dropbox/Circadian/Circadian_LD_20220505/Videos/Circadian_LD_20220505_m1_[185705_191715]_[20220505] Rapha.avi'
cap = cv2.VideoCapture(video_path)

mImages = []
for i in range(100):
    ret, frame = cap.read() 
    mImages.append(frame)

viewer = napari.Viewer()
viewer.add_image(np.array(mImages), name="testImages")
input('Press Enter to exit')