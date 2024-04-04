from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QScrollArea, QMessageBox
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject

import io
import numpy as np
import napari
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'

import matplotlib.pyplot as plt
import cv2
import threading
import h5py
class nematostella_detector(QWidget):
    add_image_signal = pyqtSignal(np.ndarray, str)
    show_message_signal = pyqtSignal(str)  # Add this line
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.selected_layer_name = None
        self.mData = None 

        # Main layout
        layout = QVBoxLayout()
        
        # Instead of setting layout directly to self, wrap it in a scrollable area
        scrollArea = QScrollArea()  # Create a new QScrollArea
        scrollArea.setWidgetResizable(True)  # Make the scroll area resizable

        # Create a container widget and set your layout on it
        containerWidget = QWidget()
        containerWidget.setLayout(layout)
        
        # Set the container widget as the scroll area's widget
        scrollArea.setWidget(containerWidget)

        # Main layout for the scrollable content
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(scrollArea)  # Add the scroll area to the main layout
        
        self.setLayout(mainLayout)  # Set the main layout to the class

        # link signals
        self.add_image_signal.connect(self.displayImage)
        self.show_message_signal.connect(self.displayPopUpMessage)

        # Step 1: Load Image Stack
        self.layer_dropdown = QComboBox()
        self.layer_dropdown.addItems([layer.name for layer in viewer.layers])
        load_button = QPushButton("Load Image Stack")
        load_button.clicked.connect(self.load_image_stack)
        refresh_stack_button = QPushButton("Refresh Image Stack")
        refresh_stack_button.clicked.connect(self.refresh_image_stack)
        layout.addWidget(QLabel("Step 1 - Load Data"))
        layout.addWidget(self.layer_dropdown)
        layout.addWidget(load_button)
        layout.addWidget(refresh_stack_button)

        
        # Step 2: Detect Circles
        detect_button = QPushButton("Detect Circles")
        detect_button.clicked.connect(self.detect_circles)
        self.param1_input = QLineEdit("100")
        self.param2_input = QLineEdit("50")
        self.minRadius_input = QLineEdit("80")
        self.maxRadius_input = QLineEdit("150")
        self.frame_number_input = QLineEdit("1")
        self.circles_list = QLabel("Detected circles will be listed here.")
        clear_button = QPushButton("Clear Selection")
        clear_button.clicked.connect(self.clear_selection)
        layout.addWidget(QLabel("Step 2 - Detect Circles"))
        layout.addWidget(detect_button)
        layout.addWidget(QLabel("Parameter 1:"))
        layout.addWidget(self.param1_input)
        layout.addWidget(QLabel("Parameter 2:"))
        layout.addWidget(self.param2_input)
        layout.addWidget(QLabel("Min Radius:"))
        layout.addWidget(self.minRadius_input)
        layout.addWidget(QLabel("Max Radius:"))
        layout.addWidget(self.maxRadius_input)        
        layout.addWidget(QLabel("Frame Number to detect Circles:"))
        layout.addWidget(self.frame_number_input)
        layout.addWidget(clear_button)
        layout.addWidget(self.circles_list)

        # Step 3: Detect Baseline
        baseline_button = QPushButton("Detect Baseline")
        baseline_button.clicked.connect(self.detect_baseline)
        self.minFramePos_input = QLineEdit("0")
        self.maxFramePos_input = QLineEdit("10")
        layout.addWidget(QLabel("Step 3 - Detect Baseline"))
        layout.addWidget(QLabel("Min Frame Position:"))
        layout.addWidget(self.minFramePos_input)
        layout.addWidget(QLabel("Max Frame Position:"))
        layout.addWidget(self.maxFramePos_input)    
        layout.addWidget(baseline_button)
        

        # Step 4: Determine Sleep Behaviour
        sleep_button = QPushButton("Determine Sleep Behaviour")
        sleep_button.clicked.connect(self.determine_sleep_behavior)
        layout.addWidget(QLabel("Step 4 - Determine Sleep Behaviour"))
        layout.addWidget(sleep_button)

        self.setLayout(layout)

    def load_image_stack(self):
        self.selected_layer_name = self.layer_dropdown.currentText()
        print("Loading image stack from layer:", self.selected_layer_name)
        self.mData = self.viewer.layers[self.selected_layer_name].data
        # Implement your logic here

    def displayPopUpMessage(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(message)
        msgBox.setWindowTitle("Information")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()

    @pyqtSlot(np.ndarray, str)
    def displayImage(self, mImage, layerName):
        self.viewer.add_image(mImage, name=layerName, colormap='gray')

    def refresh_image_stack(self):
        print("Refreshing image stack")
        self.layer_dropdown.addItems([layer.name for layer in self.viewer.layers])
        
    def detect_circles(self):
        param1 = int(self.param1_input.text())
        param2 = int(self.param2_input.text())
        minRadius = int(self.minRadius_input.text())
        maxRadius = int(self.maxRadius_input.text())
        frame_number = int(self.frame_number_input.text())
        self.viewer.dims.set_point(value=frame_number, axis=0)
        if 0:
            param1 = 100
            param2 = 50
            minRadius = 100
            maxRadius = 150
            frame_number = int(self.frame_number_input.text())
        
        # Optionally convert the frame to grayscale
        if self.mData is None or type(self.mData)!=np.ndarray:
            self.displayPopUpMessage("Please first select the data you want to process!")
            return
        if len(self.mData)>3: # RGB
            mFrame = self.mData[int(frame_number),:,:,:]
            frame_gray = cv2.cvtColor(mFrame, cv2.COLOR_BGR2GRAY)
        else: # Mono
            mFrame = self.mData[int(frame_number),:,:]
            frame_gray = mFrame
            frame_gray = np.uint8(255.*frame_gray/np.max(frame_gray))   
        if "Detected ROIs" in self.viewer.layers:
            self.viewer.layers["Detected ROIs"]
        else:
            self.viewer.add_image(frame_gray, name='Detected ROIs', colormap='gray')
        # %%
        # detect circles in the image
        frame_gray = cv2.medianBlur(frame_gray, 5)
        rows = frame_gray.shape[0]
        circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                    param1=param1, param2=param2,
                                    minRadius=minRadius, maxRadius=maxRadius)
        # display all circles
        allMasks = []
        if circles is not None:
            circles = np.uint16(np.around(circles))  # Convert the circle parameters to integers
            for i in circles[0, :]:
                center = (i[0], i[1])  # Circle center
                radius = i[2]  # Circle radius

                # Create a mask with the same dimensions as the image, initially filled with zeros (black)
                mask = np.zeros_like(frame_gray)
                cv2.circle(mask, center, radius, (255), thickness=-1)
                allMasks.append(mask)
                
                if 0:
                    # Fill the circle in the mask with ones (white)
                    cv2.circle(mask, center, radius, (255), thickness=-1)

                    # Use the mask to select the pixels inside the circle
                    circle_pixels = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

                    self.viewer.add_image(circle_pixels, name='Detected ROIs', colormap='gray', blending="additive")
                else:
                    cv2.circle(frame_gray, (i[0], i[1]), i[2], (255, 0, 255), 5) 
            self.circles = circles
            self.allMasks = allMasks
            self.viewer.layers["Detected ROIs"].data = frame_gray
            
            
    def clear_selection(self):
        print("Clearing circle selection")
        self.circles_list.setText("Circle selection cleared.")
        self.viewer.layers["Detected ROIs"].data= np.empty((0,2))
        self.circles = None
        self.allMasks = None
        
    def detect_baseline(self):
        self.is_detecting_baseline = True
        self.thread = threading.Thread(target=self.detect_baseline_background)
        self.thread.start()
        
    def detect_baseline_background(self):
        if self.allMasks is None: 
            self.displayPopUpMessage("Please first run the circle detection!")
            return
        # retreive parameters from GUI
        min_frame_num = int(self.minFramePos_input.text())
        max_frame_num = int(self.maxFramePos_input.text())
            
        # save the first frame
        frame_gray = cv2.cvtColor(self.mData[min_frame_num], cv2.COLOR_BGR2GRAY)
        lastFrame = frame_gray/np.mean(frame_gray)
        allDiffs = []
        iFrame = 0
        # initialize the list of tiles per mask
        allTilesPerMask = []
        for mask in self.allMasks:
            allTilesPerMask.append([])

        # iterate over all frames and compute the motion variance for each mask
        # reserve some memory 
        nFrames = max_frame_num-min_frame_num
        nMasks = len(self.allMasks)
        mMotionPerFramePerMask = np.zeros((nFrames, nMasks)) # dimensions: frameNumber, MaskNumber
        frameMeans = np.zeros(nFrames)
        for frameIndex, iFrame in enumerate(self.mData[min_frame_num:max_frame_num]):
            frame_gray = cv2.cvtColor(iFrame, cv2.COLOR_BGR2GRAY)
            frameMeans[frameIndex] = np.mean(frame_gray)
            frame_gray = frame_gray/frameMeans[frameIndex]
            
            # iterate over all masks (e.g. round well circles )
            for maskIndex, mask in enumerate(self.allMasks):
                # find roi of mask 
                roi = np.where(mask>0)
                roi = np.array([np.min(roi[0]), np.max(roi[0]), np.min(roi[1]), np.max(roi[1])])
                
                # Use the mask to select the pixels inside the circle
                circle_pixels = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
                circle_pixels_last = cv2.bitwise_and(lastFrame, lastFrame, mask=mask)
                
                # crop the circle pixels to the roi
                circle_pixels = circle_pixels[roi[0]:roi[1], roi[2]:roi[3]]
                circle_pixels_last = circle_pixels_last[roi[0]:roi[1], roi[2]:roi[3]]
                
                # cross correlation of the two images to detect motion
                diff = cv2.filter2D(circle_pixels, ddepth=-1, kernel=circle_pixels_last)
                # diff = scipy.signal.correlate2d(cv2.resize(circle_pixels,None, None, 0.25,0.25), cv2.resize(circle_pixels_last,None, None, 0.25,0.25), mode='full', boundary='fill', fillvalue=0)
                # diff = cv2.absdiff(circle_pixels, circle_pixels_last)
                # store values for all masks
                diffMean = np.mean(diff)
                mMotionPerFramePerMask[frameIndex, maskIndex] = diffMean
            print("Processing Frame "+str(frameIndex)+" / "+str(max_frame_num-min_frame_num))
            lastFrame = frame_gray.copy()
        #%
        # visualize the motion variance
        fig, ax = plt.subplots()
        for iMask in range(mMotionPerFramePerMask.shape[1]): # dimensions: frameNumber, MaskNumber
            ax.plot(mMotionPerFramePerMask[:, iMask])
        ax.set_ylabel('Motion Variance')  
        ax.set_xlabel('Frames')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Go to the start of the buffer
        imgplot = plt.imread(buf)
        buf.close()
        
        # show the plot as a napari layer 
        self.add_image_signal.emit(imgplot, "Motion Variance")
        self.is_detecting_baseline = False
            
        # store values in HD5 file (frameMeans and mMotionPerFramePerMask)
        # Creating the HDF5-file
        with h5py.File('data.h5', 'w') as f:
            # saving 
            f.create_dataset('frameMeans', data=frameMeans)
            f.create_dataset('mMotionPerFramePerMask', data=mMotionPerFramePerMask)
        self.show_message_signal.emit("Baseline detection finished!")
    
    def determine_sleep_behavior(self):
        print("Determining sleep behavior")
        # Implement your logic here
