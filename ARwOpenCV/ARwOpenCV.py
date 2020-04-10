# -*- coding: utf-8 -*-
"""Live AR with OpenCV - Live Augmented Reality/Pose Estimation with OpenCV

This library aims at providing structure around the Augmented Reality/Pose Estimation blocks provided in OpenCV.
OpenCV provides a lot of great functions in this context along with tutorials.
However, tutorials aren't working anymore and aren't providing a structure required to build actual tools,
especially for a live context.

This library provides one-liner for:
1. Camera Calibration Dataset - Capturing, Storing and Adjusting an Images Dataset + MetaData required for the Camera Calibration.
2. Camera Calibration Signature - Based on the refined Images Dataset + MetaData, creates a signature of the camera characteristics for future image correction.
3. Image Calibration based on Camera Signature - Actual Image Correction based on the Camera Signature
4. Live Rendering - Based on the Camera Signature, creates a live aumented reality video stream

This Camera Calibration process requires a chessboard with squares.

Lots of credits for these tutorials, even if a bit clumsy, still great stuffs - I'll commit some fixes in a near future:
- https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
- https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html

Example:
    Usual complete flow (python)

        c = Calibration() # Launching Calibration Class
        c.capture() # Creating an Image Dataset and deriving the Camera Signature
        c.retune() # Adjusting the Dataset and the Camera Signature
        c.render() # Rendering AR (default is a Cube)
"""


import cv2
from datetime import datetime, timedelta
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from time import time, sleep
import json

class Calibration():
    """Full Calibration-Tuning-Rendering Process Class
    
    This class does is being used for the full calibration, tuning and rendering process,
    just launch it and follow the steps"""
    
    def __init__(self,
                 path="Calibration_Images",
                 mirror=True,  
                 chessboard=[6,9]):
        """Initiation
    
        By initiating the class, it will check if an Image Dataset or if a camera signature is available.
        Depending on the results it will suggest next steps and if possible load already existing camera signature.

        Args:
            path (str): Folder used to store the Image Dataset, Camera Signature, etc.
            mirror (bool): If true, flips the camera
            chessboard (tuple): Chessboard intersection corner shape. Example: for a chessboard with 5x5 square, 
                there will be 4x4 intersection between squares."""
        
        self.path = path
        self.mirror = mirror
        self.chessboard = chessboard
        

        # Creation of the Calibration Folder
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        
        # Informing user about next steps depending on its Calibration Folder
        with open('config.json') as f:
            config = json.load(f)
            
            if os.path.exists('{:s}//Images_MetaData.npz'.format(self.path)) & (len(glob('{:s}//Images_*.jpg'.format(self.path)))<10) | (len(glob('{:s}//Images_*.jpg'.format(self.path)))==0):
                print(config["Instructions"]["Empty"][0])
                   
            elif os.path.exists('{:s}//CameraSignature.npz'.format(self.path)):

                npz = np.load('{:s}//CameraSignature.npz'.format(self.path))
                self.mapx = npz["mapx"]
                self.mapy = npz["mapy"]
                self.mtx = npz["mtx"]
                self.dist = npz["dist"] 
                self.roi = npz["roi"]
                self.roi = npz["roi"]
                                                                                
                print(config["Instructions"]["CameraSignature"][0])
                                                                                
            elif os.path.exists('{:s}//Images_MetaData.npz'.format(self.path)) & (len(glob('{:s}//Images_*.jpg'.format(self.path)))>10):
                print(config["Instructions"]["NoCameraSignature"][0])
                                                                                  
            else:
                print(config["Instructions"]["NoImageXORMetaData"][0])
                YN_clean = input()
                                                                                  
                if YN_clean.upper()[0] == "Y":
                    path_files = (glob("{:s}//Image_*.jpg".format(self.path))+['{:s}//CameraSignature.npz'.format(self.path),'{:s}//Images_MetaData.npz'.format(self.path)])
                    for path_file in path_files:
                        try:
                            os.remove(path_file)
                        except:
                            pass
                    print(config["Instructions"]["NoImageXORMetaData"][1])
                                                                                  
                else:
                    return "0"

                                                                                  
    def capture(self, fps=1):
        """Capturing Chessboard Images
    
        Launch your camera, everytime a chessboard is being detected, it will be highlighted, 
        the camera flux will be stopped for a second and the image with the chessboard will be saved,
        along with the several metadata (related to the chessboard positionning)
        
        By pressing any key, the camera flux will stop.

        Args:
            fps (int): Limit the number of image/frame being captured per second
        
        Returns:
            /"""

        if os.path.exists('{:s}//Images_MetaData.npz'.format(self.path)):
            # Arrays to store object points and image points from all the images. If already exists.
            npz = np.load('{:s}//Images_MetaData.npz'.format(self.path))
            objpoints = list(npz["obj"])
            imgpoints = list(npz["img"])
            timepoints = list(npz["time"])
        else:
            # Arrays to store object points and image points from all the images. If doesn't exists.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            timepoints = []

        # Termination Criteria For Double Checking Iterative Process
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboard[0]*self.chessboard[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboard[1],0:self.chessboard[0]].T.reshape(-1,2)

        # Launch Camera
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        Image_proc = datetime.now() - timedelta(seconds=1/fps)
        while True:

            proc = datetime.now()

            # Reading the images
            ret, img = cap.read()

            # Image Processing
            if self.mirror: 
                img = cv2.flip(img, 1)

            # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, 
                                                     (self.chessboard[1],self.chessboard[0]),
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

            # If found, add object points, image points (after refining them)
            if ret == True:




                # Refyning Process by dounle checking around detected corners
                corners = cv2.cornerSubPix(gray,
                                           corners,
                                           (11,11), 
                                           (-1,-1), 
                                           criteria)


                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.chessboard[1], self.chessboard[0]), corners, ret)
                
                # Making sure we are not saving more than the fps parameter
                if (proc - Image_proc).total_seconds() > 1/fps:

                    # Current Image Time Stamp (being saved)
                    Image_proc = proc
                    proc_int = int(Image_proc.strftime('%H%M%S%f')[:-2])
                    timepoints.append(proc_int)

                    # Saving the image & MetaData
                    cv2.imwrite('{:s}//Image_{:d}.jpg'.format(self.path, proc_int), img)
                    objpoints.append(objp)
                    imgpoints.append(corners)

            # Text saying Image is being saved        
            if (proc - Image_proc).total_seconds() < 1/(fps*3):
                cv2.putText(img=img,
                    text='Image Saved: Image_{:d}.jpg'.format(proc_int), 
                    org=(0, img.shape[0]-5), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=(0,0,0),
                    lineType=1)

            # Displaying image
            cv2.imshow('Input', img)

            # Key control
            c = cv2.waitKey(1)
            if c != -1: # Esc: Exit
                break
                

        # Getting img shape
        cap.release()
        cv2.destroyAllWindows()

        # Writing the lists
        np.savez('{:s}//Images_MetaData.npz'.format(self.path), 
                 time=timepoints, 
                 obj=objpoints, 
                 img=imgpoints, 
                 imgshape=np.array(img.shape[1::-1]))
        
        
    def tune(self):
        """Tuning Camera Calibration Signature
    
        Based on the Image Dataset and their related Metadata, 
        creates a Camera Calibration Signature saved under CameraSignature.npz
        Finally print the camera error (value between 0 to 1),
        assuming the camera signature is correct.
        
        Args:
            /
        
        Returns:
            /"""        
        
        # Updating the lists based on the images left in the Calibration Folder
        paths_chessboard = glob('{:s}//Image*.jpg'.format(self.path))

        # Only reading the file if required
        npz = np.load('{:s}//Images_MetaData.npz'.format(self.path))

        boolean = np.isin(npz["time"], [int(path_chessboard[-14:-4]) for path_chessboard in paths_chessboard])

        # Actual udate
        objpoints = npz["obj"][boolean]
        imgpoints = npz["img"][boolean]
        imgshape = tuple(npz["imgshape"])

        # Calculating the calibration parameters
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape, None, None)

        # New image matrix ROI (in order to drop black pixel due tu undistrotion)
        newcameramtx, self.roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,imgshape,1,imgshape)

        # undistorttion map
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,newcameramtx,imgshape,5)

        # Writin Job
        np.savez('{:s}//CameraSignature.npz'.format(self.path),
                 mapx=self.mapx, 
                 mapy=self.mapy, 
                 mtx=self.mtx, 
                 dist=self.dist,
                 roi=self.roi
                )

        # Error Calculation
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.mtx, self.dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        self.error = error

        print("Calibration Error: ", round(mean_error/len(objpoints), 4))
        
        
    def retune(self):
        """Re-Tuning Camera Calibration Signature
    
        First, helps the user to delete potentially problematic image,
        usually where the chessboard isn't being detected correctly,
        by displaying every image from the Image Database in a simplistic interface.
        
        Then re-tune the Camera Calibration Signature with the function tune() above. 
        
        Args:
            /
        
        Returns:
            /"""        
        
        # Going through the image folder 
        i = 0    
        while True:

            # Need to be updated in case deleted images
            paths_chessboard = glob('{:s}//Image*.jpg'.format(self.path))
            bondaries = [0, len(paths_chessboard)-1]   

            # In Case Image Index goes too high/low
            if i < bondaries[0]:
                i = bondaries[1]
            elif i > bondaries[1]:
                i = bondaries[0]

            # Loading Image
            path_chessboard = paths_chessboard[i]
            img = cv2.imread(path_chessboard)

            # Text instruction
            cv2.putText(img=img,
                        text='Keys: Left, Right, Delete - Any Other: Exit', 
                        org=(0, img.shape[0]-5), 
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.5,
                        color=(0,0,0),
                        lineType=1)
            # Show    
            cv2.imshow('Input', img)

            # Key Controls
            c = cv2.waitKeyEx(0)
            if c == 2555904: # Go to next image
                i+=1
            elif c == 2424832: # Go to previous image
                i-=1
            elif c == 3014656: # Delete current image & go to next image
                os.remove(path_chessboard)
                print(path_chessboard, "- Deleted")
                i+=1
            else:
                break

        cv2.destroyAllWindows()
                
        # Retune        
        self.tune()
        
        
    def calibrate(self, img, crop=True):
        """Calibrate an image
    
        Calibrates an image based on the Camera Calibration Signature.

        Args:
            img (2D np.array): Image that needs calibration
            crop (boolean): If True, proceed to delete image borders deformed by the calibration process
        
        Returns:
            img (2D np.array): Image calibrated""" 
                
        # Undistort
        img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # Crop the image
        if crop:
            x,y,w,h = self.roi
            img = img[y:y+h, x:x+w]

        return(img)
    
        
    def render(self,
              fps=60,
              resize_ratio=0,
              crop=True):
        """Rendering a 3d object on a video flux
    
        Based on the video flux and the Camera Calibration Signature, 
        renders a 3d object (per default a cube) on the chessboard.  
        
        Args:
            /
        
        Returns:
            /"""     

        def cube(img, corners, imgpts):
            imgpts = np.int32(imgpts).reshape(-1,2)

            # draw ground floor in green
            img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

            # draw top layer in red color
            img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

            return img
        
        # Position of the cube on the chessboard
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
               [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

        # Launch Camera
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        # Initialize parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001) #???
        objp = np.zeros((self.chessboard[0]*self.chessboard[1],3), np.float32) #???
        objp[:,:2] = np.mgrid[0:self.chessboard[1],0:self.chessboard[0]].T.reshape(-1,2) #???  

        # Launch
        proc = datetime.now()
        while True:

            # Applying fps
            time_elapsed = (datetime.now() - proc).total_seconds()

            if True:#time_elapsed > 1./fps:
                proc = datetime.now()

                # Reading the images
                ret, img = cap.read()

                # Image Processing
                if self.mirror: 
                    img = cv2.flip(img, 1)
                if resize_ratio:    
                    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)

                # Calibration
                img = self.calibrate(img, crop=crop)

                # Chessboard Detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # enhancing chessboard
                gray = cv2.bilateralFilter(gray, 3, 21, 21) 

                ret, corners = cv2.findChessboardCorners(image=gray, 
                                                         patternSize=(self.chessboard[1],self.chessboard[0]), 
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

                if ret == True:
                    # Refines corner detection
                    corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                    # Find the rotation and translation vectors.          
                    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, self.mtx, self.dist)

                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)

                    img = cube(img,corners,imgpts)

                # Displaying
                cv2.imshow('Input', img)

                # Key control
                c = cv2.waitKey(1)
                if c == 32: # Space: Pause
                    c = cv2.waitKey(0)
                    if c != 32: # Space: Un-Pause
                        break
                elif c != -1:
                    break

        cap.release()
        cv2.destroyAllWindows()
