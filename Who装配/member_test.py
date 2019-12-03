#! /usr/bin/env python
#coded by Lin Shen, 20190513
#########################################
#import modules needed
import os
import rospy
from  geometry_msgs.msg import PoseStamped
from std_msgs.msg import String,Int16
import tf
import tf2_ros
from sound_play.libsoundplay import SoundClient
import sys
import freenect
import cv2
import numpy as np
from PIL import Image
import time
import math
from std_srvs.srv import Empty
import actionlib
import actionlib_msgs.msg
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseResult
from geometry_msgs.msg import Pose, Point, Quaternion
from beginner_tutorials.msg import  PeoplePose
import shutil
from cv2 import cv as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Remember_Member:
	def __init__(self):
		self.path = '/home/ros/robocup/src/beginner_tutorials/launch/member'
		rospy.loginfo("In the __init__()")
		self.recognizer = cv2.createLBPHFaceRecognizer()
		self.face_cascade=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
		self.frame = ''
		self.face_rects = ''
		self.load_path = '/home/ros/robocup/src/beginner_tutorials/launch/train_member/member.yml'
		shutil.copyfile('/home/ros/robocup/src/beginner_tutorials/launch/WhoIsWho_yaml/trainningdata.yml',self.load_path)
		self.train()
		self.recognize()
		

	def getIMagesWithID(self, path):
		imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
		faces = []
		IDs = []
		for imagePath in imagePaths:
			faceImg = Image.open(imagePath).convert('L')
			faceNp = np.array(faceImg, 'uint8')
			ID = int(os.path.split(imagePath)[-1].split('.')[1])
			faces.append(faceNp)
			IDs.append(ID)
			cv2.waitKey(10)
		return IDs, faces

	def train(self):
		for i in range(1,101):
			img_path = self.path + "/" + str(i) + ".jpg"
			print img_path
			self.frame = cv2.imread(img_path)
			#cv2.imshow('face',self.frame)
			gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
 			self.face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)  
			for (x,y,w,h) in self.face_rects:
				face = self.frame[y:y+h-w*0.07,x+w*0.15:x+w-w*0.15]
				cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/member_faces/user."+str(i)+".jpg",face)
			#cv2.imshow('FaceDetector',self.frame)
			path = '/home/ros/robocup/src/beginner_tutorials/launch/member_faces'
			Ids, faces = self.getIMagesWithID(path)
			self.recognizer.train(faces,np.array(Ids))
			self.recognizer.save(self.load_path)
			cv2.destroyAllWindows()

	def recognize(self):
		self.recognizer.load(self.load_path)
		frame = cv2.imread(self.path+'/18.jpg')
		self.frame = cv2.resize(frame, None,fx=1.0,fy=1.0,interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray,1.3,5)
		for (x,y,w,h) in faces:
			ids,conf = self.recognizer.predict(gray[y:y+h-w*0.07,x+w*0.15:x+w-w*0.15])
			cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),1)
			print ids,conf


if __name__ == "__main__":
	Remember_Member()