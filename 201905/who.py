#! /usr/bin/env python
#coded by Lin Shen, 20190428
#####################################################################################
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

class TalkBack:
	def __init__(self, script_path):
		shutil.rmtree('/home/ros/robocup/src/beginner_tutorials/launch/dataset')
		shutil.rmtree('/home/ros/robocup/src/beginner_tutorials/launch/person')
		shutil.rmtree('/home/ros/robocup/src/beginner_tutorials/launch/recognizer')
		os.mkdir('/home/ros/robocup/src/beginner_tutorials/launch/dataset')
		os.mkdir('/home/ros/robocup/src/beginner_tutorials/launch/person')
		rospy.loginfo("In the __init__()")
		#initialize the node.
		rospy.init_node('who is who')
		rospy.on_shutdown(self.cleanup)
		#set the voice
		self.voice = rospy.get_param("~Voice", "voice_don_diphone")
		self.wavepath = rospy.get_param("~wavepath", script_path + "/../sounds")
		self.soundhandle = SoundClient()
		self.exit_point = PoseStamped()
		rospy.sleep(1)
		self.soundhandle.stopAll()
		rospy.sleep(3)
		#send the sign of ready.
		self.soundhandle.say("I am ready", self.voice)
		rospy.loginfo("Say one of the navigation commands...")
		#All subscribers and publishers are defined here
		rospy.Subscriber('/recognizer/output', String, self.talkback)
		self.sub_begin=rospy.Subscriber("/go",Int16,self.callback_begin)
		self.voice_vel_pub = rospy.Publisher('who_voice', String , queue_size=5)
		self.point_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=5)
		self.peopleposepub = rospy.Publisher("people_pose_info",PeoplePose)
		#the word dictionary
		#need to edit
		self.keywords_to_command = {
			'water':['water'],
			'coffee':['coffee'],
			'red bull':['red bull','red','bull'],
			'cola':['cola'],
			'paper':['paper'],
			'michael':['michael'],
			'jack':['jack'],
			'fisher':['fisher'],
			'kevin':['kevin'],
			'daniel':['daniel'],
			'yes':['yes'],
			'no':['no']
		}
		self.srcfile = '/home/ros/robocup/src/beginner_tutorials/launch/WhoIsWho_yaml/trainningdata.yml'
		self.dstfile = '/home/ros/robocup/src/beginner_tutorials/launch/recognizer/trainningdata.yml'
		self.copyfile()
		self.name = [[0 for i in range(2)] for i in range(6)]
		self.position = [[0 for i in range(3)] for i in range(6)]

		self.flag_begin = 0
		self.people_num = 0
		self.tempname = ''
		self.talk_flag = 0
		self.frame = None
		self.array = None
		self.face_rects = None
		self.face_rects_judge = None
		self.scaling_factor = 1.0
		self.num=0
		self.i = 0
		self.a = 0
		self.end = 450
		self.flag_face_detect = 0
		self.recognize = 0

		self.recognizer = cv2.createLBPHFaceRecognizer()
		self.face_cascade=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
		self.begin=time.time()
		
		self.quaternion_get_into_the_door = [0,0,0.016557,0,999863]
		self.quaternion_get_into_the_room=[0,0,-0.134185,0.999995]
		self.pos_get_into_the_door = [11.18065,5.967353,0]
		self.pos_get_into_the_room=[12.8934,3.3162,0]
		self.euler_point=[11.9269,3.2924,0]
		self.quaternion_go_to_the_exit = [0,0,0.999756,-0.022053]#gai
		self.pos_go_to_the_exit= [10.616634,0.949925,0]#gai
		
		#initialize move_base
		self.move_base_client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
		connected_befor_timeout = self.move_base_client.wait_for_server(rospy.Duration(2.0))
		if connected_befor_timeout:
			rospy.loginfo('succeeded connecting to move_base server')
		else:
			rospy.logerr('failed connecting to move_base server')
			return
		rospy.wait_for_service('move_base/clear_costmaps',5.0)
		self.service_clear_costmap_client = rospy.ServiceProxy('move_base/clear_costmaps',Empty)
		rospy.loginfo('connected to move_base/clear_costmaps')
		rospy.loginfo('node initialized')

	def copyfile(self):
		if not os.path.isfile(self.srcfile):
			print"%s not exist!" %(self.srcfile)
		else:
			fpath,fname = os.path.split(self.dstfile)
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		shutil.copyfile(self.srcfile,self.dstfile)
		print "copy %s -> %s"%(self.srcfile,self.dstfile)

	#send sign of begin. defined as the callback funtion in line 55
	def callback_begin(self, msg):
		if(msg.data == 1 and self.flag_begin == 0):
			self.flag_begin = 1
			self.begin = time.time()
			self.service_clear_costmap_client()
			rospy.loginfo("begin")
			self.soundhandle.say("begin")
			rospy.sleep(10)
			self.get_into_the_door()

	#action of robot to get into the door(the 1st step of the test)
	def get_into_the_door(self):
		self.sl('I will get into the door.',1,2)
		rot=[self.quaternion_get_into_the_door[0],self.quaternion_get_into_the_door[1],self.quaternion_get_into_the_door[2],self.quaternion_get_into_the_door[3]]
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'map'
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = self.pos_get_into_the_door[0]
		goal.target_pose.pose.position.y = self.pos_get_into_the_door[1]
		goal.target_pose.pose.position.z = self.pos_get_into_the_door[2]
		goal.target_pose.pose.orientation.x = rot[0]
		goal.target_pose.pose.orientation.y = rot[1]
		goal.target_pose.pose.orientation.z = rot[2]
		goal.target_pose.pose.orientation.w = rot[3]
		self.move_base_client.send_goal(goal,done_cb=self.move_base_done_cb_door)
	
	#actions done after getting into the door:self-intro and start to remember person(the 2nd step)
	def move_base_done_cb_door(self, state, result):
		rospy.sleep(2)
		self.sl("Hello, nice to meet you. I come from Shanghai University. My name is Michael.",1,2)
		rospy.sleep(6)
		self.sl("Now I am going to remember the person.",1,2)
		rospy.sleep(5)
		self.sl("What is your name",1,2)
		self.talk_flag = 1
		#self.talkback()

	

	#speech interactions between robot and guests
	#procedure:"What is your name?" "(say name)" "Are you (name)?"
	#if 'yes':take picture and remember
	#if 'no':ask his/her name again.
	def talkback(self,msg):
		command = ''
		command = self.get_command(msg.data)
		#print command
		if(self.people_num < 1 and self.talk_flag == 1):
			print self.people_num
			for (commands, keywords) in self.keywords_to_command.iteritems():
				for word in keywords:
					if(command == word and command != 'yes' and command != 'no'):
						self.tempname = command
						self.sl("Are you " + command + "?",1,2)
						break
					elif(command == 'yes'):
						if(self.tempname != ''):
							print self.tempname
							self.name[self.people_num][0] = self.tempname
							self.sl(self.tempname + ", please look at the camera.",1,2)
							self.tempname = ''
							rospy.sleep(2)
							self.face_catch()
							self.sl("I remember!",1,2)
							self.people_num += 1
							command = ''
							if(self.people_num == 1):
								self.talk_flag = 0
								self.get_into_the_room()
							else:
								self.sl("What is your name?",1,2)
								self.talk_flag = 1
								break
						else:
							pass
					elif(command == 'no'):
						self.tempname = ''
						self.sl("Sorry, please say again.",1,2)
						rospy.sleep(2)
						self.sl("What is your name?",1,2)
						self.talk_flag = 1
#####################################################
						command = ''
						break
					else:
						pass



	#after remembering the name of the person, camera will take pictures of the person and save as anchor image to compare.
	def face_catch(self):
		self.time_check()
		begin_face_catch = time.time()
		num = 1
		pic_num = 0
		while True:
			pic_num += 1
			if(pic_num % 10 == 0):
				print pic_num / 10
				self.soundhandle.say(str(pic_num/10))
			#face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
			#scaling_factor = 1.0
			self.frame = self.get_video()
			gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			self.face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
			for(x,y,w,h) in self.face_rects:
				key=cv2.waitKey(10)&0xff
				#cut the face of the person
				face=self.frame[y:y+h-w*0.07,x+w*0.15:x+w-w*0.15]
				cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/dataset/user."+str(self.people_num)+"."+str(num)+".jpg",face)
				num=num+1
			#show instant image
			cv2.imshow('FaceDetecter', self.frame)
			if((time.time()-begin_face_catch > 30 or num >= 50) and num >= 20):
				rospy.loginfo('Have caught your face')
				cv2.destroyAllWindows()
				#save images as training data
				path = '/home/ros/robocup/src/beginner_tutorials/launch/dataset'
				Ids, faces = self.getIMagesWithID(path)
				self.recognizer.train(faces, np.array(Ids))
				self.recognizer.save('/home/ros/robocup/src/beginner_tutorials/launch/recognizer/trainningdata.yml')
				cv2.destroyAllWindows()
				break
			elif(time.time()-begin_face_catch > 30 and num < 20):
				rospy.loginfo('not enough sample.')
				rospy.loginfo('Have caught your face')
				cv2.destroyAllWindows()
				#the same as above
				path = '/home/ros/robocup/src/beginner_tutorials/launch/dataset'
				Ids, faces = self.getIMagesWithID(path)
				self.recognizer.train(faces, np.array(Ids))
				self.recognizer.save('/home/ros/robocup/src/beginner_tutorials/launch/recognizer/trainningdata.yml')
				cv2.destroyAllWindows()				
				break

	
	#action of robot to get into the room after remeber three guests
	def get_into_the_room(self):
		#self.recognize_member()
		self.sl('I will get into the room.',1,2)
		rot=[self.quaternion_get_into_the_room[0],self.quaternion_get_into_the_room[1],self.quaternion_get_into_the_room[2],self.quaternion_get_into_the_room[3]]
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'map'
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = self.pos_get_into_the_room[0]
		goal.target_pose.pose.position.y = self.pos_get_into_the_room[1]
		goal.target_pose.pose.position.z = self.pos_get_into_the_room[2]
		goal.target_pose.pose.orientation.x = rot[0]
		goal.target_pose.pose.orientation.y = rot[1]
		goal.target_pose.pose.orientation.z = rot[2]
		goal.target_pose.pose.orientation.w = rot[3]
		self.move_base_client.send_goal(goal,done_cb=self.move_base_done_cb_room)

	#actions done after getting into the room
	def move_base_done_cb_room(self,state,result):
		rospy.sleep(2)
		self.sl('I have got into the room', 1,2)
		rospy.sleep(2)
		self.findperson()

	#to count how many people are there in the room in sum
	#if the number detected is smaller than target, it will output 'not enough quantity' until time uses up.
	def findperson(self):
		self.time_check()#TIME_CHECK
		begin_findperson=time.time()
		duration_findperson=0.0
		self.sl("l will find the person",1,2)
		rospy.loginfo("I will find the person")
		num=0
		depth=0
		face_num_max=0
		while (time.time()-begin_findperson<60.0):
			#=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
			#scaling_factor =1.0
			frame = self.get_video()
			depth_array=np.transpose(self.get_depth())
			frame_judge=cv2.resize(frame,None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
			gray_judge=cv2.cvtColor(frame_judge,cv2.COLOR_BGR2GRAY)
			self.face_rects_judge=self.face_cascade.detectMultiScale(gray_judge,1.3,5)
			#get person's face image and save.
			if(len(self.face_rects_judge)>=face_num_max):
				face_num_max=len(self.face_rects_judge)
				key=cv2.waitKey(10)&0xff
				cv2.imshow('FaceDetecter', frame)
				cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/photo/user."+"find_person"+".jpg",frame)
				cv2.destroyAllWindows()
				self.frame=cv2.resize(frame,None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
				gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
				self.face_rects=self.face_cascade.detectMultiScale(gray,1.3,5)

			#quantity
			if(len(self.face_rects)<2  and time.time()-begin_findperson<40.0):#renshu
				key=cv2.waitKey(10)&0xff
				cv2.imshow('FaceDetecter', frame)
				rospy.loginfo("face quantity:")
				rospy.loginfo(len(self.face_rects))
				self.sl('Not enough quantity.',1,2)

				rospy.sleep(2)
				continue

			cv2.destroyAllWindows()
			
			#use tf-transform and the depth-image of the crowd to get the position of each person
			for(x,y,w,h) in self.face_rects:
				cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),1)
				x0=x
				y0=y
				#rospy.loginfo("x:")
				#print x
				#rospy.loginfo("y:")
				#print y
				#rospy.loginfo("w:")
				#print w
				#rospy.loginfo("h:")
				#print h
				#flag_findperson=0
				if(x+w>=640 or y+h>=480):
					continue
				#flag_break=0

				min_dep=10
				i_min=x0
				j_min=y0
				for i in range(int(w)):
					#print("i=",i)
					#print("x=",x0+i)
					for j in range(int(h)):	
						#print("j=",j)
						#print("y=",y0+j)
						depth=1.0 / (depth_array[x0+i][y0+j] * -0.0030711016 + 3.3309495161)
						#print depth
						if(depth_array[x0+i][y0+j]<2047 and depth>0 and depth<min_dep):
							min_dep=depth
							#i_min=i
							#j_min=j
							#print "changed"
				depth=min_dep-0.45
				line=320-(x+w*0.5)
				#angle=8.13587-0.03162*line+0.000975239*line*line
				angle=np.arctan((np.tan(3.1415926*28.75/180)/320)*line)
				a=math.sin(angle)
				b=math.cos(angle)
				x_len=depth*a+0.45*math.sin(angle)
				#self.position[num][0]=x_len
				#self.position[num][1]=depth*b
				self.tflistener = tf.TransformListener()
				self.tflistener.waitForTransform('map','camera_link',rospy.Time(),rospy.Duration(4.0))
				(trans1,rot1) = self.tflistener.lookupTransform('map','camera_link',rospy.Time(0))
				print "map to camera_link: "
				print trans1,rot1
				self.tfbroadcaster = tf.TransformBroadcaster()
				self.tfbroadcaster.sendTransform((x_len,depth*b,0.0),(0.0,0.0,0.0,1.0),rospy.Time.now(),"camera_link","people")
				self.peopleposepub.publish(PeoplePose(x_len,depth*b))
				rospy.sleep(1)
				(trans2,rot2) = self.tflistener.lookupTransform('camera_link','people',rospy.Time(0))
				print "camera_link to people: "
				print trans2,rot2
				self.tflistener.waitForTransform('map','people',rospy.Time(),rospy.Duration(4.0))
				try:
					(trans,rot) = self.tflistener.lookupTransform('map','people',rospy.Time(0))
				except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
					rospy.logerr('Error occure when transfrom.')
				self.position[num][0]=trans[0]
				self.position[num][1]=trans[1]
				#self.peopleposepub.publish(2.0,-1.0)
				self.num=num
				num=num+1
				#print self.position
				#rospy.loginfo("angle:")
				#print angle
				#rospy.loginfo("x_len:")
				#print x_len
				#rospy.loginfo("depth")
				#print depth
				#print x0,y0,x0+i_min,y0+j_min

				cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/photo/user."+"find_person"+".jpg",self.frame)
			
				#self.pub_ObjRecSig.publish(1)
				#rospy.loginfo("already run command")

				rospy.loginfo("l have found the person")
				self.soundhandle.say("l have found the person")
			if(self.position[0][0]!=0 or self.position[0][1]!=0):
				break
		rospy.sleep(3)
		if((self.position[0][0]!=0 or self.position[0][1] != 0)):
			print self.num
			self.go_to_the_point1(self.position[self.num][0],self.position[self.num][1])
		else:
			if(not self.time_check()):
				rospy.loginfo("Overtime!")
				self.soundhandle.say("Overtime   Overtime   Overtime.")
				rospy.sleep(2)
				self.go_to_the_exit()
			else:
				rospy.loginfo("I cannot find any person! I will try again!")
				self.soundhandle.say("I cannot find any person! I will try again!")
				rospy.sleep(4)
				self.findperson()

	#go to the point in front of the target person
	def go_to_the_point1(self,coord_x,coord_y):
		self.time_check()#TIME_CHECK
		rospy.loginfo("I will go to the point.")
		self.soundhandle.say("i will go to the point")
		#rot = tf.transformations.quaternion_from_euler(self.euler_get_into_the_door[0],self.euler_get_into_the_door[1],self.euler_get_into_the_door[2])
		rot = tf.transformations.quaternion_from_euler(self.euler_point[0],self.euler_point[1],self.euler_point[2])
		rot=[self.quaternion_get_into_the_room[0],self.quaternion_get_into_the_room[1],self.quaternion_get_into_the_room[2],self.quaternion_get_into_the_room[3]]
		#start move	
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'map'
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = coord_x
		goal.target_pose.pose.position.y = coord_y
		goal.target_pose.pose.position.z = 0
		goal.target_pose.pose.orientation.x = rot[0]
		goal.target_pose.pose.orientation.y = rot[1]
		goal.target_pose.pose.orientation.z = rot[2]
		goal.target_pose.pose.orientation.w = rot[3]
		#rospy.loginfo("get")
		#debug
		self.move_base_client.send_goal(goal,done_cb=self.move_base_done_cb_person)
		print "Have Sended Goal to the move_base."
		#debug
		print "target pose: "
		print coord_x,coord_y

	#actions done after reach the point in front of person	
	def move_base_done_cb_person(self,state, result):
		#target point success
		#then start kinect  .......
		rospy.loginfo("I have got the point")
		self.soundhandle.say('i have got the point')
		rospy.sleep(3)
		self.flag_face_detect=0
		self.face_detect()
	
	def detect_member(self, frame):
		recognizer = cv2.createLBPHFaceRecognizer()
		recognizer.load("/home/ros/robocup/src/beginner_tutorials/launch/train_member/member.yml")
		#img = cv2.imread('/home/ros/robocup/src/beginner_tutorials/launch/member/18.jpg')
		#img1 = cv2.resize(img, None,fx=1.0,fy=1.0,interpolation=cv2.INTER_AREA)
		#gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		self.frame = cv2.resize(frame, None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray,1.3,5)
		if len(faces) > 0:
			for(x,y,w,h) in faces:
				id,conf = recognizer.predict(gray[y:y+h-w*0.07,x+w*0.15:x+w-w*0.15])
				print conf
				#cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),1)
				cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),1)
				cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/person/user."+str(self.i)+".jpg",self.frame)
				#cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/person/user."+str(self.i)+".jpg",img)
				self.i += 1
				print conf
				rospy.loginfo(conf)
				if(conf < 60):
					return 1
				else:
					return None
		else:
			return -1
		
	#recognize particular person who is in front of the robot
	def face_detect(self):
		self.time_check()
		self.sl("I will recognize the person",1,2)
		rospy.sleep(2)
		self.sl("Please look at the camera, I will recognize you now.", 1,2)
		rospy.sleep(3)
		if(self.flag_face_detect == 0):
			self.recognize = 0
		#loading training mode
		#self.recognizer.load("/home/ros/robocup/src/beginner_tutorials/launch/train_member/member.yml")
		frame = self.get_video()
		result = self.detect_member(frame)
		print result
		if(result == None):
			self.recognizer.load("/home/ros/robocup/src/beginner_tutorials/launch/recognizer/trainningdata.yml")
			#scaling_factor = 1.0
			self.frame = cv2.resize(frame, None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = self.face_cascade.detectMultiScale(gray,1.3,5)
			#get faces
			if len(faces) > 0:
				for (x,y,w,h) in faces:
					#use LBPH Face Recognizer to perdict which person he/she is
					id,conf = self.recognizer.predict(gray[y:y+h-w*0.07,x+w*0.15:x+w-w*0.15])
					cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),1)
					cv2.imwrite("/home/ros/robocup/src/beginner_tutorials/launch/person/user."+str(self.i)+".jpg",self.frame)
					self.i += 1
					#output the result of prediction
					if (id >= 0 and id <= 3):
						rospy.loginfo(id)
						if(conf < 55):
							s = 'Hello,' + self.name[id][0]
							self.a = 2
						else:
							s = 'Sorry, I can not recognize you.'
							self.a += 1
					rospy.loginfo(conf)
					self.sl(s,1,2)
					rospy.sleep(3)
			
			else:
				self.sl("sorry,i can not find person",1,2)
				rospy.sleep(3)
				self.a += 1

		elif (result == 1):
			self.sl('Hello team member.',1,2)
			self.a = 2
			#rospy.loginfo(conf)
			#self.sl(s,1,2)
			rospy.sleep(3)

		else:
			self.sl("sorry,i can not find person",1,2)
			rospy.sleep(3)
			self.a += 1


		if(self.a >= 2):
			self.num -= 1
			print self.num
			if(self.position[self.num][0] != 0 and self.num >=0):
				self.go_back_recognize()
			else:
				self.go_to_the_exit()
		else:
			self.flag_face_detect = 1
			self.face_detect()
		
		
	
	#after recognizing a person, robot will return a specific point to restart recognition again.
	#since while robot is in a crowd, there may be a strong disturbance to navigation module.
	def go_back_recognize(self):
		self.time_check()
		self.sl('I will go back',1,2)
		rot=[self.quaternion_get_into_the_room[0],self.quaternion_get_into_the_room[1],self.quaternion_get_into_the_room[2],self.quaternion_get_into_the_room[3]]
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'map'
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = self.pos_get_into_the_room[0]
		goal.target_pose.pose.position.y = self.pos_get_into_the_room[1]
		goal.target_pose.pose.position.z = self.pos_get_into_the_room[2]
		goal.target_pose.pose.orientation.x = rot[0]
		goal.target_pose.pose.orientation.y = rot[1]
		goal.target_pose.pose.orientation.z = rot[2]
		goal.target_pose.pose.orientation.w = rot[3]
		self.move_base_client.send_goal(goal,done_cb=self.move_base_done_cb_go_back_recognize)
	
	def move_base_done_cb_go_back_recognize(self,state,result):
		self.time_check()
		self.sl('I have gone back.',1,2)
		self.go_to_the_point1(self.position[self.num][0],self.position[self.num][1])
	
	def go_to_the_exit(self):
		self.sl('I will exit.',1,2)
		rot=[self.quaternion_go_to_the_exit[0],self.quaternion_go_to_the_exit[1],self.quaternion_go_to_the_exit[2],self.quaternion_go_to_the_exit[3]]
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'map'
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = self.pos_go_to_the_exit[0]#gai
		goal.target_pose.pose.position.y = self.pos_go_to_the_exit[1]#gai
		goal.target_pose.pose.position.z = self.pos_go_to_the_exit[2]#gai
		goal.target_pose.pose.orientation.x = rot[0]
		goal.target_pose.pose.orientation.y = rot[1]
		goal.target_pose.pose.orientation.z = rot[2]
		goal.target_pose.pose.orientation.w = rot[3]
		self.move_base_client.send_goal(goal,done_cb=self.move_base_done_cb_go_to_the_exit)
	
	def move_base_done_cb_go_to_the_exit(self, state, result):
		self.sl('game over',1,2)
		print "Duration:"
		print time.time()-self.begin
	
	#used before actions start, to check if there is enough time to get out of the door
	def time_check(self):
		if(time.time()-self.begin>=self.end):
			s='overtime!  I will go to the exit now!'
			rospy.loginfo(s)
			self.soundhandle.say(s)
			rospy.sleep(3)
			self.soundhandle.stopAll()
			self.go_to_the_exit()
			while(1):
				continue
			return 0
		else:
			return 1	
	
	def get_depth(self):
		array,_ = freenect.sync_get_depth()
		return array
	
	#speak_log
	def sl(self, txt, before, after):
		rospy.sleep(before)
		self.voice = rospy.get_param("~voice", "voice_don_diphone")
		self.soundhandle.say(txt, self.voice)
		rospy.loginfo(txt)
		rospy.sleep(after)

	def get_command(self,data):
		for(command, keywords) in self.keywords_to_command.iteritems():
			for word in keywords:
				if data.find(word) > -1:
					return command

	def cleanup(self):
		self.soundhandle.stopAll()
		rospy.loginfo("Shutting down talkback node...")
	
	def get_video(self):
		self.array,_ = freenect.sync_get_video()
		self.array = cv2.cvtColor(self.array, cv2.COLOR_RGB2BGR)
		return self.array

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

if __name__ == "__main__":
	try:
		TalkBack(sys.path[0])
		rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Talkback node terminated.")
