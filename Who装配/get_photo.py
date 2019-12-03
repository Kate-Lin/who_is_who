#! /usr/bin/env python
#coded by Lin Shen, 20190517
import freenect
import cv2


key=cv2.waitKey(10)
while key !=ord('q'):
    key=cv2.waitKey(10)
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imshow('FaceDetecter', array)
    cv2.imwrite('/home/ros/robocup/src/beginner_tutorials/launch/member/'+ str(i)+'.jpg',array)
