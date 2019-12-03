#!/usr/bin/env python
# coding:utf-8
import sys 
import rospy 
from sensor_msgs.msg import LaserScan 
from std_msgs.msg import String 
flag = False #只发送一次开门信号
cnt = 0
class DoFilter: 
    def __init__(self): 
        self.sub = rospy.Subscriber("scan", LaserScan, self.callback) 
        self.pub = rospy.Publisher("start", String,queue_size=10)
    def callback(self, data):
        xx = list(data.ranges)
        global flag
        if((xx[340] != 0 and xx[340] > 1)and flag == False):#一段时间没检测到雷达或者检测到的距离小于1
            Ok = String()
            Ok.data = "ok"
            print(Ok.data)
            self.pub.publish(Ok)
            flag = True
            
if __name__ == '__main__': 
    rospy.init_node('is_start', anonymous=False) 
    lidar = DoFilter() 
    rospy.spin()