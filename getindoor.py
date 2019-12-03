#!/usr/bin/env python
# coding:utf-8
import sys 
import rospy 
from std_msgs.msg import String 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String,Int16
import math
class indoor: 
    def __init__(self): 
        self.sub=rospy.Subscriber('/start',String,self.callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)
        self.sub2=rospy.Subscriber('/odom',Odometry,self.get_x)
	self.pub2 = rospy.Publisher("/go", Int16, queue_size = 10)
        self.x = 0
    def callback(self, data):
        if(data.data != "ok"):
            return
        rate = rospy.Rate(10)
        while(math.fabs(self.x-1) > 0.02):
            if(math.fabs(self.x-1) > 0.95):
                speed = 0.05
            elif(math.fabs(self.x-1) > 0.5):
                speed = 0.25
            else:
                speed = 0.15
            if(self.x > 1):
                speed = -speed
            twist_msg = Twist()
            twist_msg.linear.x = speed
            self.pub.publish(twist_msg)
            print(speed)
            rate.sleep()
        self.pub2.publish(1)
    def get_x(self, odom_msg):
        self.x = odom_msg.pose.pose.position.x
        #print(self.x)
            
if __name__ == '__main__': 
    rospy.init_node('indoor', anonymous=False) 
    lidar = indoor() 
    rospy.spin()
