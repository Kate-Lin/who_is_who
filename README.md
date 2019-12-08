# who_is_who
Source code for WhoIsWho during 2018~2019

## Part 1
项目调试流程归纳&理清代码函数逻辑。
见示意图。
 

### 【完成60%代码撰写】，写完的函数有：
copyfile()<br>call_back_begin()<br>get_into_the_door()<br>move_base_done_cb_door()<br>talkback()<br>
get_into_the_room()<br>move_base_done_cb_room()<br>findperson()<br>

即比赛第一阶段的任务撰写完成。<br>
其中语音交互的talkback函数进行了大规模的修改和简化，导航函数也进行了大规模简化。

### 【仍需要完成的函数代码】有：
完成face_catch()<br>
face_detect<br>
go_to_the_point1<br>
move_base_done_cb_person()<br>
go_back_recognize()<br>
move_base_done_cb_go_back_recognize()<br>
go_to_the_exit()<br>
move_base_cb_done_exit()<br>
即比赛第二阶段的任务。


## Part 2
【完成所有代码的撰写】，成功进行仿真demo调试，按照得分点比对，基本上可以完成所有的任务。<br>

完成代码重构之后，给整组下发了新版代码用于五一调试，并撰写了详细的代码问题总结，操作手册和调试注意事项。

***2019.4版whoiswho的代码相较2018.4的代码，在不影响机器人功能的情况下，进行了全方位的算法简化，代码缩短到了600行以内。2018年8月的whoiswho代码，由于大赛的规则严格，任务复杂，曾经一度被改写到了800多行，这次的代码重写和重构是对代码的一次大的优化。***

## 【调试过程中遇到的问题】：

* 在运行face_catch()函数时，<br>
face = self.frame[y:y+h-w*0.07, x+w*0.05:x+w-w*0.15]行报错，错误为切片下标必须为整数或None。
*原因：电脑安装的numpy模块版本过高，矩阵索引传入浮点数
解决办法：下载低版本的numpy*
```
sudo -H pip install -U numpy=1.11.0
```

* 语音交互阶段说“no”进入死循环：<br>
What is your name?<br>
Sorry please say again.<br>
虽然在说话无错误的情况下不影响比赛，但一旦说错就会卡死。（已解决）

* 上届代码遗留的问题：<br>
进入房间走到人前认人的face_detect函数中调用了go_to_the_point1用于根据人的位置走早人前。其中euler_position数组含义不明。（已解决）

### 【除了修改上述bug之外仍可以改进的部分】：
* 修改字典，剔除不需要的command
* 各项防超时的限制完善。
* 实际比赛机器人站在房间门口认人，若人群分散，受摄像头视角的限制，则可能拍不全五个人。如何避免/减少损失？
* 提高认人准确度。
* 减少高度对人识别的损失。

## Part 3
针对昨天所述的所有问题都进行了解决，并且都成功解决。项目目前没有导致比赛直接结束的缺陷。
向整组组员讲解项目代码，每个项目组成员都完成了仿真模拟的全过程。


## 【调试细则和注意事项】

### 【前提】
全组都已经装配完成家庭组代码包大全集（800多M的那个）

### 【如何进行WhoIsWho的代码装配？】
打开总安装包 Strive@home2018XXX文件夹->最新项目代码合集->who is who文件夹内
有装配手册和仿真的操作手册。
把修改后的who_new（注释版）.py重命名为who.py，右键属性修改文件权限为读写&可执行后，放到/home/ros/robocup/beginner_tutorials/nodes文件夹内。
操作步骤按照手册内的跑就可以了。

### 【注意】
摄像头的操作在整个文件的最上面的第0步，roslaunch freenect看到……stream flush后需Ctrl-C关闭该launch。
修改后的代码需要将缩进切换至制表符长度：8，切换位置在gedit或者vscode右下角。

### 【运行期间可能会碰到的问题】
在运行face_catch()函数时，face = self.frame[y:y+h-w*0.07, x+w*0.05:x+w-w*0.15]行报错，错误为切片下标必须为整数或None。
原因：电脑安装的numpy模块版本过高，矩阵索引传入浮点数
解决办法：下载低版本的numpy
```
sudo -H pip install -U numpy=1.11.0
（待补充）
```

## 【五一期间实验室调试须知】
* 每个人都要跑完一遍仿真demo，根据注释看懂代码逻辑。
* 调试过程中，务必留记录。特别时修改了代码的部分，请务必留下修改之前的代码原件，记录下修改了哪一行的什么代码。这既是给老师交进度的反馈，也是对整个项目组调试的负责。

其他暂时没有什么注意的了。大家加油调试。
