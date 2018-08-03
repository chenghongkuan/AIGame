# -*- coding: UTF-8 -*-

import threading
import time

exitFlag = 0


class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.learn_flag = 1
        # globals(learn_flag)

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print("Starting ", self.name)
        print("输入0退出接受线程，输入1开始学习，输入2停止学习")
        # print_time(self.name, self.counter, 5)
        while(exitFlag == 0):
            receive = input()
            if receive == "":
                pass
            elif receive == "0":
                break
            elif receive == "1":
                self.learn_flag = 1
                print("开始学习")
            elif receive == "2":
                self.learn_flag = 2
                print("停止学习")
            else:
                print("输入错误")
        print("Exiting ", self.name)


def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1