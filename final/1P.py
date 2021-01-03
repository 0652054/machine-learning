"""
The template of the script for the machine learning process in game pingpong
"""
# ball_position_history[-1][-2] ball_x
# ball_position_history[-1][-1] ball_y
# scene_info["ball_speed"][-2]  ball_speed_x
# scene_info["ball_speed"][-1]  ball_speed_y
# scene_info["platform_1P"][-2] platform_center_x
# scene_info["platform_1P"][-1] platform_center_y

from sklearn.linear_model import LinearRegression
import numpy as np
import math
import random
model = LinearRegression()
ball_position_history = []
bounces_leftwall_y=0
bounces_rightwall_y=0
temp_y=0
temp_y2=0
temp_y3=0
temp_x=0
temp_x2=0
temp_x3=0
ball_served_random=random.randrange(1,3)
class MLPlay:
    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False    #未發球
        self.side = side

    def update(self, scene_info):

     global bounces_leftwall_y,bounces_rightwall_y,temp_x,temp_y,temp_x2,temp_y2,temp_x3,temp_y3
     ball_down = False    
     while True:
        if scene_info["status"] != "GAME_ALIVE":    #比出勝負
            return "RESET"  #遊戲重製
        if not self.ball_served:    #如果未發球
            self.ball_served = True
            print("ball_pos:",scene_info["ball"])
            if(ball_served_random==1):
                return "SERVE_TO_RIGHT"  #往右發球
            else:
                return "SERVE_TO_LEFT"  #往左發球
                
        else:
            ball_position_history.append(scene_info["ball"])    #取得球的(X,Y)    
            if(len(ball_position_history)>1):
                if (ball_position_history[-1][1] - ball_position_history[-2][1]) > 0:   #最後一次球的Y座標-倒數第二次的球Y座標
                    ball_down = True
                else:
                    ball_down = False
                m = 0

                if (ball_position_history[-1][-2]-ball_position_history[-2][-2]) != 0:  #如果球在移動，求斜率
                    m = ((ball_position_history[-1][-1]-ball_position_history[-2][-1]) / (ball_position_history[-1][-2]-ball_position_history[-2][-2]))
                    #print('m=',m)
                #(91,408)->(84,401) (84,401)->(77,394) (0,317) (195,121)
                if(ball_down == True and m>0):
                    if(ball_position_history[-1][-1]<200): #如果球在Y=300~500這裡反彈，下一次反彈會超過板子，不用再計算球的落點位置，只計算Y=0~300區間的彈跳
                        temp_y=ball_position_history[-1][-1]+abs((195-ball_position_history[-1][-2]) /scene_info["ball_speed"][-2]* scene_info["ball_speed"][-1])#第一次反彈的Y座標 
                        temp_x=195    #第一次反彈的X座標
                        temp_y2=temp_y+abs((195/scene_info["ball_speed"][-2])*scene_info["ball_speed"][-1]) #第二次反彈的Y座標
                        temp_x2=0 #第二次反彈的X座標
                        temp_x3=temp_x2+abs(((415-temp_y2)/scene_info["ball_speed"][-1])*scene_info["ball_speed"][-2])    #球最後的落點X座標
                        temp_y3=415  #球最後的落點Y座標,因為P2板子Y座標在385，板子高度30
                        
                if(ball_down == True and m<0):
                    if(ball_position_history[-1][-1]<200): 
                        temp_y=ball_position_history[-1][-1]+abs((ball_position_history[-1][-2]/scene_info["ball_speed"][-2])*scene_info["ball_speed"][-1])
                        temp_x=0 
                        temp_y2=temp_y+abs((195/scene_info["ball_speed"][-2])*scene_info["ball_speed"][-1]) #第二次反彈的Y座標
                        temp_x2=195 
                        temp_x3=temp_x2-abs(((415-temp_y2)/scene_info["ball_speed"][-1])*scene_info["ball_speed"][-2])    
                        temp_y3=415 
                #print(ball_served_random,scene_info["platform_1P"][-2])        
                #print(temp_x,temp_y,"        ",temp_x2,temp_y2,"        ",temp_x3,temp_y3,"        ",ball_position_history[-1][-2],ball_position_history[-1][-1])
                
                if(ball_down != True):  #擊球後，板子回到中心
                    if(scene_info["platform_1P"][-2]+20<100):
                        return "MOVE_RIGHT" #板子右移
                    if(scene_info["platform_1P"][-2]+20>100):
                        return "MOVE_LEFT" #板子左移
                    if(scene_info["platform_1P"][-2]+20==100):
                        return "NONE"
                else:
                    if(scene_info["platform_1P"][-2]+20<temp_x3):   #板子中心<球最後落點
                        return "MOVE_RIGHT" #板子右移
                    if(scene_info["platform_1P"][-2]+20>temp_x3):#板子中心>球最後落點
                        return "MOVE_LEFT" #板子左移
                    else:
                        return "NONE"
            return "NONE"
     
    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        ball_served_random=random.randrange(1,3)
