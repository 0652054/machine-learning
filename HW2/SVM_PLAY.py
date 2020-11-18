"""The template of the main script of the machine learning process
"""
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import games.arkanoid.communication as comm
import numpy as np
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process
    This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.

    # 2. Inform the game process that ml process is ready before start the loop.
    filename = "C:\\Users\\aiolb\\Desktop\\MLGame-beta4.0_EX\\games\\arkanoid\\ml\\SVM.sav" 
    model = pickle.load(open(filename,'rb'))
    g=0
    last_ball_x = 0
    last_ball_y = 0
    #games = 0
    #gameover = 0
    inputx = 0
    inputy = 0
    hit = False
    LRUP = 0
    ball_to_200 = 0
    comm.ml_ready()
    bricks = [] 
    # 3. Start an endless loop.
    scene_info = comm.get_scene_info()
    #print('ball_x',scene_info.ball[0], 'ball_y',scene_info.ball[1])


    while True:

        # 3.1. Receive the scene information sent from the game process.
        
    
        table = np.array(np.zeros((208,408)))
																			  
        	
        last_ball_x = scene_info.ball[0]
        last_ball_y = scene_info.ball[1]
        
        scene_info = comm.get_scene_info()
        bricks = scene_info.bricks
		
        for k in range(len(bricks)):		
            for p in range(0,25):			
                for j in range(0,10):		
                    table[bricks[k][0] + p][bricks[k][1] + j ] = 1
                    
        #print("x=",scene_info.ball[0],"y=",scene_info.ball[1])
        if(last_ball_x - scene_info.ball[0] > 0): #球往左
                if(last_ball_y - scene_info.ball[1] > 0):   #球往下
                    #going up
                    inputx = scene_info.ball[0] #取得球X座標
                    inputy = scene_info.ball[1] #取得球Y座標
                    LRUP = 3
                    #U.L
                    j = scene_info.ball[0]  #取得球X座標
                    k = scene_info.ball[1]  #取得球Y座標
                    while(j > 0 and k > 0): #球不在牆邊
                        if(table[j][k] == 1):   #磚塊座標
                            hit = True #打到磚塊
                            
                            break
                        j = j-1 #球X座標減一
                        k = k-1 #球Y座標減一
                        
                        hit = False 
                    if(hit==False): #未擊到球
                        j = scene_info.ball[0]  
                        k = scene_info.ball[1]+7
                        while(j > 0 and k > 0):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j-1
                            k = k-1
                            hit = False
                        
                  
                else:
                    #going down
                    #DL
                    LRUP = 1
                    inputx = scene_info.ball[0]
                    inputy = scene_info.ball[1]
                    j = scene_info.ball[0]
                    k = scene_info.ball[1]
                    while(j > 0 and k < 400):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j-1
                        k = k+1
                        hit = False
                    if(hit==False):
                        j = scene_info.ball[0]
                        k = scene_info.ball[1]+7
                        while(j > 0 and k < 400):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j-1
                            k = k+1
                            hit = False
                   
                    
        else:
                
                if(last_ball_y - scene_info.ball[1] > 0):
                    #going up
                    LRUP = 4
                    inputx = scene_info.ball[0]+7
                    inputy = scene_info.ball[1]
                    #U.R
                    j = scene_info.ball[0]
                    k = scene_info.ball[1]
                    while(j < 200 and k > 0):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j+1
                        k = k-1
                        hit = False
                    if(hit==False):
                        j = scene_info.ball[0]+7
                        k = scene_info.ball[1]+7
                        while(j < 200 and k > 0):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j+1
                            k = k-1
                            hit = False
                   
                else:
                    #going down
                    inputx = scene_info.ball[0]+7
                    inputy = scene_info.ball[1]
                    LRUP = 2
                    #D.R.
                    j =scene_info.ball[0]+7
                    k = scene_info.ball[1]
                    while(j < 200 and k < 400):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j+1
                        k = k+1
                        hit = False
                    if(hit==False):
                        j = scene_info.ball[0]+7
                        k = scene_info.ball[1]+7
                        while(j < 200 and k < 400):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j+1
                            k = k+1
                            hit = False
                    
                    
        ballx_to_platmid = np.abs((scene_info.ball[0]-scene_info.platform[0]))
        bally_to_platmid = 400-scene_info.ball[1]
#        print(j,k)
        inp_temp = np.array([inputx,inputy,j,k,LRUP,(200 - int(scene_info.ball[0])),ballx_to_platmid,bally_to_platmid,scene_info.platform[0]+20])
        input = inp_temp[np.newaxis,:]
        
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            #games = games + 1
            #print('played:',games,'ball_x',scene_info.ball[0], 'ball_y',scene_info.ball[1])
            #if(scene_info.status == GameStatus.GAME_OVER):
                #print('game over')
                #gameover = gameover + 1
                #print('pass rate',(games - gameover) / games)
                
            # Do some stuff if needed
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()

            continue
			

        move = model.predict(np.around(input))
        #print(move)
        #print("j=",j,"k=",k)

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        
        if(move < -0.1):
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
        elif(move > 0.1):
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
        else:
            comm.send_instruction(scene_info.frame, PlatformAction.NONE)

