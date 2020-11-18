"""The template of the main script of the machine learning process
"""
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
import csv

dirpath = 'C:\\Users\\aiolb\\Desktop\\MLGame-beta4.0_EX\\games\\arkanoid\\log'
BallPosition = []
PlatformPosition = []
#DL = 1 , DR = 2 , UL = 3 , UR = 4
LRUP = []
Slope = []
last_ball_x = 0
last_ball_y = 0
files = listdir(dirpath)
log_number = 0
Frame = []
bricks = []
hit = False
hitarray = []
Plat_X = []
PlatX = []
ball_to_200 = 0
ballx_to_platmid = 0
bally_to_platmid = 0
ball_to_plat = []
zerocount = 0
np.set_printoptions(threshold=np.inf)

for k in range(0,1):
    for f in files:
      log_number = log_number + 1
      fullpath = join(dirpath, f)
      if isfile(fullpath):
        with open(fullpath , "rb") as f1:
            data_list1 = pickle.load(f1)
        for i in range(0 , len(data_list1)):
            table = np.array(np.zeros((208,408)))
            bricks = data_list1[i].bricks
            for k in range(len(bricks)):
                for p in range(0,25):
                    for j in range(0,10):
                        table[bricks[k][0] + p][bricks[k][1] + j ] = 1
            BallPosition.append(data_list1[i].ball)
              
            PlatformPosition.append(data_list1[i].platform)
            Frame.append(data_list1[i].frame)
            if(last_ball_x - data_list1[i].ball[0] > 0):
                if(last_ball_y - data_list1[i].ball[1] > 0):
                    #going up
                    ball_to_200 = 200 - int(data_list1[i].ball[0])
                    LRUP.append(np.array((3,ball_to_200)))
                    ballx_to_platmid = np.abs((data_list1[i].ball[0]-data_list1[i].platform[0]))
                    ball_to_plat.append(np.array((ballx_to_platmid,400-data_list1[i].ball[1])))
                    #U.L
                    j = data_list1[i].ball[0]
                    k = data_list1[i].ball[1]
                    while(j > 0 and k > 0):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j-1
                        k = k-1
                        hit = False
                    if(hit==False):
                        j = data_list1[i].ball[0]
                        k = data_list1[i].ball[1]+3
                        while(j > 0 and k > 0):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j-1
                            k = k-1
                            hit = False
                    
                    hitarray.append(np.array((j,k)))
                    
                else:
                    #going down
                    ball_to_200 = 200 - int(data_list1[i].ball[0])
                    LRUP.append(np.array((1,ball_to_200)))
                    ballx_to_platmid = np.abs((data_list1[i].ball[0]-data_list1[i].platform[0]))
                    ball_to_plat.append(np.array((ballx_to_platmid,400-data_list1[i].ball[1])))
                    #DL
                    j = data_list1[i].ball[0]
                    k = data_list1[i].ball[1]
                    while(j > 0 and k < 400):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j-1
                        k = k+1
                        hit = False
                    if(hit==False):
                        j = data_list1[i].ball[0]
                        k = data_list1[i].ball[1]+3
                        while(j > 0 and k < 400):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j-1
                            k = k+1
                            hit = False
                    hitarray.append(np.array((j,k)))
                    
            else:
                if(last_ball_y - data_list1[i].ball[1] > 0):
                    #going up
                    ball_to_200 = 200 - int(data_list1[i].ball[0])
                    LRUP.append(np.array((4,ball_to_200)))
                    ballx_to_platmid = np.abs((data_list1[i].ball[0]-data_list1[i].platform[0]))
                    ball_to_plat.append(np.array((ballx_to_platmid,400-data_list1[i].ball[1])))
                    #U.R
                    j = data_list1[i].ball[0]+3
                    k = data_list1[i].ball[1]
                    while(j < 200 and k > 0):
                        
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j+1
                        k = k-1
                        hit = False
                    if(hit==False):
                        j = data_list1[i].ball[0]+3
                        k = data_list1[i].ball[1]+3
                        while(j < 200 and k > 0):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j+1
                            k = k-1
                            hit = False
                    hitarray.append(np.array((j,k)))
                else:
                    #going down
                    ball_to_200 = 200 - int(data_list1[i].ball[0])
                    LRUP.append(np.array((2,ball_to_200)))
                    ballx_to_platmid = np.abs((data_list1[i].ball[0]-data_list1[i].platform[0]))
                    ball_to_plat.append(np.array((ballx_to_platmid,400-data_list1[i].ball[1])))
                    #D.R.
                    j = data_list1[i].ball[0]+3
                    k = data_list1[i].ball[1]
                    while(j < 200 and k < 400):
                        if(table[j][k] == 1):
                            hit = True
                            break
                        j = j+1
                        k = k+1
                        hit = False
                    if(hit==False):
                        j = data_list1[i].ball[0]+3
                        k = data_list1[i].ball[1]+3
                        while(j < 200 and k < 400):
                            if(table[j][k] == 1):
                                hit = True
                                break
                            j = j+1
                            k = k+1
                            hit = False
                    hitarray.append(np.array((j,k)))
            

            last_ball_x = data_list1[i].ball[0]
            last_ball_y = data_list1[i].ball[1]
PlatX = np.array(PlatformPosition) [:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instrust = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5
Plat_X = PlatX
for i in range(0,len(Plat_X)):
    Plat_X[i][0] = Plat_X[i][0] +20


ball_to_plat =  np.array(ball_to_plat[:-1])
Ballarray = np.array(BallPosition[:-1])
hitarray = np.array((hitarray[:-1]))
LRUP = np.array((LRUP[:-1]))
x = np.hstack((Ballarray,hitarray,LRUP,ball_to_plat,Plat_X[0:-1,0][:,np.newaxis]))



y = instrust 


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 41)

from sklearn.svm import SVR

svr = SVR(gamma=0.001,C = 1,epsilon = 0.1,kernel = 'rbf')


svr.fit(x_train,y_train)   
y_predict = svr.predict(x_test)




filename = "C:\\Users\\aiolb\\Desktop\\MLGame-beta4.0_EX\\games\\arkanoid\\ml\\SVM.sav"
pickle.dump(svr,open(filename,"wb"))