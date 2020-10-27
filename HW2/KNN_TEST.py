import pickle
import numpy as np
import pygame
from os import listdir
from os.path import isfile, join

path = "C:\\Users\\aiolb\\Desktop\\MLGame-beta4.0\\games\\arkanoid\\log\\"
Frame = []
Status = []
BallPosition = []
PlatformPosition = []
Brick = []
files = listdir(path)    ## import os 取路徑底下檔名

for f in files:         ##將路徑底下的檔名與路徑結合
  allpath = join(path, f)
  if isfile(allpath):
    with open(allpath , "rb") as f1:
        data_list1 = pickle.load(f1)
    for i in range(0 , len(data_list1)):
        BallPosition.append(data_list1[i].ball)
        PlatformPosition.append(data_list1[i].platform)
        Frame.append(data_list1[i].frame)
        Status.append(data_list1[i].status)
        Brick.append(data_list1[i].bricks)
###############################################################################

PlatX = np.array(PlatformPosition) [:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instrust = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

Ballarray = np.array(BallPosition[:-1])

BallX_position = np.array(BallPosition)[:,0][:,np.newaxis]
BallX_position_next = BallX_position[1:,:]
Ball_Vx = BallX_position_next - BallX_position[0:len(BallX_position_next),0][:,np.newaxis]

BallY_position = np.array(BallPosition)[:,1][:,np.newaxis]
BallY_position_next = BallY_position[1:,:]
Ball_Vy = BallY_position_next - BallY_position[0:len(BallY_position_next),0][:,np.newaxis]
x = np.hstack((Ballarray,PlatX[0:-1,0][:,np.newaxis],Ball_Vx,Ball_Vy))


y = instrust
np.set_printoptions(threshold=np.inf)
#--------------------------- train & test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 41)
#--------------------------- train model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler,y_test)


#--------------------------- save
filename = "C:\\Users\\aiolb\\Desktop\\MLGame-beta4.0\\games\\arkanoid\\ml\\knn.sav"
pickle.dump(knn,open(filename,"wb"))
