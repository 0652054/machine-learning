import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from numba import cuda 
print(cuda.gpus)


     
path='C:\\Users\\aiolb\\Desktop\\MLGame-beta8.0.1\\games\\pingpong\\log'
Frame = []
BallPosition = []
BallSpeed=[]
BallSpeed_x=[]
BallSpeed_y=[]
PlatformPosition1P = []
PlatformPosition1P_x = []
PlatformPosition2P = []
BallPosition_x=[]
BallPosition_y=[]
files = listdir(path)
k=0

for f in files:         ##將路徑底下的檔名與路徑結合
    allpath = join(path, f)  
    if isfile(allpath):
        with open(allpath , "rb") as f1:
            data_list1 = pickle.load(f1)     
            for ml_name in data_list1.keys():
                if ml_name == "record_format_version":
                    continue
        target_record = data_list1[ml_name]
        for n in range(0,len(target_record["scene_info"])):
            Frame.append(target_record["scene_info"][n]["frame"])
            PlatformPosition1P.append(target_record["scene_info"][n]["platform_1P"])
            PlatformPosition2P.append(target_record["scene_info"][n]["platform_2P"])
            BallPosition.append(target_record["scene_info"][n]["ball"])
            BallSpeed.append(target_record["scene_info"][n]["ball_speed"])
            BallPosition_x.append(target_record["scene_info"][n]["ball"][0])
            BallPosition_y.append(target_record["scene_info"][n]["ball"][1])   
        
    
PlatX = np.array(PlatformPosition1P) [:,0][:,np.newaxis] #[:,0]->取所有第一陣列的第一個數值(X座標) #[:,np.newaxis]->陣列變成直的
PlatX_next = PlatX[1:,:] #除了第一個值以外都要
instrust = (PlatX_next-PlatX[0:len(PlatX_next)])/5 #板子位移量為5

Ballarray = np.array(BallPosition[:-1]) 

# =============================================================================
# BallX_position = np.array(BallPosition)[:,0][:,np.newaxis]
# BallX_position_next = BallX_position[1:,:]
# Ball_Vx = BallX_position_next - BallX_position[0:len(BallX_position_next),0][:,np.newaxis] #球X座標的位移量
# =============================================================================
Ball_Vx=np.array(BallSpeed)[:-1,0][:,np.newaxis]
# =============================================================================
# BallY_position = np.array(BallPosition)[:,1][:,np.newaxis]
# BallY_position_next = BallY_position[1:,:]
# Ball_Vy = BallY_position_next - BallY_position[0:len(BallY_position_next),0][:,np.newaxis]#球Y座標的位移量
# =============================================================================
Ball_Vy=np.array(BallSpeed)[:-1,1][:,np.newaxis]
x = np.hstack((Ballarray,PlatX[:-1,0][:,np.newaxis],Ball_Vx,Ball_Vy))   #[195     415     186         -9            -7          ] 
                                                                        #[ball_x  ball_y  platform_x  ball_x_speed  ball_y_speed]

np.set_printoptions(threshold=np.inf)
y = instrust 
y = np.array(y, dtype=int)
print(len(instrust))
#data=open("C:\\Users\\aiolb\\Desktop\\data.txt",'w+') 
#print(y,file=data)
#data.close()


# =============================================================================
# import csv
# 
# # 開啟輸出的 CSV 檔案
# with open('C:\\Users\\aiolb\\Desktop\\data.csv', 'w', newline='') as csvFile:
#   # 建立 CSV 檔寫入器
#   writer = csv.writer(csvFile)
# 
#   # 1.直接寫出-標題
#   writer.writerow(['球座標X','球座標Y','球速X','球速Y','板子X','預測'])
#   for n in range(0,len(y)):
#       writer.writerow([Ballarray[n][-2],Ballarray[n][-1],Ball_Vx[n],Ball_Vy[n],PlatX[:-1,0][:,np.newaxis][n],y[n]])
# 
# fig, ax = plt.subplots()
# #ax.scatter(BallPosition_x,instrust)
# ax.scatter(BallPosition_x,BallPosition_y)
# plt.show()
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)

# =============================================================================
# with open('C:\\Users\\aiolb\\Desktop\\data2.csv', 'w', newline='') as csvFile:
#   # 建立 CSV 檔寫入器
#   writer = csv.writer(csvFile)
# 
#   # 1.直接寫出-標題
#   writer.writerow(['x_train','x_test','y_train','y_test'])
#   for n in range(0,len(y)):
#       writer.writerow([x_train,x_test,y_train,y_test])
# =============================================================================


svr = SVR(gamma=0.001,C = 1,epsilon = 0.1,kernel = 'rbf')

svr.fit(x_train,y_train)   
y_predict = svr.predict(x_test)
# =============================================================================
# r2 = accuracy_score(y_predict,y_test)
# mse = mean_squared_error(y_test, y_predict)
# =============================================================================
filename = "C:\\Users\\aiolb\\Desktop\\MLGame-beta8.0.1\\games\\pingpong\\ml\\SVM3.sav"
pickle.dump(svr,open(filename,"wb"))
# =============================================================================
# print("R2:",r2) #R2越趨近1越好
# print("MSE:",mse) #MSE越小越好
# =============================================================================
PlatX = np.array(PlatformPosition2P) [:,0][:,np.newaxis] #[:,0]->取所有第一陣列的第一個數值(X座標) #[:,np.newaxis]->陣列變成直的
PlatX_next = PlatX[1:,:] #除了第一個值以外都要
instrust = (PlatX_next-PlatX[0:len(PlatX_next)])/5 #板子位移量為5

Ballarray = np.array(BallPosition[:-1]) 

# =============================================================================
# BallX_position = np.array(BallPosition)[:,0][:,np.newaxis]
# BallX_position_next = BallX_position[1:,:]
# Ball_Vx = BallX_position_next - BallX_position[0:len(BallX_position_next),0][:,np.newaxis] #球X座標的位移量
# =============================================================================
Ball_Vx=np.array(BallSpeed)[:-1,0][:,np.newaxis]
# =============================================================================
# BallY_position = np.array(BallPosition)[:,1][:,np.newaxis]
# BallY_position_next = BallY_position[1:,:]
# Ball_Vy = BallY_position_next - BallY_position[0:len(BallY_position_next),0][:,np.newaxis]#球Y座標的位移量
# =============================================================================
Ball_Vy=np.array(BallSpeed)[:-1,1][:,np.newaxis]
x = np.hstack((Ballarray,PlatX[:-1,0][:,np.newaxis],Ball_Vx,Ball_Vy))   #[195     415     186         -9            -7          ] 
                                                                        #[ball_x  ball_y  platform_x  ball_x_speed  ball_y_speed]

np.set_printoptions(threshold=np.inf)
y = instrust 
y = np.array(y, dtype=int)
print(len(instrust))
#data=open("C:\\Users\\aiolb\\Desktop\\data.txt",'w+') 
#print(y,file=data)
#data.close()


# =============================================================================
# import csv
# 
# # 開啟輸出的 CSV 檔案
# with open('C:\\Users\\aiolb\\Desktop\\data.csv', 'w', newline='') as csvFile:
#   # 建立 CSV 檔寫入器
#   writer = csv.writer(csvFile)
# 
#   # 1.直接寫出-標題
#   writer.writerow(['球座標X','球座標Y','球速X','球速Y','板子X','預測'])
#   for n in range(0,len(y)):
#       writer.writerow([Ballarray[n][-2],Ballarray[n][-1],Ball_Vx[n],Ball_Vy[n],PlatX[:-1,0][:,np.newaxis][n],y[n]])
# 
# fig, ax = plt.subplots()
# #ax.scatter(BallPosition_x,instrust)
# ax.scatter(BallPosition_x,BallPosition_y)
# plt.show()
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)

# =============================================================================
# with open('C:\\Users\\aiolb\\Desktop\\data2.csv', 'w', newline='') as csvFile:
#   # 建立 CSV 檔寫入器
#   writer = csv.writer(csvFile)
# 
#   # 1.直接寫出-標題
#   writer.writerow(['x_train','x_test','y_train','y_test'])
#   for n in range(0,len(y)):
#       writer.writerow([x_train,x_test,y_train,y_test])
# =============================================================================


svr = SVR(gamma=0.001,C = 1,epsilon = 0.1,kernel = 'rbf')

svr.fit(x_train,y_train)   
y_predict = svr.predict(x_test)
# =============================================================================
# r2 = accuracy_score(y_predict,y_test)
# mse = mean_squared_error(y_test, y_predict)
# =============================================================================
filename = "C:\\Users\\aiolb\\Desktop\\MLGame-beta8.0.1\\games\\pingpong\\ml\\SVM4.sav"
pickle.dump(svr,open(filename,"wb"))