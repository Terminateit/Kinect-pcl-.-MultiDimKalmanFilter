#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utm

# read the data from both sensors: GPS and accelerometer

acc_data = 'sensordata/accelerometer.csv'
gps_data = 'sensordata/gps.csv'

acc_data = pd.read_csv(acc_data)
gps_data = pd.read_csv(gps_data)


lat = gps_data['lat'].values
lon = gps_data['lng'].values

# Convert lat and lng measurements into world coordinates

gps_x, gps_y, lon_zone, lat_zone = utm.from_latlon(lat, lon)

ddx = acc_data['x'].values
ddy = acc_data['y'].values

gps_time = gps_data['time'].values
acc_time = acc_data['time'].values


# In[2]:


# function for plot

def plot_xy(x,y, xlabel, ylabel, legend):
    fig = plt.figure(figsize=(8,8))
    
    plt.scatter(x,y, s=5, label='State', c='grey')
    plt.scatter(x[0],y[0], s=30, label='Start', c='g')
    plt.scatter(x[-1],y[-1], s=30, label='Goal', c='r')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(legend)
    plt.legend(loc='best')
    plt.axis()


# In[3]:


plot_xy(gps_x, gps_y, 'x coordinate', 'y coordinate', 'GPS data')

plot_xy(gps_time, gps_x, 'time', 'x coordinate', 'GPS data')

plot_xy(gps_time, gps_y, 'time', 'x coordinate', 'GPS data')


# In[4]:


plot_xy(ddx, ddy, 'x acceleration', 'y acceleration', 'Accelerometer data')

plot_xy(acc_time, ddx, 'time', 'x acceleration', 'Accelerometer data')

plot_xy(acc_time, ddy, 'time', 'y acceleration', 'Accelerometer data')


# In[5]:


time = acc_time
# we have data from sensors with different shapes. So, we should interpolate the GPS data
# in order to get the same shape as accelerometer data

x = np.interp(time, gps_time, gps_x)
y = np.interp(time, gps_time, gps_y)

plot_xy(x, y, 'x coordinate', 'y coordinate', 'Interpolated GPS data')

plot_xy(time, x, 'time', 'x coordinate', 'Interpolated GPS data')

plot_xy(time, y, 'time', 'x coordinate', 'Interpolated GPS data')


# In[6]:


# Let's introduce the means and stds of our measurements and process
std_x = x.std()
mean_x = x.mean()
std_y = y.std()
mean_y = y.mean()
std_ddx = ddx.std()
mean_ddx = ddx.mean()
std_ddy = ddy.std()
mean_ddy = ddy.mean()

#GUESS Correction
# Error in gps
std_x = 25
std_y = 25

# #GUESS Correction
# # Error in accelerator
std_ddx = 7
std_ddy = 7


# In[7]:


# Measurement noise covariance matrix R
def matrix_R(std_x, std_y):
    R = np.array([[std_x**2, 0],
                  [0, std_y**2]], dtype=float)
    R = R.reshape(2, 2)
    return R

# Process noise co-variance matrix Q 
def matrix_Q(std_ddx, std_ddy, dt):
    sigma_x = std_ddx*(dt**2)/2
    sigma_y = std_ddy*(dt**2)/2
    sigma_dx = std_ddx*dt
    sigma_dy = std_ddy*dt
    Q = np.array([[sigma_x**2, 0, sigma_x*sigma_dx, 0],
                  [0, sigma_y**2, 0, sigma_y*sigma_dy],
                  [sigma_x*sigma_dx, 0, sigma_dx**2, 0],
                  [0, sigma_y*sigma_dy, 0, sigma_dy**2]], dtype=float)
    Q = Q.reshape(4, 4)
    return Q

# State vector X [x, y, dx, dy]
def vector_X(x, y, dx, dy):
    X = np.array([[x],
                  [y],
                  [dx],
                  [dy]], dtype=float)
    X = X.reshape(4, 1)
    return X

# Dynamics matrix A
def matrix_A(dt):
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    A = A.reshape(4, 4)
    return A

# Control Matrix
def matrix_B(dt):
    B = np.array([[(dt**2)/2, 0],
                  [0, (dt**2)/2],
                  [dt, 0],
                  [0, dt]], dtype=float)
    B = B.reshape(4, 2)
    return B

# Control Input u
def vector_u(ddx, ddy, dt):
    u = np.array([[ddx],
                  [ddy]], dtype=float)
    u = u.reshape(2, 1)
    return u

# Observation Matrix
def matrix_C():
    C = np.array([[1, 0],
                  [0, 1]], dtype=float)
    C = C.reshape(2, 2)
    return C

# Measuring Matrix
def matrix_H():
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    H = H.reshape(2, 4)
    return H

# Matrix of measurements' noise
def vector_Z(std_x, std_y):
    Z = np.array([[np.random.normal(loc = 0, scale = std_x)],
                  [np.random.normal(loc = 0, scale = std_y)]], dtype=float)
    Z = Z.reshape(2, 1)
    return Z

# Matrix of process' noise
def vector_W(std_ddx, std_ddy, dt):
    sigma_x = std_ddx*(dt**2)/2
    sigma_y = std_ddy*(dt**2)/2
    sigma_dx = std_ddx*dt
    sigma_dy = std_ddy*dt
    W = np.array([[np.random.normal(loc = 0, scale = sigma_x)],
                  [np.random.normal(loc = 0, scale = sigma_y)],
                  [np.random.normal(loc = 0, scale = sigma_dx)],
                  [np.random.normal(loc = 0, scale = sigma_dy)]], dtype=float)
    W = W.reshape(4, 1)
    return W

# Co-variance matrix P
def vector_P():
    P = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    P = P.reshape(4, 4)
    return P
n = time.shape[0] - 1
u = np.zeros((2,n), dtype=float)


# In[8]:


time = time/1000
dt = []

for i in range(n):
    dt.append(time[i+1] - time[i])
    u_i = vector_u(ddx[i], ddy[i], dt[i])
    u[:, i] = u_i[:, 0]
    
dt = np.array(dt)

Ym = np.array([x[0:-1], y[0:-1]])

C = matrix_C()
R = matrix_R(std_x, std_y)
H = matrix_H()

X_0 = vector_X(x[0], y[0], 0, 0)
X = np.zeros((4,n), dtype=float)
X[:, 0] = X_0[:, 0]

def Kalman_algorithm():
    Kalman_list = []
    P = vector_P()
    for i in range(n-1):
        A = matrix_A(dt[i])
        B = matrix_B(dt[i])
        Q = matrix_Q(std_ddx, std_ddy, dt[i])
        Z = vector_Z(std_x, std_y)
        W = vector_W(std_ddx, std_ddy, dt[i])
        # PREDICTION
        # Calculate the state pred.
        X_pred = np.matmul(A, X[:, i]) + np.matmul(B, u[:, i]) #+ W[:, 0]
        # Calculate the covariance matrix
        P = np.matmul(A, np.matmul(P, A.transpose())) + Q

        # CORRECTION
        # Compute the Kalman gain
        K = np.matmul(np.matmul(P, H.transpose()), np.linalg.pinv(np.matmul(H, np.matmul(P, H.transpose())) + R))
        # Compute the measurements
        Y = np.matmul(C, Ym[:, i]).reshape(-1,1) #+ Z
        # Update state extimate
        x = X_pred.reshape(-1, 1) + np.matmul(K, Y - np.matmul(H, X_pred).reshape(-1,1))
        X[:, i+1] = x.reshape(-1,)
        # Update the covariance matrix
        P = np.matmul((np.eye(4, k=0) - np.matmul(K, H)), P)
        Kalman_list.append(K)
        Kalman_gains = np.array(Kalman_list)
    return X, Kalman_gains


# In[9]:


X, Kalman_gains = Kalman_algorithm()


# In[10]:


plot_xy(X[0], X[1], 'x coordinate', 'y coordinate', 'Position after Fusion')


# In[11]:


plot_xy(x,y, 'x coordinate', 'y coordinate', 'Position before Fusion')


# In[ ]:




