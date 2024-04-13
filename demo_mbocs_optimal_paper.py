import copy
import pdb
import random
import control
import os
import sys
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_injection_molding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# save the figure or not

save_figure=False
save_csv=False
batch_length = 200
batch_num=50
# cost function
Q = np.matrix([[100.]])
R = np.matrix([[0.5]])
"1. the state space of the injection molding"
'1.1 the time-invariant parameters'
A= np.matrix([[1.607, 1.0], [-0.6086, 0.0]])
B= np.matrix([[1.239], [-0.9282]])
C= np.matrix([[1.0,0.0]])
n=2 # state
m=1 # input
q=1 # output
'1.2 the time-varying parameters'
A_t = []
B_t = []
C_t=  []
for time in range(batch_length):
    A_t.append(copy.deepcopy(A))
    A_t[time]=A_t[time]*(0.5+0.2*np.exp(time/200))
    B_t.append(copy.deepcopy(B))
    B_t[time] = B_t[time] *(1+0.1*np.exp(time/200))
    C_t.append(copy.deepcopy(C))
C_t.append(copy.deepcopy(C))
"1.3 the system uncertainty"
A_t_env = []
B_t_env = []
C_t_env=  []
delta_A_t=np.matrix([[0.0604, -0.0204], [-0.0204, 0.0]])
delta_B_t=np.matrix([[0.062], [-0.0464]])
delta_C_t=np.matrix([[0.01,-0.01]])
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    A_t_env[time] = A_t[time] + delta_A_t * 1.0*np.exp(time/200)
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] = B_t[time]+delta_B_t*np.sin(time)
    C_t_env.append(copy.deepcopy(C))
    C_t_env[time] = C_t[time] + delta_C_t *np.sin(time)
'1.4 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))
C_t_env[batch_length] = C_t[batch_length] + delta_C_t *np.sin(time)
'1.5 the reference trajectory'
y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
y_ref[121:] = 300. * y_ref[121:]

'1.6 calculate the virtual reference trajectory matrix D_{t}'
D_t= []

for time in range(batch_length):
    D_t.append(np.ones((q,q)))
    D_t[time] = D_t[time] * (y_ref[time+1,0]/y_ref[time,0])


"2. compute the model-based optimal control law"
# the unknown system uncertainty are known
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t,Q=Q, R=R)
mb_ocs.control_law()


"3. save the data"
if save_csv==True:

    "7.1 save the data"
    mb_ocs.save_K(name='optimal_control_law')
    "7.2 save the output response"
    y_out_df = pd.DataFrame(y_out_list)
    #pdb.set_trace()
    y_out_df.to_csv('./Data/optimal_control_output_response.csv')
    "7.3 save the control sigal"
    u_df = pd.DataFrame(state_u)
    #pdb.set_trace()
    u_df.to_csv('./Data/optimal_control_control_signal.csv')
    pdb.set_trace()
a=2