import copy
import pdb
import random

import control
import os
import sys
#pdb.set_trace()
#config_path=os.path.split(os.path.abspath(__file__))[0]
#config_path=config_path.rsplit('/',1)[0]
#sys.path.append(config_path)
#pdb.set_trace()
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_injection_molding
from algorithm.q_learning_ocs import Q_learning_OCS
import numpy as np
import matplotlib.pyplot as plt
# save the figure or not

save_figure=False
save_csv=False
#optimal_batch=10
#dis_fac=0.95
batch_length = 200
batch_explore=10
# cost function
#Q = np.matrix([[100., 0.,0.], [0., 100.,0.], [0., 0.,10000.]])
#R = np.matrix([[0.001]])
#Q = np.matrix([[100., 0.,0.], [0., 100.,0.], [0., 0.,100.]])
# Benchmark
Q = np.matrix([[100.]])
R = np.matrix([[0.5]])
#Q = np.matrix([[50.]])
#R = np.matrix([[1.]])
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
    #pdb.set_trace()
    A_t[time]=A_t[time]*(0.5+0.2*np.exp(time/200))
    #pdb.set_trace()
    B_t.append(copy.deepcopy(B))
    B_t[time] = B_t[time] *(1+0.1*np.exp(time/200))
    C_t.append(copy.deepcopy(C))
#pdb.set_trace()
C_t.append(copy.deepcopy(C))
"the system uncertainty"
A_t_env = []
B_t_env = []
C_t_env=  []
delta_A_t=np.matrix([[0.0604, -0.0204], [-0.0204, 0.0]])
delta_B_t=np.matrix([[0.062], [-0.0464]])
delta_C_t=np.matrix([[0.01,-0.01]])
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    #pdb.set_trace()
    #A_t_env[time]=A_t[time]+delta_A_t*(0.5+np.sin(0.5*time))
    A_t_env[time] = A_t[time] + delta_A_t * 1.0*np.exp(time/200)
    #pdb.set_trace()
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] = B_t[time]+delta_B_t*np.sin(time)
    C_t_env.append(copy.deepcopy(C))
    C_t_env[time] = C_t[time] + delta_C_t *np.sin(time)
    #C_t_env[time] = C_t[time]
    #C_t_env.append(copy.deepcopy(C))
'1.3 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))
#C_t_env[batch_length] = C_t[batch_length]
C_t_env[batch_length] = C_t[batch_length] + delta_C_t *np.sin(time)

'1.4 the reference trajectory'
#y_ref=np.
y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
    #y_ref[time, 0] = 250
y_ref[121:] = 300. * y_ref[121:]

'1.5 calculate the virtual reference trajectory matrix D_{t}'
D_t= []

for time in range(batch_length):
    D_t.append(np.ones((q,q)))
    # pdb.set_trace()
    D_t[time] = D_t[time] * (y_ref[time+1,0]/y_ref[time,0])


"2. compute the model-based optimal control law"
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,Q=Q, R=R)
#pdb.set_trace()
#mb_ocs.control_law()
mb_ocs.load_K(name='initial_control_law')



"4. initial the q-learning control scheme"
Buffer=Q_learning_OCS(batch_length=batch_length,D_t=D_t,n=n,m=m,q=q,Q=Q,R=R)
Buffer.initial_buffer(data_length=batch_explore)
Buffer.load_buffer(num=batch_explore)
Buffer.initial_control_policy(pi=mb_ocs.K)
"5. compute the q-learning optimal control scheme"

Buffer.q_learning_iteration(batch_num=batch_explore)
Buffer.save_K(batch_num=batch_explore)

pdb.set_trace()
a=2