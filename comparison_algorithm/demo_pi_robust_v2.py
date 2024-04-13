import copy
import pdb
import random

import control
import os
import sys
#pdb.set_trace()
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',1)[0]
sys.path.append(config_path)
#pdb.set_trace()
from env.time_variant_batch_sys import Time_varying_injection_molding
import numpy as np
import matplotlib.pyplot as plt
import math
# save the figure or not

save_figure=True
save_csv=False
#optimal_batch=10
#dis_fac=0.95
batch_length = 200
batch_num=50
Q = np.matrix([[100.]])
R = np.matrix([[0.5]])
#Q = np.matrix([[50.]])
#R = np.matrix([[1.]])
"1. the state space of the injection molding"
'1.1 the time-invariant parameters'
A=np.array([[1.607,-0.608,-0.928],[1.,0.,0.],[0.,0.,0.]])
B=np.array([[1.239],[0.0],[1.]])
C=np.array([[1.,0.,0.]])
n=3 # state
m=1 # input
q=1 # output
'1.2 the time-varying parameters'
A_t = []
B_t = []
C_t=  []
for time in range(batch_length):
    A_t.append(copy.deepcopy(A))
    #pdb.set_trace()
    A_t[time]=A_t[time]*(1+0.1*np.exp(time/200))
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
delta_A_t=np.matrix([[0.0804,-0.0304,-0.0464],[0.,0.,0.],[0.,0.,0.]])
delta_B_t=np.matrix([[0.062],[0.],[0.]])
#delta_C_t=np.matrix([[0.1,-0.1,]])
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    #pdb.set_trace()
    #A_t_env[time]=A_t[time]+delta_A_t*(0.5+np.sin(0.5*time))
    A_t_env[time] = A_t[time] + delta_A_t * 0.5*np.sin(time)
    #pdb.set_trace()
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] = B_t[time]+delta_B_t*0.5*np.sin(time)
    C_t_env.append(copy.deepcopy(C))
    #C_t_env[time] = C_t[time] + delta_C_t *np.sin(time)
    #C_t_env[time] = C_t[time]
    #C_t_env.append(copy.deepcopy(C))
'1.3 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))
#C_t_env[batch_length] = C_t[batch_length]
C_t_env[batch_length] = C_t[batch_length]

'1.4 the reference trajectory'
#y_ref=np.
y_ref = np.ones((batch_length+1, q))
y_ref[0]=10.* y_ref[0]
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
"2. set the PI controller"
# PI control parameters
#K_p=0.32300948
#K_i= 0.037681506696208775
# robust control parameters
#L1=-0.1044541
#L2=-5.65523151e-12
#L3=0.60202661

#K_p=0.99993268
#K_i= 0.0486206677064522
# robust control parameters
#L1=-0.04636975
#L2=-3.60277968e-12
#L3=0.95336513

"""
K_p=1.00038718
K_i= 0.04635021754653251
# robust control parameters
L1=-0.04428252
L2=-4.10911689e-13
L3=0.85740116
"""
K_p=0.75437992
K_i= 0.03873222838438218
# robust control parameters
L1=-0.04883649
L2=1.0690029e-12
L3=1.18589588



"3. set the simulated env"
#初始0 期望150
'3.1 the initial state'
x_k0 = np.array((10., 10.,10.))
sample_time = 1

#pdb.set_trace()
"3.2 set the batch system"
def state_update(t, x, u, params):
    # get the parameter from the params
    # pdb.set_trace()
    # Map the states into local variable names
    # the state x_{k+1,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3= np.array([x[2]])

    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    C_t_env = params.get('C_t')
    #pdb.set_trace()
    #if t==11:
        #pdb.set_trace()
    #if t==12:
    #    pdb.set_trace()
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +A_t_env[0,2]* z3 +B_t_env[0,0]*u
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +A_t_env[1,2]* z3+B_t_env[1,0]*u
    dz3 = A_t_env[2, 0] * z1 + A_t_env[2, 1] * z2 + A_t_env[2, 2] * z3 + B_t_env[2, 0] * u
    # pdb.set_trace()
    return [dz1, dz2, dz3]


def ouput_update(t, x, u, params):
    # Parameter setup
    # pdb.set_trace()
    # Compute the discrete updates
    C_t_env = params.get('C_t')
    #pdb.set_trace()
    y1 = C_t_env[0,0]* x[0] + C_t_env[0,1]* x[1]+ C_t_env[0,2]* x[2]

    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y1'),
    states=('dz1', 'dz2', 'dz3'), dt=1, name='Injection_Molding')

#controlled_system = TwoDimSys(batch_length=batch_length, sample_time=sample_time, sys=batch_sys, x_k0=x_k0, x_t0=x_t0)
#controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t)
controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t)


"4. simulations"
e_k_list = np.zeros(batch_length + 1)
e_k_1_list = np.zeros(batch_length + 1)

# set the e_k_1_list
#pdb.set_trace()
e_k_1_list[:] = y_ref[:, 0]
# define the sum of the e_{s} not the e
# the initial e_s_sum=0
'for e_s_sum 0=-1  1=0'
e_s_k_sum_list = np.zeros(batch_length + 1)
e_s_k_1_sum_list = np.zeros(batch_length + 1)
# define the y_s
# the initial y_s=0
ys_k_list = np.zeros(batch_length)
#pdb.set_trace()
ys_k_1_list = copy.deepcopy(0.4 * y_ref[:,0])
RMSE = np.zeros(batch_num)
state_u=[]
y_out_list=[]
y_out=[]
for batch in range(batch_num):
    # reset the sum of the current error
    e_s_sum = 0

    x_tem,y_current = controlled_system.reset()
    #if batch == 10:
    y_out=[]
    y_out.append(copy.deepcopy(y_current))
    # e
    #pdb.set_trace()
    e_current = y_ref[0,0] - y_current
    e_k_list[0] = copy.deepcopy(e_current)
    for time in range(batch_length):
        # e_sum_current
        # delta_e_s
        delta_e_s = e_s_k_sum_list[time] - e_s_k_1_sum_list[time]
        'y_s'
        y_s = ys_k_1_list[time] + L1 * delta_e_s + L2 * e_current + L3 * e_k_1_list[time + 1]
        #pdb.set_trace()
        ys_k_list[time] = copy.deepcopy(y_s)
        # e_s
        e_s = y_s - y_current
        # e_s_sum
        e_s_sum = e_s_sum + e_s
        'using the last time'
        e_s_k_sum_list[time+1] = copy.deepcopy(e_s_sum)
        # u
        u = K_p * e_s + K_i * e_s_sum
        control_signal=np.matrix([[u]])

        #pdb.set_trace()
        x_tem,y_current = controlled_system.step(control_signal)
        #pdb.set_trace()
        #if batch == 10:
            #state_u.append(copy.deepcopy(u))
        y_out.append(copy.deepcopy(y_current))
        # e
        e_current = y_ref[time+1,0] - y_current
        e_k_list[time + 1] = copy.deepcopy(e_current)
    y_out_list.append(copy.deepcopy(y_out))
    e_k_1_list = copy.deepcopy(e_k_list)
    e_s_k_1_sum_list = copy.deepcopy(e_s_k_sum_list)
    ys_k_1_list = copy.deepcopy(ys_k_list)
    # calculation of the RMSE
    tem = 0.0
    for time in range(batch_length):
        tem += (abs(y_out_list[batch][time+1]-y_ref[time+1,0])) ** 2
        RMSE[batch] = math.sqrt(tem / batch_length)
    #pdb.set_trace()
pdb.set_trace()

"6. plot the figures"
"set the global parameters"
config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 12,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
"6.1 plot the output response"

time=range(0,batch_length+1)
ax0.plot(time, y_ref, label='smoothed')
ax0.plot(time, y_out_list[batch_num-1],linestyle='dotted')


ax0.grid()
xlable = 'Time:$t$'
ylable = 'Output:$y$'
font = {'family': 'Arial',
         'weight': 'bold',
         'size': 14,
         }
ax0.set_ylabel('Output:$y$',font)
#plt.xlabel(xlable,font)
#plt.ylabel(ylable,font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
#plt.legend(['Reference Trajectory','Output Response'])
ax0.legend(['Reference Trajectory','Output Response'])
"""
"6.2 plot the control singal"

time=range(1,batch_length+1)
ax1.plot(time, state_u, label='smoothed')


plt.grid()
xlable = 'Time:$t$'
ylable = 'Control Signal:$u$'
plt.xlabel(xlable,font)
plt.ylabel(ylable,font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
#plt.legend(['Reference Trajectory','Output Response'])
if save_figure==True:
    #plt.savefig('Injection_molding_output.pdf')
    plt.savefig('mb_ocs.jpg',dpi=700)
"""
plt.show()