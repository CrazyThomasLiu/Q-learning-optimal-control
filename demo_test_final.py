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
import pandas as pd
# save the figure or not

save_figure=True
save_csv=False
batch_length = 200
batch_explore=10
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
C_t_env.append(copy.deepcopy(C))
C_t_env[batch_length] = C_t[batch_length] + delta_C_t *np.sin(time)
'1.4 the reference trajectory'
y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
y_ref[121:] = 300. * y_ref[121:]

'1.5 calculate the virtual reference trajectory matrix D_{t}'
D_t= []

for time in range(batch_length):
    D_t.append(np.ones((q,q)))
    D_t[time] = D_t[time] * (y_ref[time+1,0]/y_ref[time,0])


"2. compute the model-based optimal control law"
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,Q=Q, R=R)
mb_ocs.load_K(name='initial_control_law_final')

"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array((50., 50.,))
sample_time = 1
"3.2 set the batch system"
def state_update(t, x, u, params):
    # get the parameter from the params
    # Map the states into local variable names
    # the state x_{k+1,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])

    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    C_t_env = params.get('C_t')
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup
    # Compute the discrete updates
    C_t_env = params.get('C_t')
    y1 = C_t_env[0,0]* x[0] + C_t_env[0,1]* x[1]

    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_Molding')

controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t)


"4. initial the q-learning control scheme"
Buffer=Q_learning_OCS(batch_length=batch_length,D_t=D_t,n=n,m=m,q=q,Q=Q,R=R)

Buffer.load_K(batch_num=batch_explore)

"5. simulation Q-learning"

"5.1 reset the system"
x_tem,y_out=controlled_system.reset()

"5.2 reset the system"
state_u=[]
state_x=[]
y_out_list=[]
y_out_list.append(y_out)
for time in range(batch_length):
    state = np.block([x_tem, y_ref[time + 1]])
    control_signal = -state * Buffer.K[time].T
    x_tem,y_out= controlled_system.step(control_signal)
    state_u.append(copy.deepcopy(control_signal[0,0]))
    state_x.append(copy.deepcopy(x_tem))
    y_out_list.append(copy.deepcopy(y_out))

"6. simulations model-based optimal control scheme"

"6.1 reset the system"
x_tem,y_out=controlled_system.reset()

"6.2 reset the system"
mb_state_u=[]
mb_state_x=[]
mb_y_out_list=[]
mb_y_out_list.append(y_out)
for time in range(batch_length):
    state = np.block([x_tem, y_ref[time + 1]])
    control_signal = -state * mb_ocs.K[time].T
    x_tem,y_out= controlled_system.step(control_signal)
    #state = np.block([x_tem, y_ref[time+1]])
    mb_state_u.append(copy.deepcopy(control_signal[0,0]))
    mb_state_x.append(copy.deepcopy(x_tem))
    mb_y_out_list.append(copy.deepcopy(y_out))

"7. plot the figures"
"set the global parameters"
config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 12,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
"7.1 plot the output response"

time=range(0,batch_length+1)
ax0.plot(time, y_ref,linewidth=2,color='#46788e', label='smoothed')
ax0.plot(time, mb_y_out_list,linewidth=2,color='orange')
ax0.plot(time, y_out_list,linewidth=2,color='black',linestyle='dotted')

ax0.grid()
xlable = 'Time:$t$'
ylable = 'Output:$y_{k,t}$'
font = {'family': 'Arial',
         'weight': 'bold',
         'size': 15,
         }
ax0.set_ylabel('Output:$y_{k,t}$',font)
plt.tick_params(labelsize=12)
ax0.legend(['Reference Trajectory','Model-based Initial Controller','Q-learning-based Optimal Controller'],fontsize=11)

"7.2 plot the control signal"

time=range(1,batch_length+1)
ax1.plot(time, mb_state_u, color='orange',label='smoothed')
ax1.plot(time, state_u, color='black',label='smoothed')
xlable = 'Time:$t$'
ylable = 'Control Signal:$u_{k,t}$'
plt.xlabel(xlable,font)
plt.ylabel(ylable,font)
ax1.grid()
ax1.legend(['Model-based Initial Controller','Q-learning-based Optimal Controller'],fontsize=11)
if save_figure==True:

    plt.savefig('Q_learning_final.pdf')

plt.show()

