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
R = np.matrix([[0.1]])
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
    A_t[time]=A_t[time]*0.3*np.exp(time/200)
    #pdb.set_trace()
    B_t.append(copy.deepcopy(B))
    B_t[time] = B_t[time] *0.2* np.exp(time / 200)
    C_t.append(copy.deepcopy(C))
#pdb.set_trace()
C_t.append(copy.deepcopy(C))
"the system uncertaiy"
A_t_env = []
B_t_env = []
C_t_env=  []
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    #pdb.set_trace()
    A_t_env[time]=A_t_env[time]*0.3*np.exp(time/200)
    #pdb.set_trace()
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] = B_t_env[time] *0.2* np.exp(time / 200)
    C_t_env.append(copy.deepcopy(C))
'1.3 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))

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


"2. compute the model-based optimal control law"
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,Q=Q, R=R)
#pdb.set_trace()
mb_ocs.control_law()

"3. set the simulated env"
#初始0 期望150
'3.1 the initial state'
x_k0 = np.array((10., 10.,))
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
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup
    # pdb.set_trace()
    # Compute the discrete updates
    y1 = x[0]

    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_Molding')

#controlled_system = TwoDimSys(batch_length=batch_length, sample_time=sample_time, sys=batch_sys, x_k0=x_k0, x_t0=x_t0)
#controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t)
controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t)


"4. initial the q-learning control scheme"
Buffer=Q_learning_OCS(batch_length=batch_length,D_t=D_t,n=n,m=m,q=q,Q=Q,R=R)

Buffer.load_K(batch_num=batch_explore)
#pdb.set_trace()

"5. simulations Q-learning"

"5.1 reset the system"
x_tem,y_out=controlled_system.reset()
#state=np.block([x_tem,y_ref[1]])
#pdb.set_trace()

"5.2 reset the system"
state_u=[]
state_x=[]
y_out_list=[]
y_out_list.append(y_out)
for time in range(batch_length):
    state = np.block([x_tem, y_ref[time + 1]])
    control_signal = state * Buffer.K[time].T
    #pdb.set_trace()
    x_tem,y_out= controlled_system.step(control_signal)
    #state = np.block([x_tem, y_ref[time+1]])
    state_u.append(copy.deepcopy(control_signal[0,0]))
    state_x.append(copy.deepcopy(x_tem))
    y_out_list.append(copy.deepcopy(y_out))
    #pdb.set_trace()

#pdb.set_trace()
"6. simulations model-based optimal control scheme"

"6.1 reset the system"
x_tem,y_out=controlled_system.reset()
#state=np.block([x_tem,y_ref[1]])
#pdb.set_trace()

"6.2 reset the system"
mb_state_u=[]
mb_state_x=[]
mb_y_out_list=[]
mb_y_out_list.append(y_out)
for time in range(batch_length):
    state = np.block([x_tem, y_ref[time + 1]])
    control_signal = -state * mb_ocs.K[time].T
    #pdb.set_trace()
    x_tem,y_out= controlled_system.step(control_signal)
    #state = np.block([x_tem, y_ref[time+1]])
    mb_state_u.append(copy.deepcopy(control_signal[0,0]))
    mb_state_x.append(copy.deepcopy(x_tem))
    mb_y_out_list.append(copy.deepcopy(y_out))
    #pdb.set_trace()

"7. plot the figures"
"set the global parameters"
config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 12,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)
"6.1 plot the output response"

fig1, ax1 = plt.subplots()
time=range(0,batch_length+1)
ax1.plot(time, y_ref, label='smoothed')
ax1.plot(time, y_out_list)
ax1.plot(time, mb_y_out_list)


plt.grid()
xlable = 'Time:$t$'
ylable = 'Output:$y$'
font = {'family': 'Arial',
         'weight': 'bold',
         'size': 14,
         }
plt.xlabel(xlable,font)
plt.ylabel(ylable,font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
plt.legend(['Reference Trajectory','Q_learning','Model-based Optimal Control'])
#plt.ylim(ymin=0)
if save_figure==True:
    #plt.savefig('Injection_molding_output.pdf')
    plt.savefig('mbocs_output_response.jpg',dpi=700)
plt.show()

"6.2 plot the control signal"

fig2, ax2 = plt.subplots()
time=range(1,batch_length+1)
ax2.plot(time, state_u, label='smoothed')
ax2.plot(time, mb_state_u, label='smoothed')
plt.grid()
xlable = 'Time:$t$'
ylable = 'Control Signal:$u$'
plt.xlabel(xlable,font)
plt.ylabel(ylable,font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
#plt.legend(['Reference Trajectory','Output Response'])
if save_figure==True:
    #plt.savefig('Injection_molding_output.pdf')
    plt.savefig('mbocs_control_signal.jpg',dpi=700)
plt.show()


pdb.set_trace()
a=2