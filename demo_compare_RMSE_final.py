import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pdb
import random
import control
from env.time_variant_batch_sys import Time_varying_injection_molding
from algorithm.q_learning_ocs import Q_learning_OCS
import math
# save the fig
# save the figure or not

save_figure=True
save_csv=True
batch_length = 200
batch_explore=10
batch_num=100
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
    #pdb.set_trace()
    A_t[time]=A_t[time]*(0.5+0.2*np.exp(time/200))
    #pdb.set_trace()
    B_t.append(copy.deepcopy(B))
    B_t[time] = B_t[time] *(1+0.1*np.exp(time/200))
    C_t.append(copy.deepcopy(C))
#pdb.set_trace()
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

"2. load the q-learning control scheme"
Buffer=Q_learning_OCS(batch_length=batch_length,D_t=D_t,n=n,m=m,q=q,Q=Q,R=R)

Buffer.load_K(batch_num=batch_explore)
"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array((10., 10.,))
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


"4. simulations Q-learning"

"4.1 reset the system"
RMSE = np.zeros(batch_num)
state_u=[]
state_x=[]
y_out_list=[]
y_out=[]
"load the same random initial state"
# keep the random initial state are identical
df_initial_x = pd.read_csv('./Q_learning_initial_state.csv', index_col=0)
initial_x  = df_initial_x.to_numpy()
for batch in range(batch_num):
    y_out=[]
    x_k0 = initial_x[:, batch]
    x_tem,y_current = controlled_system.reset_randomly(xk_0=x_k0)
    y_out.append(copy.deepcopy(y_current))
    for time in range(batch_length):
        state = np.block([x_tem, y_ref[time + 1]])
        control_signal = -state * Buffer.K[time].T
        x_tem,y_current= controlled_system.step(control_signal)
        state_u.append(copy.deepcopy(control_signal[0,0]))
        state_x.append(copy.deepcopy(x_tem))
        y_out.append(copy.deepcopy(y_current))
    y_out_list.append(copy.deepcopy(y_out))
    tem=0.0
    for time in range(batch_length):
        tem += (abs(y_out_list[batch][time + 1] - y_ref[time + 1, 0])) ** 2
        RMSE[batch] = math.sqrt(tem / batch_length)
if save_csv == True:
    df_RMSE = pd.DataFrame(RMSE)
    df_RMSE.to_csv('Q_learning_RMSE_final.csv')




"5. load the RMSE from Q-learning"
df_RMSE_Q_learning = pd.read_csv('./Q_learning_RMSE_final.csv', index_col=0)
RMSE_Q_learning = df_RMSE_Q_learning.to_numpy()

"6.  load the RMSE from PI indirect ILC"
df_RMSE_PI = pd.read_csv('./comparison_algorithm/PI_Robust_RMSE_final.csv', index_col=0)
RMSE_PI  = df_RMSE_PI.to_numpy()


'7 Plot the RMSE plot'

batch_axis=range(1,batch_num+1)
fig,ax0=plt.subplots(1,1)
x_major_locator=MultipleLocator(int(batch_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
ax0.plot(batch_axis,RMSE_Q_learning,linewidth=2,color='#46788e',linestyle = 'dashdot')
plt.plot(batch_axis,RMSE_PI,linewidth=1.5,color='orange',linestyle='solid')
ax0.grid()

xlable = 'Batch'
ylable = 'RMSE'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15,
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.tick_params(labelsize=12)
ax0.legend(['Q-learning-based Optimal Controller','PI-based Indirect ILC [JPC,2019]'],fontsize=11)
if save_figure==True:
    plt.savefig('Compare_RMSE_final.pdf')
plt.show()
