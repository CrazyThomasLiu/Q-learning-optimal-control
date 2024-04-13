import copy
import pdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
save_figure=True
iteration_num=20
batch_explore=10

#config_path = os.path.split(os.path.abspath(__file__))[0]
#data_path = os.path.join(config_path, "control_policy_{}".format(batch_num))


"1. load the optimal control law K_optimal from the csv"
df_K_optimal = pd.read_csv('./Data/optimal_control_law.csv', index_col=0)
K_optimal = df_K_optimal.to_numpy()

#pdb.set_trace()
"2. load the initial control law K_initial from the csv"
df_K_initial = pd.read_csv('./Data/initial_control_law.csv', index_col=0)
K_initial  = df_K_initial.to_numpy()
# set the list for the difference between optimal control law and the iterative control law
norm_list=[]
diff_K=K_optimal-K_initial
norm = np.linalg.norm(diff_K, ord=2)
norm_list.append(copy.deepcopy(norm))
#pdb.set_trace()
"3. load the iterative Q-learning control law K from the csv"
for ite in range(iteration_num):
    df_K_q_learning=pd.read_csv('./Data/policy_iteration_10/q_learning_control_policy{}'.format(ite+1), index_col=0)
    #df_K_q_learning = pd.read_csv('./Data/policy_iteration_{}/q_learning_control_policy{}.csv'.format(batch_explore,ite+1), index_col=0)
    K_q_learning = df_K_q_learning.to_numpy()
    #pdb.set_trace()
    #K_off_policy=K_off_policy[:, 0: 52]
    diff_K=K_optimal-K_q_learning
    #tem=0
    #for item in range(diff_K.shape[1]):
    #    tem+= abs(diff_K[0,item])
    #tem=tem/diff_K.shape[1]
    norm = np.linalg.norm(diff_K, ord=2)
    norm_list.append(copy.deepcopy(norm))
    #norm_list.append(copy.deepcopy(tem))
    #pdb.set_trace()
    #a=2

#pdb.set_trace()

'3 Plot the difference of the control law'
"set the global parameters"
"""

config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 15,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)
"""
iteration_axis=range(0,iteration_num+1)
fig,ax0=plt.subplots(1,1)
x_major_locator=MultipleLocator(int(iteration_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
#ax0.plot(iteration_axis,norm_list,linewidth=2,color='tab:blue',linestyle = 'dashdot')
#ax0.plot(iteration_axis,norm_list,linewidth=2,color='#78b7c9',linestyle = 'dashdot')
ax0.plot(iteration_axis,norm_list,linewidth=2,color='#46788e',linestyle = 'dashdot')
#plt.plot(batch_time,y_2dilc_rl_show,linewidth=1.5,color='tab:orange',linestyle='solid')
ax0.grid()

xlable = 'Policy Iteration'
ylable = '$\| \pi^{i}-\pi^{*} \|$'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15,
         }
ax0.set_ylabel('$\| \pi^{i}-\pi^{*} \|$',font2)
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.tick_params(width=2,labelsize=12)
plt.tick_params(labelsize=12)
#ax3.set_title('10-iteration')
#plt.legend(['2D Iterative Learning Control Scheme','2D ILC-RL Control Scheme'])
if save_figure==True:
    #plt.savefig('Compare_K_{}_paper.jpg'.format(batch_explore),dpi=700)
    plt.savefig('Compare_K_{}_paper.pdf'.format(batch_explore))
plt.show()
pdb.set_trace()

a=2