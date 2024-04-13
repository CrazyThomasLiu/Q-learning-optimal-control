import copy
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
save_figure=True
iteration_num=20
batch_explore=10


"1. load the optimal control law K_optimal from the csv"
df_K_optimal = pd.read_csv('./Data/optimal_control_law_final.csv', index_col=0)
K_optimal = df_K_optimal.to_numpy()

"2. load the initial control law K_initial from the csv"
df_K_initial = pd.read_csv('./Data/initial_control_law_final.csv', index_col=0)
K_initial  = df_K_initial.to_numpy()
norm_list=[]
diff_K=K_optimal-K_initial
norm = np.linalg.norm(diff_K, ord=2)
norm_list.append(copy.deepcopy(norm))
"3. load the iterative Q-learning control law K from the csv"
for ite in range(iteration_num):
    df_K_q_learning=pd.read_csv('./Data/policy_iteration_10_final/q_learning_control_policy{}_final'.format(ite+1), index_col=0)
    K_q_learning = df_K_q_learning.to_numpy()
    diff_K=K_optimal-K_q_learning
    norm = np.linalg.norm(diff_K, ord=2)
    norm_list.append(copy.deepcopy(norm))
'3 Plot the difference of the control law'
iteration_axis=range(0,iteration_num+1)
fig,ax0=plt.subplots(1,1)
x_major_locator=MultipleLocator(int(iteration_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
ax0.plot(iteration_axis,norm_list,linewidth=2,color='#46788e',linestyle = 'dashdot')
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
plt.tick_params(labelsize=12)
if save_figure==True:
    plt.savefig('Compare_pi_final.jpg', dpi=800)
    plt.savefig('Compare_pi_final.pdf')
plt.show()
