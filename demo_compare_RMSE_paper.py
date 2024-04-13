import copy
import pdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
save_figure=True
batch_num=100

#config_path = os.path.split(os.path.abspath(__file__))[0]
#data_path = os.path.join(config_path, "control_policy_{}".format(batch_num))


"1. load the RMSE from Q-learning"
df_RMSE_Q_learning = pd.read_csv('./Q_learning_RMSE.csv', index_col=0)
RMSE_Q_learning = df_RMSE_Q_learning.to_numpy()

#pdb.set_trace()
"2.  load the RMSE from PI indirect ILC"
df_RMSE_PI = pd.read_csv('./comparison_algorithm/PI_Robust_RMSE.csv', index_col=0)
RMSE_PI  = df_RMSE_PI.to_numpy()


'3 Plot the RMSE plot'
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
batch_axis=range(1,batch_num+1)
fig,ax0=plt.subplots(1,1)
x_major_locator=MultipleLocator(int(batch_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
#ax0.plot(batch_axis,RMSE_Q_learning,linewidth=2,color='tab:blue',linestyle = 'dashdot')
ax0.plot(batch_axis,RMSE_Q_learning,linewidth=2,color='#46788e',linestyle = 'dashdot')
plt.plot(batch_axis,RMSE_PI,linewidth=1.5,color='orange',linestyle='solid')
ax0.grid()

xlable = 'Batch'
ylable = 'RMSE'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15,
         }
#ax0.set_ylabel('$\| \pi^{i}-\pi^{*} \|$',font2)
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.tick_params(width=2,labelsize=12)
plt.tick_params(labelsize=12)
#ax3.set_title('10-iteration')
#plt.legend(['2D Iterative Learning Control Scheme','2D ILC-RL Control Scheme'])
ax0.legend(['Q-learning-based Optimal Controller','PI-based Indirect ILC [JPC,2019]'],fontsize=11)
if save_figure==True:
    #plt.savefig('Compare_K_{}_paper.jpg'.format(batch_explore),dpi=700)
    plt.savefig('Compare_RMSE_paper.pdf')
plt.show()
pdb.set_trace()

a=2