import sys
import os
import pdb
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import math
import csv
save_figure=True
batch_length = 200
batch_explore=10
n=2 # state
m=1 # input
q=1 # output
"1. Load the output response from csv"

df_y_out = pd.read_csv('./Data/Buffer_y_{}.csv'.format(batch_explore), index_col=0)
y_out = df_y_out.to_numpy()
y_out = y_out.reshape(q, int(y_out.shape[0] / q), batch_explore)

"2. Load the output response from csv"

df_y_ref = pd.read_csv('./Data/Buffer_y_ref_{}.csv'.format(batch_explore), index_col=0)
y_ref = df_y_ref.to_numpy()
y_ref = y_ref.reshape(q, int(y_ref.shape[0] / q), batch_explore)
#pdb.set_trace()

"3. plot the 3D figure for the exploration"
"""
config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 12,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)
"""
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
batch_before=np.ones(batch_length+1,dtype=int)
t=range(batch_length+1)
#pdb.set_trace()
ax.plot3D(batch_before*(batch_explore),t, y_ref[0,:,0].squeeze(),linestyle='dashed',linewidth=1.5,color='#97b319')
for batch in range(batch_explore):
    batch_plot=batch_before*(batch+1)
    ax.plot3D(batch_plot, t, y_out[0,:,batch].squeeze(), linewidth=1.5, color='black')
    #if (item2%2)==0:
    #    ax.plot3D(batch_plot,t, u_rl_data_list[item2].squeeze(),linewidth=1,color='black')

xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Output Response'
#change the font3

font3 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15
         }

ax.set_xlabel(xlable,font3)
ax.set_ylabel(ylable,font3)
ax.set_zlabel(zlable,font3)
#ax.set_xlabel(xlable)
#ax.set_ylabel(ylable)
#ax.set_zlabel(zlable)
ax.legend(['$y_{k,t}^{r}$','$y_{k,t}$'],fontsize=13)
plt.tick_params(labelsize=12)
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('sample_data_3D.pdf')
plt.show()

pdb.set_trace()
a=2