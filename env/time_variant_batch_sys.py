
import pdb
import numpy as np
import control
import copy


class  Time_varying_injection_molding:
    def __init__(self,batch_length:int,sample_time,sys,x_k0,A_t,B_t,C_t,D_t):
        self.batch_length=batch_length
        # the state space
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        # the virtual reference trajectory
        self.D_t=D_t

        self.sys=sys
        self.x_k0=x_k0   # k=1,2,3.... \infty
        self.T=sample_time

        self.T_in=np.array((0.0,0.0))

        self.X0=copy.deepcopy(self.x_k0)

    def reset(self):
        self.T_in = np.array((0.0, 0.0))
        self.X0=copy.deepcopy(self.x_k0)
        return self.X0,self.X0[0]

    def reset_randomly(self,xk_0):
        self.T_in = np.array((0.0, 0.0))
        self.X0=xk_0
        return self.X0,self.X0[0]
    def step(self, input_signal):
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T
        # the current time step
        t = int(self.T_in[1])
        input_signal = np.repeat(input_signal, 2, axis=1)
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, input_signal, X0=self.X0,
                                                               params={
                                                                       "A_t": self.A_t[int(t - 1)],
                                                                       "B_t": self.B_t[int(t - 1)],
                                                                       "C_t": self.C_t[int(t)]}, return_x=True)
        self.X0 = x_step[:, 1]
        y_out=y_step[1]
        return self.X0,y_out  # the first item for the control, the second item for the observation
