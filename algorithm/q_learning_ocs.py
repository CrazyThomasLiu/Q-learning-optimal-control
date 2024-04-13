import pdb
import numpy as np
import copy
import pandas as pd
import copy
from scipy.optimize import lsq_linear
import os
class Q_learning_OCS:
    """
    Model-free Optimal control
        :param batch_length: The fixed batch length each iteration
        :param n: dimensions of the state variable
        :param m: dimensions of the input variable
        :param Q: state cost matrix
        :param R: input cost matrix
        :param F_0: initial value matrix
        :param K_t: current control policy
        :param k: after k iterations updating the control policy
    """
    def __init__(self, batch_length: int,D_t,n,m,q,Q,R):
        self.batch_length=batch_length
        # the dimensions of the state space
        self.n=n # the dimensions of the state variable
        self.m=m   # the dimensions of the input variable
        self.q=q  # the dimensions of the output variable
        # the cost function
        self.Q = Q
        self.R = R
        # the known virtual reference trajectory
        self.D_t=D_t

        # the initial control policy
        self.pi=[]
        for time in range(self.batch_length):
            self.pi.append(np.zeros([self.m,self.n+self.q]))
        self.pi.reverse()
        self.pi_improved=copy.deepcopy(self.pi)

        # the V-function matrix P_t^{i}
        self.P_t=[]
        self.P_t_length=0  # the sum of the all time P_t
        for time in range(self.batch_length):
            self.P_t.append(np.zeros([self.n+self.q,self.n+self.q]))
            self.P_t_length+=self.n+self.q*(time+1)
        self.P_t.reverse()


        # the Q-function matrix H_t^{i}
        self.H_t=[]
        for time in range(self.batch_length):
            self.H_t.append(np.zeros([self.n+self.q+self.m,self.n+self.q+self.m]))
        self.H_t.reverse()


    def initial_buffer(self,data_length:int):

        # data structure:dimension: time length: batch length
        self.x = np.zeros((self.n, self.batch_length + 1, data_length))
        self.u = np.zeros((self.m,self.batch_length,data_length))
        self.y = np.zeros((self.q, self.batch_length+1, data_length))
        # the reference trajectory is identical
        self.y_ref = np.zeros((self.q, self.batch_length+1,data_length))
        #pdb.set_trace()
        self.data_index=0

    def initial_control_policy(self,pi):

        for time in range(self.batch_length):
            self.pi[time]=-pi[time]


    def save_data(self,x,u,y,y_ref):
        self.x[:,:,self.data_index]=x
        self.u[:,:,self.data_index]=u
        self.y[:, :, self.data_index] = y
        self.y_ref[:, :, self.data_index] = y_ref.T
        self.data_index+=1

    def save_buffer(self, num):
        # tranformate the 3D to 2D for save
        self.x = self.x.reshape(self.x.shape[0] * self.x.shape[1], self.x.shape[2])
        df_x = pd.DataFrame(self.x)
        df_x.to_csv('./Data/Buffer_x_{}_final.csv'.format(num))


        self.u = self.u.reshape(self.u.shape[0] * self.u.shape[1], self.u.shape[2])
        df_u = pd.DataFrame(self.u)
        df_u.to_csv('./Data/Buffer_u_{}_final.csv'.format(num))

        self.y = self.y.reshape(self.y.shape[0] * self.y.shape[1], self.y.shape[2])
        df_y = pd.DataFrame(self.y)
        df_y.to_csv('./Data/Buffer_y_{}_final.csv'.format(num))

        self.y_ref = self.y_ref.reshape(self.y_ref.shape[0] * self.y_ref.shape[1], self.y_ref.shape[2])
        df_y_ref = pd.DataFrame(self.y_ref)
        df_y_ref.to_csv('./Data/Buffer_y_ref_{}_final.csv'.format(num))

    def load_buffer(self,num):
        df_x = pd.read_csv('./Data/Buffer_x_{}_final.csv'.format(num), index_col=0)
        self.x = df_x.to_numpy()
        self.x = self.x.reshape(self.n,int(self.x.shape[0]/self.n), num)

        df_u = pd.read_csv('./Data/Buffer_u_{}_final.csv'.format(num), index_col=0)
        self.u = df_u.to_numpy()
        self.u = self.u.reshape(self.m, int(self.u.shape[0] / self.m), num)

        df_y = pd.read_csv('./Data/Buffer_y_{}_final.csv'.format(num), index_col=0)
        self.y = df_y.to_numpy()
        self.y = self.y.reshape(self.m, int(self.y.shape[0] / self.m), num)

        df_y_ref = pd.read_csv('./Data/Buffer_y_ref_{}_final.csv'.format(num), index_col=0)
        self.y_ref = df_y_ref.to_numpy()
        self.y_ref = self.y_ref.reshape(self.q, int(self.y_ref.shape[0] / self.q), num)


    def q_learning_iteration(self, batch_num):
        '1. consruct the x_bar,x_tilde, u'
        # x_bar= list[np.array[self.x+self.y_ref][batch_num],...,np.array[][batch_num]]
        # t= 0,1,2,3,...T-1
        x_bar = []
        # x_tilde=list[np.array[self.x+self.y+self.y_ref][batch_num],...,np.array[][batch_num]]
        # t= 1,2,3,...T
        x_tilde = []

        # u=list[np.array[m][batch_num],...,np.array[m][batch_num]]
        # t= 0,1,2,3,...T-1
        u = []
        for time in range(self.batch_length):
            x_bar.append(np.zeros([self.n+self.q, batch_num]))
            x_tilde.append(np.zeros([self.n+self.q+self.q, batch_num]))
            u.append(np.zeros([self.m, batch_num]))


        '2. assignment the x_bar,x_tilde, u'
        for time in range(self.batch_length):
            for batch in range(batch_num):
                # x_bar=[x_k,t, y_ref{t+1}]
                x_bar[time][:, batch] = np.block([self.x[:, time, batch],self.y_ref[:, time+1, batch]])
                # x_bar=[x_k,t, y_k,t, y_ref{t+1}]
                x_tilde[time][:, batch] = np.block([self.x[:, time+1, batch], self.y[:, time+1, batch],self.y_ref[:, time+1, batch]])
                # u=[self.k_t]
                u[time][:, batch] = self.u[:, time, batch]


        '3. construct the psi*L=b for lsq_linear'
        psi_t = []
        b_t = []
        psi_dim_t = int(0.5 * (x_bar[time].shape[0] + self.q) * (x_bar[time].shape[0] + self.q + 1))
        for time in range(self.batch_length):
            #psi_dim_t = int(0.5 * (x_bar[time].shape[0] + self.q) * (x_bar[time].shape[0] + self.q + 1))
            psi_t.append(np.zeros([batch_num, psi_dim_t]))
            b_t.append(np.zeros(batch_num))
        # the dimensional of H_11 H_12 H_22
        H_11_dim_t = x_bar[0].shape[0]
        H_12_dim_1_t = x_bar[0].shape[0]
        H_12_dim_2_t = u[0].shape[0]
        H_22_dim_t =  u[0].shape[0]
        # the dimensional of vec H_11 H_12 H_22
        H_11_vec_dim_t = int(0.5 * H_11_dim_t * (H_11_dim_t + 1))
        H_12_vec_dim_t = H_12_dim_1_t * H_12_dim_2_t
        H_22_vec_dim_t = int(0.5 * H_22_dim_t * (H_22_dim_t + 1))

        '3.1 psi'
        # the psi is invariant
        for time in range(self.batch_length):
            for batch in range(batch_num):
                psi1 = self.symkronecker_product(x_bar[time][:, batch])
                psi2 = 2 * self.non_symkronecker_product(x_bar[time][:, batch], u[time][:, batch])
                psi3 = self.symkronecker_product(u[time][:, batch])
                psi = np.vstack((psi1, psi2, psi3))
                psi_t[time][batch] = psi[:, 0]

        '3.2 b'

        iteration_num = self.batch_length + 1 + 10

        for iteration in range(iteration_num):

            for time in range(self.batch_length - 1, -1, -1):

                if time == (self.batch_length - 1):
                    Q_t = np.block(
                        [[np.zeros((self.n, self.n)), np.zeros((self.n, self.q)), np.zeros((self.n, self.q))],
                         [np.zeros((self.q, self.n)), self.Q, -self.Q],
                         [np.zeros((self.q, self.n)), -self.Q, self.Q]])
                else:
                    Q_t = np.block(
                        [[self.P_t[time+1][0:self.n, 0:self.n], np.zeros((self.n, self.q)), self.P_t[time+1][0:self.n, self.n:] @ self.D_t[time+1]],
                         [np.zeros((self.q, self.n)), self.Q, -self.Q],
                         [self.D_t[time+1].T @ self.P_t[time+1][self.n:, 0:self.n], -self.Q,
                          self.Q + self.D_t[time+1].T @ self.P_t[time+1][self.n:, self.n:] @ self.D_t[time+1]]])
                "b_t=x_tilde@Q_t@x_tilde"
                for batch in range(batch_num):
                    b_tem = x_tilde[time][:, batch].T @ Q_t @ x_tilde[time][:, batch] + u[time][:,batch].T @ self.R @ u[time][:,batch]

                    b_t[time][batch] = b_tem[0, 0]

                #rank = np.linalg.matrix_rank(psi_t[time])

                res = lsq_linear(psi_t[time], b_t[time], lsmr_tol='auto', verbose=1)



                "4. translate the vector into the matrix"
                H_vec_11 = res.x[0:H_11_vec_dim_t]
                H_vec_12 = res.x[H_11_vec_dim_t:(H_11_vec_dim_t + H_12_vec_dim_t)]
                H_vec_22 = res.x[(H_11_vec_dim_t + H_12_vec_dim_t):]



                H_11 = self.vector_to_symmatrix(H_vec_11, H_11_dim_t)
                H_12 = self.vector_to_non_symmatrix(H_vec_12, H_12_dim_1_t, H_12_dim_2_t)
                H_22 = self.vector_to_symmatrix(H_vec_22, H_22_dim_t)


                "4.1 construct the H_t^{i}"
                H_t = np.block([[H_11, H_12], [H_12.T, H_22]])

                self.H_t[time][:, :] = H_t[:, :]

                "5. policy improvement"
                K = np.linalg.inv(H_22) @ (H_12.T)
                "6. save the control law in the control policy"
                self.pi_improved[time] = copy.deepcopy(-K)
                "7. construct the cost function under the current control policy"
                tem = np.block([[np.eye(self.n + self.q)], [self.pi[time]]])
                self.P_t[time] = tem.T @ H_t @ tem

            self.pi = copy.deepcopy(self.pi_improved)
            self.save_K_each_iteration(iteration_num=iteration+1, batch_num=batch_num)

    def save_K(self,batch_num):

        tem=self.pi_improved[0]
        for time in range(1,len(self.pi_improved)):
            tem=np.block([[tem,self.pi_improved[time]]])
        df = pd.DataFrame(tem)
        df.to_csv('./Data/control_law_off_policy_{}.csv'.format(batch_num))


    def load_K(self,batch_num):

        df_K = pd.read_csv('./Data/policy_iteration_10/q_learning_control_policy20', index_col=0)
        tem_K=df_K.to_numpy()
        K_from_csv=[]
        tem_1=0
        tem_2=0
        for time in range(self.batch_length):
            tem_1=copy.deepcopy(tem_2)
            tem_2=tem_2+self.n+self.q
            K_from_csv.append(np.matrix(tem_K[:,tem_1:tem_2]))
        self.K=K_from_csv


    def symkronecker_product(self,vector):
        dim=vector.shape[0]
        tem = np.kron(vector, vector)
        tem = tem.reshape((dim, dim))
        tril=np.tril(tem)  # the low triangle of matrix
        triu=np.triu(tem)  # the upper triangle of matrix
        tem2=tem-triu+tril
        kronecked_product=np.zeros((int(0.5*dim*(dim+1)),1))
        num=0
        for column in range(dim):

            for row in range(dim):
                if row>=column:
                    kronecked_product[num,0]=tem2[row,column]
                    num=num+1
        return kronecked_product
    def non_symkronecker_product(self,vector1,vector2):
        dim1=vector1.shape[0]
        dim2=vector2.shape[0]
        tem = np.kron(vector2, vector1)
        kronecked_product=tem.reshape((dim1*dim2,1))
        return kronecked_product

    def vector_to_symmatrix(self,vector,dim):
        tem=np.zeros((dim,dim))
        num=0
        for column in range(dim):
            for row in range(dim):
                if row>=column:
                    tem[row,column]=vector[num]
                    num=num+1
        tem2=tem+tem.T
        tem3=np.diagflat(np.diag(tem))
        A_t=tem2-tem3
        return A_t
    def vector_to_non_symmatrix(self,vector,dim1,dim2):
        tem=np.zeros((dim1,dim2))
        num=0
        for column in range(dim2):
            for row in range(dim1):
                tem[row,column]=vector[num]
                num=num+1
        return tem

    def save_K_each_iteration(self,iteration_num,batch_num):
        config_path = os.path.split(os.path.abspath(__file__))[0]
        config_path = config_path.rsplit('/', 1)[0]

        '1. save the K'
        K_tem=-self.pi[0]
        for time in range(1,self.batch_length):
            K_tem=np.block([[K_tem,-self.pi[time]]])
        df = pd.DataFrame(K_tem)

        dir_path = os.path.join(config_path, "Data/policy_iteration_{}_final".format(batch_num))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data_path = os.path.join(dir_path, "q_learning_control_policy{}_final".format(iteration_num))
        df.to_csv(data_path)


