# qoc: Q-learning-based Optimal Control Scheme for the Time-varying Batch Processes with Time-varying Uncertainty


## Catalog

* Data: all the data and figures for the paper. 
* env: the time-varying batch processes with time-varying uncertainty.
* algorithm:  Two optimal control scheme for the batch systems: the model-based optimal control scheme and the Q-learning-based data-driven optimal control scheme.
* comparison_algorithm:  The comparison control scheme: PI-based indirect-type ILC from paper 'PI based indirect-type iterative learning control for batch processes with time-varying uncertainties: A 2D FM model based approach' Journal of process control,2019

## Getting Started
* Clone this repo: `git clone https://github.com/CrazyThomasLiu/Q-learning_optimal_control.git`
* Create a python virtual environment and activate. `conda create -n qoc python=3.10` and `conda activate qoc`
* Install dependenices. `cd qoc`, `pip install -r requirement.txt` 

## Calculation of the model-based initial controller and optimal controller
Run the following command to obtain the initial control policy and optimal control policy based on the model.
```
python demo_mbocs_final.py
```
## Getting the Q-learning-based optimal control scheme
Run the following command to sample the data and iteratively compute the Q-learning-based optimal control policy.
```
python demo_Q_learning_final.py
```
## Getting RMSEs for the comparison method.
Run the following command to compute the PI-based indirect-type iterative learning control law.
```
cd comparison_algorithm
python pi_controller_final.py
python robust_pi_controller_final.py
```
Run the following command to get the RMSEs data.
```
python demo_pi_robust_final.py
```

## Test for the control performance
Run the following command to show the sample data figure.
```
python demo_sample_3D_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Q_learing.jpg)

Run the following command to illustrate the convergence of Q-learning.
```
python demo_compare_pi_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Q_learing.jpg)



Run the following command to compare the control performance between the initial control policy and the Q-learning-based optimal control policy.
```
python demo_test_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Q_learing.jpg)




Run the following command to compare the RMSEs.
```
python demo_compare_RMSE_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Compare_RMSE_final.pdf)
