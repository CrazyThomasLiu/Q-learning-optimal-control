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

## Calculation of the 2D ICL controller
Run the following command to obtain the control law of the 2D ILC Controller.
```
cd ILC_Controller
python controllaw_injection_molding_process.py or controllaw_nonlinear_batch_reactor.py 
```

## Training of the 2D DRL compensator
Run the following command to train the 2D DRL Compensator.
```
cd DRL_Compensator
python demo_nominal_injection_molding_process.py or demo_nominal_nonlinear_batch_reactor.py 
python demo_practical_injection_molding_process.py or demo_nominal_nonlinear_batch_reactor.py 
```
The training usually takes 4 hours for the injection_molding_process and 6 hours for the nonlinear batch reactor.


## Test for the control performance simulation 
Run the following command to test the final control performance.
* Injection Molding Process

```
cd Trained_2D_ILC_RL_Controller/Injection_Molding_Process
python demo_injection_molding_process.py
```
![image](https://github.com/CrazyThomasLiu/2dilc-rl/raw/master/Trained_2D_ILC_RL_Controller/Injection_Molding_Process/Injection_molding_output.jpg)


* Nonlinear Batch Reactor

```
cd Trained_2D_ILC_RL_Controller/Nonlinear_Batch_Reactor
python demo_nonlinear_batch_reactor.py
```

![image](https://github.com/CrazyThomasLiu/2dilc-rl/raw/master/Trained_2D_ILC_RL_Controller/Nonlinear_Batch_Reactor/Nonlinear_batch_reactor_output.jpg)






