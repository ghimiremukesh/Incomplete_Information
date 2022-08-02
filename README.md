# Incomplete Information Game 

## A Simple Soccer Game 

<img width="776" alt="image" src="https://user-images.githubusercontent.com/25011477/177612900-032e3a06-55f0-4eab-b9ba-62eaafcfa80d.png">

We consider a simple soccer game as shown in figure 1. The attacker tries to avoid defender by maximizing its distance. 

### Environment
To install the environment run: ``conda env create -f env.yml``

### Repo Structure 

- modules_picnn.py contains the network architecture (PICNN)
- dataio.py loads the training data
- training.py contains the training loop
- utils.py has utility functions
- loss_functions.py specifies losses for experiments
- diff_operators.py has functions for differential operations
- training_curve_plot.py plots loss over epochs
- ./experiment_scripts/ has scripts to run the experiments
- ./validation_scripts/ has scripts to evaluate the trained models

### Training

To train the model, run experiment_scripts/train_soccer_hji.py


### Simulation

To generate the trajectory, run validation_scripts/soccer_trajectory.py


### Credits
- The network architecture (PICNN) is based on the [ICNN](https://github.com/locuslab/icnn) repo.
- Curriculum learning and code flow based on [Deepreach](https://github.com/smlbansal/deepreach/) repo. 

