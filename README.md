# FastSLAM
Python simulation of FastSLAM

## Intall Dependencies
Using a new virtual env to install the packages:
`pip install -r requirements.txt`

## Run Simulation
1. Run FastSLAM 1.0
`python fast_slam.py`

2. Run FastSLAM 2.0

## Control
Using arrow keys to control the robot, you can set number of steps in `fast_slam.py`.

## Sensor
Currently, there are 4 landmarks in the world. You can add more landmarks in the `world.py` by modifying `setup_world` method. The coordinates are using the bottom-left corner point as the origin.

In the `sense` method in the `particle.py`, the robot randomly observe 2 landmarks and measure the distance and the direction to the landmarks. Then it adds the Gaussian noise to the measurements. The noise level is set up in the `set_noise` method. Only robot has the `bearing_noise`:measurement errors for the angles, and `distance_noise`: measurement errors for the distance. You can also set the motion noise for the robot and particles.

The `obs_noise` is the additive part of the prediction step of the EKF. First term specifies the error for distance and the second term specify the error for angles. Larger the value, more relax the model will be when considering the data association. `obs_noise` should be at the same magnitude as `distance_noise` and `bearing_noise`.

The `control_noise` attribute model the motion noise. First two terms specify the error for the x, y coordinates and the third term for the orientation.

## Souce
[fastSLAM paper](https://www.ri.cmu.edu/pub_files/pub4/montemerlo_michael_2003_1/montemerlo_michael_2003_1.pdf)
