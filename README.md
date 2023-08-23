# Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight

Paper: [Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](http://arxiv.org/abs/2305.02772)

## video
[![Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](https://res.cloudinary.com/marcomontalbano/image/upload/v1683187170/video_to_markdown/images/youtube--E6QVHWcvB6E-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/E6QVHWcvB6E "Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight")

## Instructions
1. Install ros noetic
2. Setup the ros workspace: `~/fast_fly_ws`
3. Install CasADi python package: `pip3 install casadi`
4. Clone this repository into `~/fast_fly_ws/src`
5. Clone the repository `https://github.com/ZhouZiyuBIT/px4_bridge.git` into `~/fast_fly_ws/src` for simulation
6. Compile: run `catkin_make` in `~/fast_fly_ws`
7. Run the example with `roslaunch fast_fly fast_fly_sim.launch`

This will run the time-optimal planning and tracking in the simulation:
![](fig/sim.jpg)
