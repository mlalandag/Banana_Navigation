[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the root folder of the repo and decompress it. 

## Installation

1. Download or clone this GitHub repository.

2. Download and install Anaconda Python 3.6 from the following link : https://www.anaconda.com/download/

2. Create a new conda environment named drlnd (or whatever name you prefer) and then activate it:

	- Linux or Mac:
	
		`conda create --name drlnd python=3.6`
	
		`source activate drnld`

	- Windows:
	
		`conda create --name drnld python=3.6`
	
		`activate drnld`

4. Install the required dependencies navigating to where you downloaded and saved this GitHub repository and then into the '.python/' subdirectory. Then run from the command line:
	
		`pip3 install .`
 
## Files

- agent.py: Contains the agent who interacts with the environment and is used to train the model. 
- model.py: Contains the Neural Network implemented in Pytorch that is used to pick the actions. 
- replay_buffer.py: Helper class to implement the Esperience Replay algorithm.
- agent_training.py: Process that delivers the trained model. 
- agent_test.py: Execution of some episodes with the agent using the trained model. 

## Training

 - Go to the root folder of the repo and run:
 
 	`python agent_training.py`
	
 - When the score reaches the value +13 it will stop and save the model weights to the file .

## Testing

 - To test the trained agent:
 
 	`python agent_test.py`
	
