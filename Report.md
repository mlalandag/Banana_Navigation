
# Report

## Learning Algorithm



  
### Model

* The Neural Network has two Hidden layers with 128 and 64 neurons each.
* The activation function used is `ReLU`
* The output layer has as many neurons as number of possible actions the agent can choose, which is 4

### Chosen hyperparameters

* Learning rate (LR): `LR = 1e-2`

* Epsilon-greedy policy with decay: Epsilon is initialized to `1.0` and then starts decaying with factor of `0.98`. When it reaches the minimum of `0.005` it remains with this value for the rest of the training

* Discout Factor Gamma: The chosen value for the Discount Factor gamma is`0.99`

* `BATCH_SIZE = 64`

* `BUFFER_SIZE = 10.000`

* `TAU = 1e-3`


### Plot of Rewards

    ![Plot][image2]

### Ideas for Future Work

In order to improve the efficiency of the agent order algorithms can be tried. Those are:

* **Double DQN **
* **Prioritized experience replay**
* **Dueling DQN**


