# reinforcement_learning

DQN --> Deep Q-Network

An agent trains itself to solve the classic control environment 'CartPole', imported from the OpenAI 'gym' toolkit: https://gym.openai.com/envs/CartPole-v1/.

The program uses the 'PyTorch' machine learning library to implement a Deep Q-learning architecture: a simple neural network with ReLU activations (referred to as the 'policy
network') is used as a function approximator for the action-value function of a given input state, while a second network with frozen parameters (the 'target' network) is used to
generate corresponding Q-targets. The internal parameters of the policy network are incrementally updated using the Adam optimisation algorithm, incorporatating batch sampling of
transition tuples from a replay memory buffer in order to decorrelate successive network updates. The frozen parameters of the target network are periodically synchronised with
those of the policy network after a specified number of episodes have terminated. 

The included PNG file shows a typical performance plot for the agent, generated from a relatively small number of episodes. The blue line represents the per-episode score, while
the orange line represents the moving average score over a 100 episode period. Although the agent is exhibiting improved performance over time, the control problem is
not considered to be 'solved' until the agent can demonstrate a minimum moving average score of 195 over a period of 100 consecutive episodes; by modifiying the structure of the
neural networks and tuning the program hyperparameters, this goal is attainable in fewer episodes than the example learning trajectory implies.    
