# reinforcement_learning

DQN --> Deep Q-Network

An agent trains itself to solve the classic control environment 'CartPole', imported from the OpenAI 'gym' toolkit: https://gym.openai.com/envs/CartPole-v1/.

The program uses the 'PyTorch' machine learning library to implement a Deep Q-learning architecture: a simple neural network with ReLU activations (referred to as the 'policy
network') is used as a function approximator for the action-value function of a given input state, while a second network with frozen parameters (the 'target' network) is used to
generate corresponding Q-targets. The internal parameters of the policy network are incrementally updated using the Adam optimisation algorithm, incorporatating batch sampling of
transition tuples from a replay memory buffer in order to decorrelate successive network updates. The frozen parameters of the target network are periodically synchronised with
those of the policy network after a specified number of episodes have terminated.   
