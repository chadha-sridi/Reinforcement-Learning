# Reinforcement-Learning
Training RL agents in some classic gym environments using PPO algorithm.
* Cartpole
* Mountain Car
* Montezuma Revenge
    * The two csv files show cumulative rewards obtained by the agent during testing throughout 1000 episodes.
    * Even though during training, the agent with 0.01 entropy coefficient manages to explore the state space and accumulate rewards faster than the agent with 0 entropy coefficient, the test results show that the agent with 0 entropy coefficient does a way better job and manages to accumulate rewards.
    * The other agent was not able to learn an efficient policy and was maybe disturbed by the incite for more exploration. However this is not a general conclusion as the entropy coefficient parameter needs to be more thouroughly finetuned in order to conclude about its effect.
