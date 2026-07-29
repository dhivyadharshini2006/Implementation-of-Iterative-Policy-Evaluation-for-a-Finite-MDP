# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
!pip install gymnasium
import gymnasium as gym
import numpy as np
# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
# Access the unwrapped environment to use the transition model
env = env.unwrapped
# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n
# Parameters
gamma = 0.99
theta = 1e-8
# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions
V = np.zeros(n_states)
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    iteration = 0
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            new_v = 0
            # Sum over all possible actions from state s
            for a, action_prob in enumerate(policy[s]):
                # Sum over all possible next states s' given action a from state s
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (reward + gamma * V[next_state])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        iteration += 1
        if delta < theta:
            break
    return V, iteration

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)
print("Name:Dhivya Dharshini B             ")
print("Register Number:212223240031   ")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name:Dhivya Dharshini B               ")
print("Register Number:212223240031    ")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

---

## Output

### Number of Iterations:

<img width="530" height="161" alt="image" src="https://github.com/user-attachments/assets/dec1226e-5043-4a8a-89d0-5675a6c8b4ff" />


### State-Value Function as 4x4 Grid:

<img width="306" height="157" alt="image" src="https://github.com/user-attachments/assets/a42828fd-43fc-4d78-90f4-60cc5175c7c4" />


---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference


The iterative policy evaluation algorithm successfully estimated the state-value function for the FrozenLake-v1 environment under a fixed random policy. By repeatedly applying the Bellman expectation equation, the value function converged after a finite number of iterations. The computed state values represent the expected long-term return from each state when the agent follows the random policy. States that are closer to the goal or have a higher probability of reaching the goal obtained larger state values, whereas hole states and terminal states retained a value of zero. This experiment demonstrates how iterative policy evaluation computes the expected utility of each state without improving or changing the policy, providing a foundation for reinforcement learning methods such as policy iteration and value iteration.






---


