import numpy as np
import pprint
import sys
if "./" not in sys.path:
    sys.path.append(".")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
shape = [4,4]
env = GridworldEnv(shape)
# env.render()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evalueate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents transition probability of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation one our value function changes is less than theta for all states.
        dicount_factor: lambda discount_factor

    Returns:
        Vector of length env.nS representing the value function.
    """
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in np.arange(env.nS):
            v = 0
            for a, acition_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += acition_prob * prob * (reward + discount_factor * V[next_state])
            delta = np.max([delta, np.absolute(v - V[s])])
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
