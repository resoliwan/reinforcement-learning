import numpy as np
import pprint
import sys
import math
if "./" not in sys.path:
    sys.path.append("./")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

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
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

# random_policy = np.ones([env.nS, env.nA]) / env.nA
# v = policy_eval(random_policy, env)
# print('v', v)

# expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

def policy_improvment(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvment Algorithm. Iteratively evalutes and improves a policy
    until an optimal policy is found.

    Args:
        env: OpenAI env.
        policy_eval_fn: Policy evalution function that take 3 arguments:
            policy, env, discount_factor
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V)
        policy is the optimal policy, a matrix of [S, A] where each stats s contains
        a valid probility distirbution over actions.
        V is the value function for the optimal policy
    """
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = None
    while True:
        print('@isChanged')
        isChanged = False
        V = policy_eval(policy, env)
        for s in range(env.nS):
            qs = []
            max_qvalue = -math.inf
            old_policy = np.array(policy[s])
            for a, _ in enumerate(policy[s]):
                qvalue = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    qvalue  += prob * (reward + discount_factor * V[next_state])

                max_qvalue = qvalue if max_qvalue < qvalue else max_qvalue
                qs.append(qvalue)

            optimal_actions = np.array(qs) >= max_qvalue
            policy[s] = optimal_actions / env.nA

            if not np.array_equal(old_policy, policy[s]):
                isChanged = True

        if isChanged == False:
            break

    return policy, V

policy, v = policy_improvment(env)
print('policy', policy)

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
