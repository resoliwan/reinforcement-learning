import numpy as np
import sys
if not "./" in sys.path:
    sys.path.append("./")

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1):
    """
    Value function evalution is stoped after only one sweep 
    and take max action until it less than theta for all states
    Combine a policy evaluation and a policy improvment.

    Args:
        env: OpeaAI.env.
        discount_factor: lambda time discount_factor.
        theta: Stopping threshold. If the value of all states changes less than theta 
			in one iteration we are done.
    Returns:
		A tuple (policy, V) of optimal policy and the optimal value function.
    """
    def q(state, action, env, discount_factor, value_fn):
        q = 0
        for prob, next_state, reward, done in env.P[state][action]:
            q += prob * (reward + discount_factor * value_fn[next_state])
        return q

    V = np.zeros(env.nS)
    policy = np.zeros((env.nS, env.nA))
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = np.max([q(s, a, env, discount_factor, V) for a in range(env.nA)])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break;
    
        for s in range(env.nS):
            best_a = np.argmax([q(s, a, env, discount_factor, V) for a in range(env.nA)])
            policy[s] = np.eye(env.nA)[best_a]

    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
