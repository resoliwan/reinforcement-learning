import numpy as np
import sys
if not "./" in sys.path:
    sys.path.append("./")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment dynamic's.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents transition probability of environment.
        discount_factor: lamdba discount_factor.
        theta: We stop evalution one our value function changes less than theta for all states.
    Returns:
        Vector of lengh env.nS representing value function.
    """
	# Start with all 0 value function.	
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at possible next_state...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Caculate expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across nay state)
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        # Stop evaluting once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

# random_policy = np.ones([env.nS, env.nA]) / env.nA
# v = policy_eval(random_policy, env)
# print('v', v)
#
# expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

def policy_improvment(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Iteratively evaluate and imporve a policy until
    optimal policy is found.
    Args:
        env: OpanAI.env.
        policy_eval_fn: Policy evalution function that takes 3 arguments:
            policy, env, discount_factor.

    Returns:
        A tuple (policy, V)
    """
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = None
    while True:
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        V = policy_eval_fn(policy, env, discount_factor)
        
        # For each state..
        for s in range(env.nS):
            # The best action we would take under the policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead.
            # Ties are resolved arbitrarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])

            best_a = np.argmax(action_values)
            
            # Greedly update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        # It the policy stable we've found an optimal policy. Return it.
        if policy_stable:
            break

    return policy, V

policy, v = policy_improvment(env)
print('policy', policy)
print(np.argmax(policy, axis=1))
print('Reshaped Grid Value Function (0=up, 1=right, 2=down, 3=left):')
print(np.reshape(np.argmax(policy, axis=1), env.shape))

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
