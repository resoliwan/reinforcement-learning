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
	# Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
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
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal poilcy, Return it.
        if policy_stable:
            return policy, V


def my_policy_improvment(env, policy_eval_fn=policy_eval, discount_factor=1.0):
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
            new_policy = optimal_actions / env.nA

            if not np.array_equal(old_policy, new_policy):
                policy[s] = new_policy
                isChanged = True

        if isChanged == False:
            break
    
        # print('V', V)
    return policy, V
policy, v = policy_improvment(env)
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

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
