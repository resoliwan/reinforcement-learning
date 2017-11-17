import numpy as np
import sys
if not "./" in sys.path:
    sys.path.append("./")

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def value_iteration(env, discount_factor=1, theta=0.00001):
    """
    Value function evalution is stoped after only one sweep 
    and take max action until it less than theta for all states
    Combine a policy evaluation and a policy improvment.

    Args:
        env: OpeaAI.env.
        discount_factor: lambda discount_factor.
        theta: we stop value iteration when difference 
            between value function less than theta for all states.
    Returns:
        V: Optimal value function.
    """
    V = np.zeros(env.nS)
    while True:
        break

    return V

v = value_iteration(env)

print('v', v)



