import numpy as np
import pprint
import sys
if "./" not in sys.path:
    sys.path.append(".")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
shape = [4,4]
env = GridworldEnv(shape)
env.render()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in np.arange(env.nS):
            v = V[s]
            tempA = 0
            for a in np.arange(env.nA):
                tempB = 0
                for p, next_s, r, done in env.P[s][a]:
                    tempB += p * (r + discount_factor * V[next_s])
                tempA += policy[s][a] * tempB
            V[s] = tempA
            delta = np.max([delta, np.absolute(v - V[s])])
        #     print('s, delta, theta', s, delta, theta)
        # print('V', V.reshape(shape))
        if delta < theta:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
print(policy_eval(random_policy, env).reshape(shape))


