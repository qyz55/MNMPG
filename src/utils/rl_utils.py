import torch as th
import numpy as np
import os.path as osp
import os
import pickle


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def get_agent_order(agent_order):
    order_ally = agent_order[0]
    order_enemy = agent_order[1]
    ally_num = len(order_ally)
    enemy_num = len(order_enemy)
    ally_st = enemy_num * 8+4
    enemy_st = 4
    order_ally = order_ally * 8 + ally_st
    order_enemy = order_enemy * 8 + enemy_st
    ally_ind = np.repeat(order_ally, 8) + np.tile(np.arange(8), ally_num).flatten()
    enemy_ind = np.repeat(order_enemy, 8) + np.tile(np.arange(8), enemy_num).flatten()
    o_st = (enemy_num+ally_num)*8 + 4
    return np.concatenate([np.arange(4),enemy_ind,ally_ind,np.arange(o_st, o_st+4)])

def save_batch(batch, loc, index, t_env):
    os.makedirs(loc, exist_ok=True)
    f_name = osp.join(loc, "batch"+str(index)+"_"+str(t_env)+".pkl")
    with open(f_name, "wb") as f:
        pickle.dump(batch, f)

def save_q(q, loc, index):
    os.makedirs(loc, exist_ok=True)
    f_name = osp.join(loc, "q_"+str(index)+".pkl")
    with open(f_name, "wb") as f:
        pickle.dump(q, f)

    
