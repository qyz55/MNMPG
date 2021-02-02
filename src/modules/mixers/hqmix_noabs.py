import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HQMixerFF(nn.Module):
    def __init__(self, args):
        super(HQMixerFF, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        activation_func=nn.LeakyReLU()
        self.latent_dim = args.latent_dim
        self.state_dim = int(np.prod(args.state_shape))
        NN_HIDDEN_SIZE = self.state_dim
        self.embed_dim = args.central_mixing_embed_dim
        self.mid_dim = args.mid_dim
        if args.concat_ori_s:
            HIDDEN_ALL = self.state_dim + NN_HIDDEN_SIZE + self.n_agents
        else:
            HIDDEN_ALL = NN_HIDDEN_SIZE + self.n_agents
        self.net = nn.Sequential(nn.Linear(HIDDEN_ALL, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, 1))
        self.V = nn.Sequential(nn.Linear(HIDDEN_ALL - self.n_agents, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

                    #(bs,t,n),
    def forward(self, agent_qs, states, skill): #skill:(bs,t,latent_dim) state:(bs,t,all_obs)
        bs = agent_qs.size(0)
        r_s = skill.reshape(-1, self.latent_dim)#(bs,t,latent_dim)
        r_s = self.latent_net(r_s) #(bs*t, NN_HIDDEN_SIZE)
        states = states.reshape(-1, self.state_dim) #(bs*t, all_obs)
        agent_qs = agent_qs.reshape(-1, self.n_agents) #(bs*t, n)
        # First layer
        if self.args.concat_ori_s:
            input = th.cat([states, r_s, agent_qs], dim=1)
        else:
            input = th.cat([r_s, agent_qs], dim=1)
        advs = self.net(input)
        # State-dependent bias
        vs = self.V(th.cat([r_s, states], dim=1))
        y = advs + vs
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot #(bs,t,1)
