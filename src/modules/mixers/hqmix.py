import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HQMixer(nn.Module):
    def __init__(self, args):
        super(HQMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        activation_func=nn.LeakyReLU()
        self.latent_dim = args.latent_dim
        self.state_dim = int(np.prod(args.state_shape))
        if args.mac == "hierarchical_noise_mac":
            self.state_dim += args.noise_dim
        NN_HIDDEN_SIZE = self.state_dim
        self.embed_dim = args.mixing_embed_dim
        if args.concat_ori_s:
            HIDDEN_ALL = self.state_dim + NN_HIDDEN_SIZE
        else:
            HIDDEN_ALL = NN_HIDDEN_SIZE
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(HIDDEN_ALL, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(HIDDEN_ALL, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(HIDDEN_ALL, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(HIDDEN_ALL, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(HIDDEN_ALL, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(HIDDEN_ALL, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

                    #(bs,t,n),
    def forward(self, agent_qs, states, skill, noise=None): #skill:(bs,t,latent_dim) state:(bs,t,all_obs)
        bs = agent_qs.size(0)
        r_s = skill.reshape(-1, self.latent_dim)#(bs,t,latent_dim)
        r_s = self.latent_net(r_s) #(bs*t, NN_HIDDEN_SIZE)
        if self.args.mac == "hierarchical_noise_mac":
            states = states.reshape(-1, self.state_dim - self.args.noise_dim) #(bs*t, all_obs)
            noise = noise.reshape(-1, self.args.noise_dim)
            states = th.cat([states, noise], dim=-1)
        else:
            states = states.reshape(-1, self.state_dim) #(bs*t, all_obs)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) #(bs*t, 1, n)
        # First layer
        if self.args.concat_ori_s:
            input = th.cat([states, r_s], dim=1)
        else:
            input = r_s
        w1 = th.abs(self.hyper_w_1(input)) #(bs*t,n*embed_dim)
        b1 = self.hyper_b_1(input) #(bs*t, embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim) #(bs*t,n,embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1) #(bs*t, 1,embed_dim)

        # Second layer
        w_final = th.abs(self.hyper_w_final(input)) #(bs*t,embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1) #(bs*t,embed_dim,1)
        # State-dependent bias
        v = self.V(input).view(-1, 1, 1) #(bs*t,1,1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot #(bs,t,1)
