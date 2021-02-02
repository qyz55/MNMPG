import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.distributions as D
from torch.distributions import kl_divergence

class HieRodeAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HieRodeAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0

        
        self.embed_fc_input_size = input_shape * args.n_agents

        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func=nn.LeakyReLU()
        
        #hierachical net, use obs of all agents as input.
        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                    #    nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state, test_mode=False, t_glob=0):
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        inputs_all_agents = inputs.reshape(-1, self.n_agents * self.input_shape)
        inputs_single_agent = inputs.reshape(-1, self.input_shape)
        hierarchy = self.embed_net(inputs_all_agents)
        x = F.relu(self.fc1(inputs_single_agent))
        h = self.rnn(x, h_in)
            
        hierarchy[:, -self.latent_dim:] = th.clamp(th.exp(hierarchy[:, -self.latent_dim:]), min=self.args.var_floor)
        gaussian_embed = D.Normal(hierarchy[:, :self.latent_dim], (hierarchy[:, self.latent_dim:]) ** (1 / 2))
        if test_mode:
            skill = gaussian_embed.mean
            log_p = gaussian_embed.log_prob(skill.detach())
        elif self.args.hie_grad_qtot:
            skill = gaussian_embed.rsample()
            log_p = gaussian_embed.log_prob(skill.detach())
        else:
            skill = gaussian_embed.sample()
            log_p = gaussian_embed.log_prob(skill)
        return h, skill, log_p
