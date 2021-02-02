import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class HieNoiseAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HieNoiseAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.latent_dim = args.latent_dim
        self.noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)
        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        self.embed_fc_input_size = input_shape * args.n_agents
        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                    #    nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       nn.LeakyReLU(),
                                       nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, noise, test_mode=False):
        agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(noise.shape[0], 1)
        noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)
        inputs_all_agents = inputs.reshape(-1, self.n_agents * self.input_shape)
        hierarchy = self.embed_net(inputs_all_agents)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.args.n_actions, self.args.rnn_hidden_dim)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz
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
        return wq, h, skill, log_p
