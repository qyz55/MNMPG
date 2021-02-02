import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.distributions as D
from torch.distributions import kl_divergence

class HieRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HieRNNAgent, self).__init__()
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
        if args.use_roma:
            self.roma_embed_fc_input_size = input_shape
            self.roma_embed_net = nn.Sequential(nn.Linear(self.roma_embed_fc_input_size, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func,
                                        nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

            self.roma_inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + input_shape, NN_HIDDEN_SIZE),
                                            nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                            activation_func,
                                            nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

            self.roma_latent = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)
            self.roma_latent_infer = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)

            self.roma_latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
                                            nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                            activation_func)
            self.roma_fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE, args.rnn_hidden_dim * args.n_actions)
            self.roma_fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, args.n_actions)

            # Dis Net
            self.roma_dis_net = nn.Sequential(nn.Linear(args.latent_dim * 2, NN_HIDDEN_SIZE ),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE ),
                                        activation_func,
                                        nn.Linear(NN_HIDDEN_SIZE , 1))

            self.roma_mi= th.rand(args.n_agents*args.n_agents)
            self.roma_dissimilarity = th.rand(args.n_agents*args.n_agents)

            if args.dis_sigmoid:
                print('>>> sigmoid')
                self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
            else:
                self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step
        else:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    
    def init_latent(self, bs):
        self.bs = bs
        if self.args.use_roma:
            var_mean=self.roma_latent[:self.n_agents, self.args.latent_dim:].detach().mean()
            mi = self.roma_mi
            di = self.roma_dissimilarity
            indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),di.max(),di.min(),di.mean(),di.std()]
            return indicator, self.roma_latent[:self.n_agents, :].detach(), self.roma_latent_infer[:self.n_agents, :].detach()


    def forward(self, inputs, hidden_state, test_mode=False, t_glob=0):
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        inputs_all_agents = inputs.reshape(-1, self.n_agents * self.input_shape)
        inputs_single_agent = inputs.reshape(-1, self.input_shape)
        hierarchy = self.embed_net(inputs_all_agents)
        x = F.relu(self.fc1(inputs_single_agent))
        h = self.rnn(x, h_in)
        c_dis_loss = th.tensor(0.0).to(self.args.device)
        ce_loss = th.tensor(0.0).to(self.args.device)
        loss = th.tensor(0.0).to(self.args.device)
        if self.args.use_roma:
            embed_fc_input = inputs_single_agent[:, - self.roma_embed_fc_input_size:]
            self.roma_latent = self.roma_embed_net(embed_fc_input)
            self.roma_latent[:, -self.latent_dim:] = th.clamp(th.exp(self.roma_latent[:, -self.latent_dim:]), min=self.args.var_floor)  # var
            roma_latent_embed = self.roma_latent.reshape(self.bs * self.n_agents, self.latent_dim * 2)

            roma_gaussian_embed = D.Normal(roma_latent_embed[:, :self.latent_dim], (roma_latent_embed[:, self.latent_dim:]) ** (1 / 2))
            roma_latent = roma_gaussian_embed.rsample()

            
            if (not test_mode) and (not self.args.roma_raw):
                self.roma_latent_infer = self.roma_inference_net(th.cat([h_in.detach(), inputs_single_agent], dim=1))
                self.roma_latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.roma_latent_infer[:, -self.latent_dim:]),min=self.args.var_floor)
                roma_gaussian_infer = D.Normal(self.roma_latent_infer[:, :self.latent_dim], (self.roma_latent_infer[:, self.latent_dim:]) ** (1 / 2))
                roma_latent_infer = roma_gaussian_infer.rsample()
                loss = roma_gaussian_embed.entropy().sum(dim=-1).mean() * self.args.h_loss_weight + kl_divergence(roma_gaussian_embed, roma_gaussian_infer).sum(dim=-1).mean() * self.args.kl_loss_weight   # CE = H + KL
                loss = th.clamp(loss, max=2e3)
                ce_loss = th.log(1 + th.exp(loss))
                # Dis Loss
                cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
                if cur_dis_loss_weight > 0:
                    dis_loss = 0
                    dissimilarity_cat = None
                    mi_cat = None
                    roma_latent_dis = roma_latent.clone().view(self.bs, self.n_agents, -1)
                    roma_latent_move = roma_latent.clone().view(self.bs, self.n_agents, -1)
                    for agent_i in range(self.n_agents):
                        roma_latent_move = th.cat(
                            [roma_latent_move[:, -1, :].unsqueeze(1), roma_latent_move[:, :-1, :]], dim=1)
                        roma_latent_dis_pair = th.cat([roma_latent_dis[:, :, :self.latent_dim],
                                                roma_latent_move[:, :, :self.latent_dim],
                                                # (latent_dis[:, :, :self.latent_dim]-latent_move[:, :, :self.latent_dim])**2
                                                ], dim=2)
                        mi = th.clamp(roma_gaussian_embed.log_prob(roma_latent_move.view(self.bs * self.n_agents, -1))+13.9, min=-13.9).sum(dim=1,keepdim=True) / self.latent_dim

                        dissimilarity = th.abs(self.roma_dis_net(roma_latent_dis_pair.view(-1, 2 * self.latent_dim)))

                        if dissimilarity_cat is None:
                            dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                        else:
                            dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)
                        if mi_cat is None:
                            mi_cat = mi.view(self.bs, -1).clone()
                        else:
                            mi_cat = th.cat([mi_cat,mi.view(self.bs,-1)],dim=1)

                        #dis_loss -= th.clamp(mi / 100 + dissimilarity, max=0.18).sum() / self.bs / self.n_agents

                    mi_min=mi_cat.min(dim=1,keepdim=True)[0]
                    mi_max=mi_cat.max(dim=1,keepdim=True)[0]
                    di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
                    di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

                    mi_cat=(mi_cat-mi_min)/(mi_max-mi_min+ 1e-12 )
                    dissimilarity_cat=(dissimilarity_cat-di_min)/(di_max-di_min+ 1e-12 )

                    dis_loss = - th.clamp(mi_cat+dissimilarity_cat, max=1.0).sum()/self.bs/self.n_agents
                    #dis_loss = ((mi_cat + dissimilarity_cat - 1.0 )**2).sum() / self.bs / self.n_agents
                    dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / self.bs / self.n_agents

                    #c_dis_loss = (dis_loss + dis_norm) / self.n_agents * cur_dis_loss_weight
                    c_dis_loss = (dis_norm + self.args.soft_constraint_weight * dis_loss) / self.n_agents * cur_dis_loss_weight
                    loss = ce_loss +  c_dis_loss

                    self.roma_mi = mi_cat[0]
                    self.roma_dissimilarity = dissimilarity_cat[0]
                else:
                    c_dis_loss = th.zeros_like(loss)
                    loss = ce_loss
            roma_latent = self.roma_latent_net(roma_latent)
            fc2_w = self.roma_fc2_w_nn(roma_latent)
            fc2_b = self.roma_fc2_b_nn(roma_latent)
            fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
            fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))
            h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
            qs = th.bmm(h, fc2_w) + fc2_b
            h = h.reshape(-1, self.args.rnn_hidden_dim)
            qs = qs.reshape(-1, self.args.n_actions)
        else:
            qs = self.fc2(h)
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
        return qs, h, skill, log_p, loss, c_dis_loss, ce_loss

    def dis_loss_weight_schedule_step(self, t_glob):
        if t_glob > self.args.dis_time:
            return self.args.dis_loss_weight
        else:
            return 0

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return self.args.dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))



        
