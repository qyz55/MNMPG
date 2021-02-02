import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.hqmix import HQMixer
import torch as th
from torch.optim import RMSprop
import numpy as np


class HierarchicalNoiseQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "hqmix":
                self.mixer = HQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        discrim_input = np.prod(self.args.state_shape) + self.args.n_agents * self.args.n_actions

        if self.args.rnn_discrim:
            self.rnn_agg = RNNAggregator(discrim_input, args)
            self.discrim = Discrim(args.rnn_agg_size, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
            self.params += list(self.rnn_agg.parameters())
        else:
            self.discrim = Discrim(discrim_input, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
        self.meta_params = list(self.mac.parameters())
        self.discrim_loss = th.nn.CrossEntropyLoss(reduction="none")

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.meta_optimiser = RMSprop(params=self.meta_params, lr=args.meta_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, chosen_index=0, return_q_all=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        noise = batch["noise"][:, 0].unsqueeze(1).repeat(1,rewards.shape[1],1)

        # Calculate estimated Q-Values
        mac_out = []
        skill_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, skills, _ = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            skill_out.append(skills)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        skill_out = th.stack(skill_out, dim=1)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_skill_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_skills, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_skill_out.append(target_skills)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_skill_out = th.stack(target_skill_out, dim=1)
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if return_q_all:
                q_all = chosen_action_qvals.detach().cpu().numpy()
                skill_all = skill_out[:,:-1].detach().cpu().numpy()
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], skill_out[:,:-1], noise=noise)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_skill_out[:,1:], noise=noise)
            if return_q_all:
                mix_q_all = chosen_action_qvals.detach().cpu().numpy()
                termed = terminated.detach().cpu().numpy()

        # Discriminator
        tmp = (th.ones(mac_out.shape)*(-9999999)).to(mac_out.device)
        mac_out_un = th.where(avail_actions == 0, tmp, mac_out)
        q_softmax_actions = th.nn.functional.softmax(mac_out_un[:, :-1], dim=3)

        if self.args.hard_qs:
            maxs = th.max(mac_out[:, :-1], dim=3, keepdim=True)[1]
            zeros = th.zeros_like(q_softmax_actions)
            zeros.scatter_(dim=3, index=maxs, value=1)
            q_softmax_actions = zeros

        q_softmax_agents = q_softmax_actions.reshape(q_softmax_actions.shape[0], q_softmax_actions.shape[1], -1)

        states = batch["state"][:, :-1]
        state_and_softactions = th.cat([q_softmax_agents, states], dim=2)

        if self.args.rnn_discrim:
            h_to_use = th.zeros(size=(batch.batch_size, self.args.rnn_agg_size)).to(states.device)
            hs = th.ones_like(h_to_use)
            for t in range(batch.max_seq_length - 1):
                hs = self.rnn_agg(state_and_softactions[:, t], hs)
                for b in range(batch.batch_size):
                    if t == batch.max_seq_length - 2 or (mask[b, t] == 1 and mask[b, t+1] == 0):
                        # This is the last timestep of the sequence
                        h_to_use[b] = hs[b]
            s_and_softa_reshaped = h_to_use
        else:
            s_and_softa_reshaped = state_and_softactions.reshape(-1, state_and_softactions.shape[-1])

        if self.args.mi_intrinsic:
            s_and_softa_reshaped = s_and_softa_reshaped.detach()

        discrim_prediction = self.discrim(s_and_softa_reshaped)

        # Cross-Entropy
        target_repeats = 1
        if not self.args.rnn_discrim:
            target_repeats = q_softmax_actions.shape[1]
        discrim_target = batch["noise"][:, 0].long().detach().max(dim=1)[1].unsqueeze(1).repeat(1, target_repeats).reshape(-1)
        discrim_loss = self.discrim_loss(discrim_prediction, discrim_target)

        if self.args.rnn_discrim:
            averaged_discrim_loss = discrim_loss.mean()
        else:
            masked_discrim_loss = discrim_loss * mask.reshape(-1)
            averaged_discrim_loss = masked_discrim_loss.sum() / mask.sum()
        self.logger.log_stat("discrim_loss", averaged_discrim_loss.item(), t_env)


        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        if self.args.mi_intrinsic:
            assert self.args.rnn_discrim is False
            targets = targets + self.args.mi_scaler * discrim_loss.view_as(rewards)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = loss + self.args.mi_loss * averaged_discrim_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        embed_grad_norm = th.nn.utils.clip_grad_norm_(list(self.mac.agent.embed_net.parameters()), self.args.grad_norm_clip)
        try:
            grad_norm = grad_norm.item()
            embed_grad_norm = embed_grad_norm.item()
        except:
            pass
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
        if return_q_all:
            return q_all, mix_q_all, termed, skill_all
    
    def train_meta(self, batch_log_p, mean_step_returns, mean_step_returns_new, t_env:int):
        delta_reward = th.tensor(mean_step_returns_new)-th.tensor(mean_step_returns)
        pg_loss = -th.sum(delta_reward.to(batch_log_p.device) * batch_log_p)
        self.meta_optimiser.zero_grad()
        pg_loss.backward()
        embed_grad_norm = th.nn.utils.clip_grad_norm_(list(self.mac.agent.embed_net.parameters()), self.args.grad_norm_clip)
        try:
            embed_grad_norm =embed_grad_norm.item()
        except:
            pass
        self.meta_optimiser.step()
        self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
        self.logger.log_stat("embed_grad_norm_meta", embed_grad_norm, t_env)
        self.logger.log_stat("delta_r_mean", delta_reward.mean().item(), t_env)
        self.logger.log_stat("delta_r_max", delta_reward.max().item(), t_env)
        self.logger.log_stat("delta_r_min", delta_reward.min().item(), t_env)
        self.logger.log_stat("log_p_mean", batch_log_p.mean().item(), t_env)
        self.logger.log_stat("log_p_max", batch_log_p.max().item(), t_env)
        self.logger.log_stat("log_p_min", batch_log_p.min().item(), t_env)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.discrim.cuda()
        if self.args.rnn_discrim:
            self.rnn_agg.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class Discrim(th.nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [th.nn.Linear(input_size, self.args.discrim_size), th.nn.ReLU()]
        for _ in range(self.args.discrim_layers - 1):
            layers.append(th.nn.Linear(self.args.discrim_size, self.args.discrim_size))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(self.args.discrim_size, output_size))
        self.model = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNAggregator(th.nn.Module):

    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        output_size = args.rnn_agg_size
        self.rnn = th.nn.GRUCell(input_size, output_size)

    def forward(self, x, h):
        return self.rnn(x, h)
