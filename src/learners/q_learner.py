import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.aqmix import AQMixer
from modules.mixers.qmix_central import QMixerCentralFF
import torch as th
import random
import numpy as np
from torch.optim import RMSprop
import torch.nn.functional as F

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.params_list = []
        if type(mac) == list:
            for mac_single in mac:
                self.params_list.append(list(mac_single.parameters()))
        else:
            self.params_list.append(list(mac.parameters()))
            self.params = self.params_list[0]

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer_list = []
            self.target_mixer_list = []
            for i in range(args.ensemble_num):
                if args.mixer == "vdn":
                    self.mixer_list.append(VDNMixer())
                elif args.mixer == "qmix":
                    self.mixer_list.append(QMixer(args))
                elif args.mixer == "aqmix":
                    self.mixer_list.append(AQMixer(args))
                elif args.mixer == "qmix_noabs":
                    self.mixer_list.append(QMixerCentralFF(args))
                else:
                    raise ValueError("Mixer {} not recognised.".format(args.mixer))
            if args.ensemble_num == 1:
                self.mixer = self.mixer_list[0]
                self.params += list(self.mixer.parameters())
                self.target_mixer_list.append(copy.deepcopy(self.mixer))
                self.target_mixer = self.target_mixer_list[0]
            else:
                self.mixer = self.mixer_list
                for i in range(args.ensemble_num):
                    if args.q_net_ensemble:
                        self.params_list[i] += list(self.mixer_list[i].parameters())
                    else:
                        self.params += list(self.mixer_list[i].parameters())
                    self.target_mixer_list.append(copy.deepcopy(self.mixer_list[i]))
        self.optimiser_list = []
        if args.q_net_ensemble:
            for i in range(args.ensemble_num):
                self.optimiser_list.append(RMSprop(params=self.params_list[i], lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
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

        # Calculate estimated Q-Values
        mac_out = []
        if self.args.q_net_ensemble:
            mac_chosen = self.mac[chosen_index]
        else:
            mac_chosen = self.mac
        mac_chosen.init_hidden(batch.batch_size)
        #self.mac.init_latent(batch.batch_size)
        index = th.randint(mac_chosen.n_agents, [batch.batch_size])
        index = F.one_hot(index, mac_chosen.n_agents).to(th.bool)
        rp = random.random() < self.args.contrary_grad_p
        if self.args.random_agent_order:
            enemy_shape = self.scheme["obs"]["vshape"] - mac_chosen.n_agents * 8
            enemy_num = enemy_shape // 8
            assert enemy_num * 8 ==enemy_shape
            order_enemy = np.arange(enemy_num)
            order_ally = np.arange(mac_chosen.n_agents - 1)
            np.random.shuffle(order_enemy)
            np.random.shuffle(order_ally)
            agent_order = [order_ally, order_enemy]
        else:
            agent_order = None
        for t in range(batch.max_seq_length):
            if self.args.mac == "robust_mac":
                agent_outs = mac_chosen.forward(batch, t=t, index = index, contrary_grad = rp, agent_order=agent_order)
            else:
                agent_outs = mac_chosen.forward(batch, t=t, agent_order=agent_order) #(bs,n,n_actions)
            mac_out.append(agent_outs) #[t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        #(bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_bm = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        if self.args.q_net_ensemble:
            target_mac_chosen = self.target_mac[chosen_index]
        else:
            target_mac_chosen = self.target_mac
        target_mac_chosen.init_hidden(batch.batch_size) # (bs,n,hidden_size)
        #self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac_chosen.forward(batch, t=t, agent_order=agent_order) #(bs,n,n_actions)
            target_mac_out.append(target_agent_outs) #[t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        #(bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999 # Q values

        # Max over target Q-Values
        if self.args.double_q: # True for QMix
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach() #return a new Tensor, detached from the current graph
            mac_out_detach[avail_actions == 0] = -9999999
                            # (bs,t,n,n_actions), discard t=0
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1] # indices instead of values
            # (bs,t,n,1)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            # (bs,t,n,n_actions) ==> (bs,t,n,1) ==> (bs,t,n) max target-Q
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        loss_ensemble_w = th.tensor(0.0).to(target_max_qvals.device)
        loss_ensemble_b = th.tensor(0.0).to(target_max_qvals.device)
        if self.mixer is not None:
            if return_q_all:
                q_all = chosen_action_qvals_bm.detach().cpu().numpy()
            # chosen_index = random.randint(0,len(self.mixer_list)-1)
            chosen_mixer = self.mixer_list[chosen_index]
            chosen_target_mixer = self.target_mixer_list[chosen_index]
            if self.args.mixer == "aqmix":
                chosen_action_qvals = chosen_mixer(chosen_action_qvals_bm, batch["state"][:, :-1], actions.detach())
                target_max_qvals = chosen_target_mixer(target_max_qvals, batch["state"][:, 1:], cur_max_actions.detach())
            else:
                chosen_action_qvals = chosen_mixer(chosen_action_qvals_bm, batch["state"][:, :-1])
                target_max_qvals = chosen_target_mixer(target_max_qvals, batch["state"][:, 1:])
            if len(self.mixer_list) > 1:
                other_w_list = []
                other_b_list = []
                chosen_action_qvals, w, b = chosen_action_qvals
                target_max_qvals, _, _ = target_max_qvals
                for i in range(len(self.mixer_list)):
                    if i != chosen_index:
                        if self.args.mixer == "aqmix":
                            _, other_w, other_b = self.mixer_list[i](chosen_action_qvals_bm, batch["state"][:, :-1], actions.detach())
                        else:
                            _, other_w, other_b = self.mixer_list[i](chosen_action_qvals_bm, batch["state"][:, :-1])
                        other_w_list.append(other_w.detach())
                        other_b_list.append(other_b.detach())
                for other_w, other_b in zip(other_w_list, other_b_list):
                    norm_delta_w = th.mean(th.abs((w-other_w).squeeze(2).sum(1)))
                    norm_w = th.mean(th.abs(w.detach().squeeze(2).sum(1)))
                    norm_w_other = th.mean(th.abs(other_w.squeeze(2).sum(1)))
                    loss_ensemble_w += norm_delta_w/(norm_w_other + norm_w + 1e-8)
                    norm_delta_b = th.mean(th.abs((b-other_b).view(-1)))
                    norm_b = th.mean(th.abs(b.detach().view(-1)))
                    norm_b_other = th.mean(th.abs(other_b.view(-1)))
                    loss_ensemble_b += norm_delta_b/(norm_b_other + norm_b + 1e-8)
                loss_ensemble_w /= len(self.mixer_list) - 1
                loss_ensemble_b /= len(self.mixer_list) - 1
            if return_q_all:
                mix_q_all = chosen_action_qvals.detach().cpu().numpy()
                termed = terminated.detach().cpu().numpy()
            # (bs,t,1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach()) # no gradient through target net
        # (bs,t,1)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss_q = (masked_td_error ** 2).sum() / mask.sum()
        loss = loss_q - self.args.en_w_alpha * loss_ensemble_w - self.args.en_b_alpha * loss_ensemble_b
        # Optimise
        if self.args.q_net_ensemble:
            current_optim = self.optimiser_list[chosen_index]
            current_para = self.params_list[chosen_index]
        else:
            current_optim = self.optimiser
            current_para = self.params
        current_optim.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(current_para, self.args.grad_norm_clip)# max_norm
        try:
            grad_norm =grad_norm.item()
        except:
            pass
        current_optim.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_q", loss_q.item(), t_env)
            self.logger.log_stat("loss_en_w", loss_ensemble_w.item(), t_env)
            self.logger.log_stat("loss_en_b", loss_ensemble_b.item(), t_env)
            
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
        if return_q_all:
            return q_all, mix_q_all, termed

    def _update_targets(self):
        if self.args.q_net_ensemble:
            for m, tm in zip(self.mac, self.target_mac):
                tm.load_state(m)
        else:
            self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            for i in range(len(self.mixer_list)):
                self.target_mixer_list[i].load_state_dict(self.mixer_list[i].state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        if self.args.q_net_ensemble:
            for m in self.mac:
                m.cuda()
            for tm in self.target_mac:
                tm.cuda()
        else:
            self.mac.cuda()
            self.target_mac.cuda()
        if self.mixer is not None:
            for i in range(len(self.mixer_list)):
                self.mixer_list[i].cuda()
                self.target_mixer_list[i].cuda()

    def save_models(self, path):
        if self.args.q_net_ensemble:
            self.mac[0].save_models(path)
            th.save(self.optimiser_list[0].state_dict(), "{}/opt.th".format(path))
        else:
            self.mac.save_models(path)
            th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.mixer is not None:
            if len(self.mixer_list) > 1:
                for i in range(len(self.mixer_list)):
                    th.save(self.mixer_list[i].state_dict(), "{}/mixer_{}.th".format(path, i))
            else:
                th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        

    def load_models(self, path):
        if self.args.q_net_ensemble:
            for i in range(self.args.ensemble_num):
                self.mac[i].load_models(path)
                self.target_mac[i].load_models(path)
                self.optimiser_list[i].load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        else:
            self.mac.load_models(path)
            self.target_mac.load_models(path)
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        
        if self.mixer is not None:
            if len(self.mixer_list) > 1:
                for i in range(len(self.mixer_list)):
                    self.mixer_list[i].load_state_dict(th.load("{}/mixer_{}.th".format(path, i), map_location=lambda storage, loc: storage))
            else:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

        
