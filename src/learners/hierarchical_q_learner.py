import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.hqmix import HQMixer
from modules.mixers.hqmix_noabs import HQMixerFF
from .q_learner import QLearner
import torch as th
import numpy as np
from torch.optim import RMSprop, Adam

class HierarchicalQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = scheme
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer_list = []
            self.target_mixer_list = []
            if args.mixer == "vdn":
                self.mixer_list.append(VDNMixer())
            elif args.mixer == "qmix":
                self.mixer_list.append(QMixer(args))
            elif args.mixer == "hqmix":
                self.mixer_list.append(HQMixer(args))
            elif args.mixer == "hqmix_noabs":
                self.mixer_list.append(HQMixerFF(args))
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            if args.ensemble_num == 1:
                self.mixer = self.mixer_list[0]
                self.params += list(self.mixer.parameters())
                self.target_mixer_list.append(copy.deepcopy(self.mixer))
                self.target_mixer = self.target_mixer_list[0]
                self.meta_params = list(self.mac.parameters())
            else:
                raise NotImplementedError
        if args.optimizer == "RMSprop":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.meta_optimiser = RMSprop(params=self.meta_params, lr=args.meta_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr)
            self.meta_optimiser = Adam(params=self.meta_params, lr=args.meta_lr)
        else:
            raise NotImplementedError

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, chosen_index=0, return_q_all=False):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
                # Calculate estimated Q-Values
        mac_out = []
        skill_out = []
        self.mac.init_hidden(batch.batch_size)
        self.mac.init_latent(batch.batch_size)
        reg_loss = 0
        dis_loss = 0
        ce_loss = 0
        if self.args.random_agent_order:
            enemy_shape = self.scheme["obs"]["vshape"] - self.mac.n_agents * 8
            enemy_num = enemy_shape // 8
            assert enemy_num * 8 ==enemy_shape
            order_enemy = np.arange(enemy_num)
            order_ally = np.arange(self.mac.n_agents - 1)
            np.random.shuffle(order_enemy)
            np.random.shuffle(order_ally)
            agent_order = [order_ally, order_enemy]
        else:
            agent_order = None
        for t in range(batch.max_seq_length):
            agent_outs, skills, _, loss_, dis_loss_, ce_loss_ = self.mac.forward(batch, t=t, t_glob=t_env, agent_order=agent_order) #agent_outs:(bs,n,n_actions), skills:(bs,latent_dim)
            reg_loss += loss_
            dis_loss += dis_loss_
            ce_loss += ce_loss_
            mac_out.append(agent_outs) #[t,(bs,n,n_actions)]
            skill_out.append(skills) #[t,(bs,latent_dim)]
        reg_loss /= batch.max_seq_length
        dis_loss /= batch.max_seq_length
        ce_loss /= batch.max_seq_length
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        skill_out = th.stack(skill_out, dim=1) #(bs,t,latent_dim)
        #(bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_skill_out = []
        self.target_mac.init_hidden(batch.batch_size) # (bs,n,hidden_size)
        self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs, target_skills, _, _, _, _ = self.target_mac.forward(batch, t=t, agent_order=agent_order) #(bs,n,n_actions)
            target_mac_out.append(target_agent_outs) #[t,(bs,n,n_actions)]
            target_skill_out.append(target_skills)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        target_skill_out = th.stack(target_skill_out, dim=1)
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
        if self.mixer is not None:
            if return_q_all:
                q_all = chosen_action_qvals.detach().cpu().numpy()
                skill_all = skill_out[:,:-1].detach().cpu().numpy()
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], skill_out[:,:-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_skill_out[:,1:])
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
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.use_roma:
            loss += reg_loss
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)# max_norm
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
            if self.args.use_roma:
                self.logger.log_stat("loss_reg", reg_loss.item(), t_env)
                self.logger.log_stat("loss_dis", dis_loss.item(), t_env)
                self.logger.log_stat("loss_ce", ce_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("embed_grad_norm", embed_grad_norm, t_env)
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
    
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.meta_optimiser.state_dict(), "{}/meta_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.meta_optimiser.load_state_dict(th.load("{}/meta_opt.th".format(path), map_location=lambda storage, loc: storage))
        


