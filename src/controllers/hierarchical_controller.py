from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import get_agent_order


class HierarchicalMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(HierarchicalMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.latents = None
        self.skill = None


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, need_log_p=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _, log_p, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        if need_log_p:
            return chosen_actions, log_p
        else:
            return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, t_glob=0, agent_order=None):
        agent_inputs = self._build_inputs(ep_batch, t, agent_order=agent_order)  # (bs*n,(obs+act+id))

        # avail_actions = ep_batch["avail_actions"][:, t]
        # (bs*n,(obs+act+id)), (bs,n,hidden_size)
        agent_outs, self.hidden_states, self.skill, self.log_p, loss_cs, diss_loss, ce_loss = self.agent.forward(agent_inputs, self.hidden_states, test_mode=test_mode, t_glob=t_glob)
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs, latent_dim)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":  # q for QMix. Ignored
            raise NotImplementedError

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.skill, self.log_p, loss_cs, diss_loss, ce_loss
        # (bs,n,n_actions), (bs, latent_dim)

    def init_hidden(self, batch_size):
        if self.args.use_cuda:
            self.hidden_states = th.zeros(batch_size, self.n_agents,
                                          self.args.rnn_hidden_dim).cuda()  # (bs,n,hidden_dim)
        else:
            self.hidden_states = th.zeros(batch_size, self.n_agents, self.args.rnn_hidden_dim)

    # for SeparateMAC
    def init_latent(self, batch_size):
        return self.agent.init_latent(batch_size)
    
    def _build_inputs(self, batch, t, agent_order=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        if self.args.obs_last_action:  # True for QMix
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))  # last actions are empty
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        inputs.append(batch["obs"][:, t])  # b1av
        if agent_order:
            agent_order = get_agent_order(agent_order)
            inputs[-1] = inputs[-1][:, :, agent_order].contiguous()
        if self.args.obs_agent_id:  # True for QMix
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))  # onehot agent ID

        # inputs[i]: (bs,n,n)
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)  # (bs*n, act+obs+id)
        # inputs[i]: (bs*n,n); ==> (bs*n,3n) i.e. (bs*n,(obs+act+id))
        return inputs

