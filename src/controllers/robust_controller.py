from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC

class RandomContraryGrad(th.autograd.Function):
    @staticmethod
    def forward(ctx, input, index):
        ctx.index = index
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # batch, n_agent, _ = grad_output.size()
        # index = th.randint(n_agent, [batch])
        # index = F.one_hot(index, n_agent).to(th.bool)
        grad_input = grad_output
        grad_input[ctx.index] = -grad_input[ctx.index]
        return grad_input, None

class RobustMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(RobustMAC, self).__init__(scheme, groups, args)
    def forward(self, ep_batch, t, test_mode=False, index = None, contrary_grad = False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        
        if contrary_grad:
            return RandomContraryGrad.apply(agent_outs.view(ep_batch.batch_size, self.n_agents, -1), index)
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
