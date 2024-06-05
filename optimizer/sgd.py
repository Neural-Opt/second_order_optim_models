import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CustomSGD does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']
                momentum = group['momentum']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                if momentum != 0:
                    momentum_buffer.mul_(momentum).add_(grad)
                    grad = momentum_buffer

                p.data.add_(grad, alpha=-group['lr'])

        return loss
