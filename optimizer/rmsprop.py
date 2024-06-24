
import torch
from torch.optim import Optimizer

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

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
                    raise RuntimeError('RMSprop does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                avg = square_avg.sqrt().add(group['eps'])

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                p.data.addcdiv_(grad, avg, value=-group['lr'])
       
        return loss
