import torch
from torch.optim import Optimizer

class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9,beta2=0.999, eps=1e-8, weight_decay=0, weight_decouple=False, fixed_decay=False, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)

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
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1=group['beta1']
                beta2=group['beta2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if group['weight_decouple']:
                        if group['fixed_decay']:
                            p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                        else:
                            p.data.mul_(1.0 - group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
