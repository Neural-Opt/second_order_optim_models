import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class SApollo(Optimizer):
    r"""Implements Apollo algorithm.
        This implementation is taken from the following GitHub repository:
        https://github.com/XuezheMax/apollo/blob/master/optim/adahessian.py
    
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            beta (float, optional): coefficient used for computing running averages of gradient (default: 0.9)
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-4)
            rebound (str, optional): recified bound for diagonal hessian:
                ``'constant'`` | ``'belief'`` (default: None)
            warmup (int, optional): number of warmup steps (default: 500)
            init_lr (float, optional): initial learning rate for warmup (default: lr/1000)
            weight_decay (float, optional): weight decay coefficient (default: 0)
            weight_decay_type (str, optional): type of weight decay:
                ``'L2'`` | ``'decoupled'`` | ``'stable'`` (default: 'L2')
        """

    def __init__(self, params, lr, beta=0.9, eps=1e-8, rebound='constant', warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr == "None":
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type == "None":
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(SApollo, self).__init__(params, defaults)
        print(defaults)

    def __setstate__(self, state):
        super(SApollo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
       
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['update'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['B'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['old_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Calculate current lr
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] == 'L2':
                    grad = grad.add(p, alpha=group['weight_decay'])

                beta1 = group['beta']
                beta2 = 0.98
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

        

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                # calc the diff grad
                delta_grad  = grad - (exp_avg_grad/bias_correction1)
                if group['rebound'] == 'belief':
                    sigma1 = delta_grad.norm(p=np.inf)
                else:
                    sigma1 = 0.00015
                    sigma2 = 0.0001
                 #   eps = eps / rebound

                #    eps = eps / rebound

                # Update the running average grad
                exp_avg_grad.mul_(beta1).add_(grad, alpha=1 - beta1)
                B_hat = B / bias_correction2
                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-1) - B_hat.mul(v_sq).sum()

                # Update B
               # B.addcmul_(v_sq, delta)
                B.mul_(beta2).addcmul_((v_sq), (delta), value=1 - beta2)
                # calc direction of parameter updates
                if group['rebound'] == 'belief':
                    denom = torch.max(B.abs(), rebound).add_(eps / alpha)
                else:
                  #  denom = B.abs().clamp_(min=rebound)
                    B_hat = B / bias_correction2
                    denom = B_hat.abs()
                    scaled = sigma2 + (sigma1-sigma2)*(denom/sigma1)
                    denom = torch.where(denom < sigma1,scaled ,denom)

                d_p.copy_((exp_avg_grad/bias_correction1).div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)
                state['old_grad'] =grad.clone().detach()

        return None