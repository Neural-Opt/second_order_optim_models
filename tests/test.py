import torch
from torch.optim import Optimizer

import torch
import random
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # For CPU and GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Environment variables for further determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(3452)
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist


class Apollo(Optimizer):
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

    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-4, rebound='constant', warmup=0, init_lr=None, weight_decay=5e-4, weight_decay_type=None):
        
        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(Apollo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

                beta = group['beta']
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # calc the diff grad
                delta_grad = grad - exp_avg_grad
                if group['rebound'] == 'belief':
                    rebound = delta_grad.norm(p=np.inf)
                else:
                    rebound = 0.01
                    eps = eps / rebound

                # Update the running average grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                if group['rebound'] == 'belief':
                    denom = torch.max(B.abs(), rebound).add_(eps / alpha)
                else:
                    denom = B.abs().clamp_(min=rebound)

                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)

        return loss
class AdaHutch(Optimizer):
    def __init__(self, params, lr=0.0012, beta1=0.9,beta2=0.999, eps=1e-8, weight_decay=5e-4, weight_decouple=True, fixed_decay=False, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad)
        print(defaults)
        super(AdaHutch, self).__init__(params, defaults)

    def getHessian(self,grad, param):

        sum_ = torch.zeros_like(grad)
        sq  = grad**2
        n = 5
        z_vectors = torch.randint(0, 2, (n, *grad.size()), device=grad.device) * 2 - 1
       
        for i in range(n):
            z = z_vectors[i]
            sum_ += z *( torch.sum(grad * z) * grad)
        return  sum_/n


    def step(self, closure=None):
        loss = None
   

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
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['old_grad'] = torch.zeros_like(p.data)

                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg,exp_avg_sq ,exp_hessian_diag_sq = state['exp_avg'],state['exp_avg_sq'], state['exp_hessian_diag_sq']
                beta1=group['beta1']
                beta2=group['beta2']

                state['step'] += 1

                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 =  (1 - beta2 ** state['step']) ** 0.5
                if group['weight_decay'] != 0:
                    if group['weight_decouple']:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])
                isFFC = False#grad.dim()>=2# hasattr(p, 'is_fc_layer') and p.is_fc_layer and grad.dim()==2
                
                diff = ((exp_avg_sq/bias_correction1) - (exp_avg/bias_correction1)**2)
               

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if exp_avg.isnan().any():
                    print((grad).flatten()[:10])
                    raise Exception()
              #  diff = torch.where(torch.abs(diff) > grad,grad,diff)


                exp_avg_sq.mul_(beta1).addcmul_(grad,grad, value=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(diff,diff, value=1 - beta2)


             
                denom = (exp_hessian_diag_sq.sqrt() / bias_correction2).add_(group['eps'])
             

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
class AdaHessian(Optimizer):
    """
    This implementation is taken from the following GitHub repository:
    https://github.com/XuezheMax/apollo/blob/master/optim/adahessian.py
    
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"
    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        warmup (int, optional): number of warmup steps (default: 0)
        init_lr (float, optional): initial learning rate for warmup (default: 0.0)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        num_threads (int, optional) -- number of threads for distributed training (default: 1)
    """

    def __init__(self, params, lr=1,beta1=0.9,beta2=0.999 , eps=1e-4, weight_decay=5e-4,
                 warmup=0, init_lr=0.0, hessian_power=1.0, update_each=1,
                 num_threads=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1< 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.update_each = update_each
        self.num_threads = num_threads
        self.average_conv_kernel = average_conv_kernel

        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps, warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        grads = [p.grad for p in params]

        # Rademacher distribution {-1.0, 1.0}
        zs = [torch.randint_like(p, high=2) * 2.0 - 1.0 for p in params]
        # sync zs for distributed setting
        if self.num_threads > 1:
            for z in zs:
                dist.broadcast(z, src=0)

        hzs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=True)

        for hz, z, p in zip(hzs, zs, params):
            hut_trace = (hz * z).contiguous()  # approximate the expected values of z*(H@z)
            if self.num_threads > 1:
                dist.all_reduce(hut_trace)
                hut_trace.div_(self.num_threads)
            p.hess = hut_trace

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                # Calculate current lr
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - curr_lr * group['weight_decay'])

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1 = group['beta1']
                beta2 = group['beta2']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

        
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = curr_lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

import torch
from torch.optim.optimizer import Optimizer



def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)


class Shampoo(Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
    Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1802.09568

    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 3,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if update_freq < 1:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure = None) :
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
      

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state["precond_{}".format(dim_id)] = group[
                            "epsilon"
                        ] * torch.eye(dim, out=grad.new(dim, dim))
                        state[
                            "inv_precond_{dim_id}".format(dim_id=dim_id)
                        ] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state["momentum_buffer"], alpha=momentum
                    )

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state["precond_{}".format(dim_id)]
                    inv_precond = state["inv_precond_{}".format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
class Adam(Optimizer):
    def __init__(self, params, lr=0.0015, beta1=0.9,beta2=0.999, eps=1e-8, weight_decay=5e-4, weight_decouple=True, fixed_decay=False, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad)
        print(defaults)
        super(Adam, self).__init__(params, defaults)

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
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 =  (1 - beta2 ** state['step']) ** 0.5

                denom = (exp_avg_sq.sqrt()/ bias_correction2).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9,beta2=0.999, eps=1e-8, weight_decay=5e-4, weight_decouple=True, fixed_decay=False, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad)
        print(defaults)
        super(AdaBelief, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None


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
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 =  (1 - beta2 ** state['step']) ** 0.5

                denom = (exp_avg_sq.sqrt()/ bias_correction2).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                #step_size = lr / bias_correction1
               # p.data += (exp_avg/denom) * -step_size
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

import torch
from torch.optim import Optimizer
import math

class AdamJan(Optimizer):
    def __init__(self, params, lr=0.0018, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=True,weight_decay=5e-4):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
        super(AdamJan, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, epoch=None):
        loss = None

        group = self.param_groups[0]

        # Iterate over each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamJan does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_belief_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['old_grad'] = torch.zeros_like(p.data)
                    state['old_param'] = p.clone().detach()
                    


                exp_avg, exp_belief_sq,exp_avg_sq = state['exp_avg'], state['exp_belief_sq'], state['exp_avg_sq']
                beta1=group['beta1']
                beta2=group['beta2']

            
            

                state['step'] += 1
                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 =  (1 - beta2 ** state['step']) ** 0.5

             
                sign = torch.sign(grad)
                if group['weight_decay'] != 0:
                    if group['weight_decouple']:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])
   #             scale = lambda x : 1 - (1 - x) * (1 - epoch / 100) ** 2
              

                grad_diff = (grad**2 - exp_avg**2)     
          
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)

                exp_belief_sq.mul_(beta2).addcmul_(exp_avg_sq -  exp_avg**2, exp_avg_sq -  exp_avg**2, value=1 - beta2)

                denom = ((exp_belief_sq).sqrt().sqrt()/ bias_correction2).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
             #   state['old_grad'] = grad.detach().clone()


                #self.compute_diagonal_hessian()
        return loss

"""
Epoch 1/30, Loss: 1.8219810724258423
Epoch 2/30, Loss: 1.391001594309904
Epoch 3/30, Loss: 1.2302509935534731
Epoch 4/30, Loss: 1.0961586163968455
Epoch 5/30, Loss: 1.0086369988869648
Epoch 6/30, Loss: 0.9206866548985851
Epoch 7/30, Loss: 0.8783906029195202
Epoch 8/30, Loss: 0.8321032864706857
Epoch 9/30, Loss: 0.7835711082633661
Epoch 10/30, Loss: 0.7447566280559618
Epoch 11/30, Loss: 0.7159473555428642
Epoch 12/30, Loss: 0.6830500169676177
Epoch 13/30, Loss: 0.6630820206233433
Epoch 14/30, Loss: 0.6286619536730708
Epoch 15/30, Loss: 0.6109545206537053
Epoch 16/30, Loss: 0.5889184791214612
Epoch 17/30, Loss: 0.5700851180115525
Epoch 18/30, Loss: 0.5499178457016848
Epoch 19/30, Loss: 0.5407338203216085
Epoch 20/30, Loss: 0.5178610573009569
Epoch 21/30, Loss: 0.5041536865185718
Epoch 22/30, Loss: 0.5000409660290699
Epoch 23/30, Loss: 0.48114587457812563
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50,resnet34

# Gerät wählen (GPU oder CPU)
print(device)
# CIFAR-10 Datensätze mit Transformationen laden
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1028, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1028, shuffle=False, num_workers=2)

# ResNet-50 Modell initialisieren
model = resnet34(num_classes=10)  # CIFAR-10 hat 10 Klassen
model = model.to(device)

# Verlustfunktion und Optimizer definieren
criterion = nn.CrossEntropyLoss()
num_epochs = 30
#optimizer = Adam(model.parameters())
optimizer = AdamJan(model.parameters())
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        # Tag the parameters of this fully connected layer
        for param in layer.parameters():
            param.is_fc_layer = True  # Add a custom attribute to mark fully connected layers
    else:
        for param in layer.parameters():
            param.is_fc_layer = False
"""

         curvature = (((grad > 0) & (old_sign < 0) & (torch.sign(-exp_avg) >0) ) | ((grad < 0) & (old_sign > 0) & (torch.sign(-exp_avg) <0)))

         
                grad_diff = grad - exp_avg
                grad_diff = torch.where(curvature,grad_diff ,grad_diff)
"""


# Lernrate anpassen
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Trainingsschleife
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Gradienten auf Null setzen
        optimizer.zero_grad()

        # Vorwärts + Rückwärts + Optimieren
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
       # optimizer.step()
        optimizer.step(epoch)

        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}')

print('Training abgeschlossen')

# Modell evaluieren
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Genauigkeit des Modells auf den 10.000 Testbildern: {100 * correct / total}%')
