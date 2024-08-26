import torch
from torch.optim import Optimizer
from benchmark.benchmark import Benchmark
 
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=False,weight_decay=0):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
        self.benchmark = Benchmark.getInstance(None)

        super(Adam, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        group = self.param_groups[0]
        all_params = (p for group in self.param_groups for p in group['params'] if p.requires_grad)
        all_grads  = [p.grad for p in all_params]

        # Iterate over each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_hq'] = torch.zeros_like(p.data)
                   
                exp_avg, exp_avg_sq, exp_avg_hq = state['exp_avg'], state['exp_avg_sq'],  state['exp_avg_hq']
                #exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1 = group['beta1']
                beta2 = group['beta2']

                state['step'] += 1
                if group['weight_decay'] != 0 and group['weight_decouple']:
                    p.data.mul_(1 - group['weight_decay'] * group['lr'])
                elif group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                hessian_diag = self.calcHessian(p.grad,p)
                # Update running averages of gradient and squared gradient
                #m_t+1 = (b1)*m_t + (1-b1)*g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                #v_t+1 = (b2)*v_t + (1-b2)*v_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                #H_t+1 = (b2)*H_t + (1-b2)*v_t^2

                exp_avg_hq.mul_(beta2).add_(hessian_diag, alpha=1 - beta2)
        
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])
             
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                self.calcHessianApproxQuality(exp_avg_hq,exp_avg_sq)
               # self.compute_diagonal_hessian()
        return loss

    def calcHessian(self,grad,param):
        hessian_diag = []
        # Compute the second derivative (d2L/dw_i^2) using autograd
        second_derivative = torch.autograd.grad(grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
        hessian_diag.append(second_derivative.view(-1))
        return torch.cat(hessian_diag).reshape(param.shape)
    def calcHessianApproxQuality(self,hessian,approx):
        hessian_flat = hessian.view(-1)
        approx_flat = approx.view(-1)    
        cos_sim = torch.nn.functional.cosine_similarity(hessian_flat, approx_flat, dim=0)
        nmse = torch.mean((hessian - approx)**2) / (torch.var(hessian) + 1e-8)

        self.benchmark.add("cosine_sim",cos_sim.item())
        self.benchmark.add("nmse",nmse.item())