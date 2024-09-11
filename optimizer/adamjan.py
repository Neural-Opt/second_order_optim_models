import torch 
from torch.optim import Optimizer

class AdamJan(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=False,weight_decay=0):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
        super(AdamJan, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self,):
        loss = None
      
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
                    raise RuntimeError('AdamJan does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_var'] = torch.zeros_like(p.data)
                                        
                                        
                    state['old_grad'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq,exp_var = state['exp_avg'], state['exp_avg_sq'],state['exp_var'] 
                beta1 = group['beta1']
                beta2 = group['beta2']

                state['step'] += 1
                if group['weight_decay'] != 0 and group['weight_decouple']:
                    p.data.mul_(1 - group['weight_decay'] * group['lr'])
                elif group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update running averages of gradient and squared gradient
                #m_t+1 = (b1)*m_t + (1-b1)*g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                #v_t+1 = (b2)*v_t + (1-b2)*v_t^2
                diff = (exp_avg -  state['old_grad'])
                exp_avg_sq.mul_(beta2).addcmul_(diff, diff, value=1 - beta2)
                #d = (exp_avg_sq-exp_avg**2)
                #exp_var.mul_(beta2).addcmul_(d, d, value=1 - beta2)
            


                denom = ((exp_avg_sq)**(1/2)).add_(group['eps'])
                #hess_acc.storeApproximation(p,denom.clone().detach())

                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])
             
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                state['old_grad'] = grad 
                #self.compute_diagonal_hessian()
        return loss

