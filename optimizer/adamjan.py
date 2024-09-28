import torch 
from torch.optim import Optimizer

class AdamJan(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=False,weight_decay=0):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
        super(AdamJan, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
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
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['ema_param_delta'] = torch.zeros_like(p.data)

                    state['exp_var'] = torch.zeros_like(p.data)
                    state['old_grad'] = torch.zeros_like(p.data)
                    state['old_param'] = torch.zeros_like(p.data)
                    state['mean_change_p'] = torch.ones_like(p.data)
                    state['mean_change_g'] = torch.ones_like(p.data)



                exp_avg, exp_avg_sq,ema_param_delta, mean_change_p,mean_change_g = state['exp_avg'], state['exp_avg_sq'],state['ema_param_delta'],state['mean_change_p'],state['mean_change_g']


                beta1=group['beta1']
                beta2=group['beta2']

                state['step'] += 1
                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 =  (1 - beta2 ** state['step']) ** 0.5 
               # delta = torch.abs(p - state['old_param']  )
          
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_diff = (grad - state['old_grad'] )
                
                exp_avg_sq.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                p_delta = torch.abs(p-state['old_param'] )
               # p_delta = torch.abs(p - (state['old_param'] ))
               # ema_param_delta.mul_(beta2).add_((p_delta), alpha=1 - beta2)

                alpha  = torch.abs(p-state['old_param'] )
                beta  = torch.abs(grad- state['old_grad']  )

                mean_change_p.mul_(beta1).add_(alpha, alpha=1 - beta1)
                mean_change_g.mul_(beta1).add_(beta, alpha=1 - beta1)

                phi = torch.abs(mean_change_g - beta)  / (torch.abs(mean_change_p - alpha) + 1e-4)



                #print(torch.log1p(beta/(alpha+1e-8)).flatten())
              #  alpha = torch.abs(p_delta / (ema_param_delta/bias_correction2 + 1e-8) - 1)
             #   beta =  torch.abs(grad_diff**2 / (exp_avg_sq/bias_correction2 + 1e-8) - 1)
                gamma = torch.clamp(phi, min=0.9, max=1.2)
               # gamma = torch.clamp(torch.log1p(beta/(alpha+1e-8)), min=0.8, max=1.2)#torch.log1p(beta/(alpha+1e-8))


                #print(beta.flatten()[:10])
                #print(alpha.flatten()[:10])
               # print(((beta/(alpha))).flatten()[:10])

              #  print(torch.log1p((beta/(alpha+1e-8))).flatten()[:10])
             #   print("___________________")

                #print(torch.exp(-(torch.abs(p-state['old_param']))))



                denom = ((exp_avg_sq).sqrt()/ bias_correction2 * 1 ).add_(group['eps'])
               # print((exp_avg_sq).sqrt()/ bias_correction2)
               # print((exp_avg_sq).sqrt()/ bias_correction2)

                step_size = group['lr'] / bias_correction1
                state['old_param'] = p.clone().detach()
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                state['old_grad'] = grad.detach()
                #self.compute_diagonal_hessian()
        return loss