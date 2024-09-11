import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim import Optimizer
from log.Logger import Logger, createNewRun
from config.loader import getConfig, getOptim, getLRScheduler 
from torch.autograd.functional import hessian 
from benchmark.benchmark import Benchmark
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(404)
class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=False,weight_decay=0):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
        super(Adam, self).__init__(params, defaults)
    @torch.no_grad()
    def step(self, hess_acc=None):
       
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
                    state['exp_avg_h'] = torch.zeros_like(p.data)
                   
                exp_avg, exp_avg_sq, exp_avg_h = state['exp_avg'], state['exp_avg_sq'],  state['exp_avg_h']
                #exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

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
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                #H_t+1 = (b2)*H_t + (1-b2)*v_t^2
                #exp_avg_h.mul_(0.999).add_(hessian_diag, alpha=(1 - 0.999))
               # self.calcHessianApproxQuality(exp_avg_h,exp_avg_sq)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])
                hess_acc.storeApproximation(p,denom.clone().detach())
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
               # self.compute_diagonal_hessian()
        return None

        #benchmark.add("cosine_sim",cos_sim.item())
        #benchmark.add("nmse",nmse.item())
class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9,beta2=0.999, eps=1e-8, weight_decay=0, weight_decouple=False, fixed_decay=False, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)

    def step(self, hess_acc=None):
 

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
               
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])
                hess_acc.storeApproximation(p,denom.clone().detach())
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return None
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

    def __init__(self, params,  lr=1e-3, beta=0.999, eps=1e-8, rebound='constant', warmup=0, init_lr=None, weight_decay=0, weight_decay_type=None):
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
        if init_lr == None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type == None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(Apollo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, hess_acc=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
       
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
                hess_acc.storeApproximation(p,denom.clone().detach())

                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)

        return None
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
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

    def __init__(self, params, lr=0.1,beta1=0.9,beta2=0.999 , eps=1e-4, weight_decay=0.0,
                 warmup=0, init_lr=0.0, hessian_power=1.0, update_each=1,
                 num_threads=1, average_conv_kernel=True):
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
    def step(self,hess_acc=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
    

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
                hess_acc.storeApproximation(p,denom.clone().detach())

                # make update
                step_size = curr_lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class HessianAccumulator:
    def __init__(self,model, beta = 0.999):
        self.beta = beta
        self.state = {}
        self.model = model
        self.layerCosineSim = []
        self.layerNMSE = []

    def calcHessianDiagonal(self,loss):
        for param in self.model.parameters():
            hessian_diag = []
            if not param in self.state:
                self.state[param] =  [torch.zeros_like(param.data).view(-1)]
            state =  self.state[param][0]
            if  param.requires_grad:
                grad = torch.autograd.grad(loss, param,create_graph=True,retain_graph=True)[0]
            
                for i in range(grad.numel()):
                    grad_i = grad.flatten()[i]
                    grad2_i = torch.autograd.grad(grad_i, param, retain_graph=True)[0].flatten()[i]
                    hessian_diag.append(grad2_i)
                
                self.state[param][0] = torch.tensor(hessian_diag).to(device)
    def calcHessianFull(self,model,images,labels):
        params_vector = parameters_to_vector(model.parameters())
        def calc_loss(params_vector):
                vector_to_parameters(params_vector, model.parameters())
                outputs = model(images)
                return criterion(outputs,labels)
        print("0")
        H = hessian(calc_loss,params_vector)
        print("1")
        diag_elements = torch.diag(H)
        print("2")
        diag_fro_norm = torch.norm(diag_elements)
        print("3")
        off_diag_fro_norm = torch.norm(H - torch.diag(torch.diag(H)))
        print("4")
        
    
    def storeApproximation(self,param,approx):
        raise Exception()
        self.state[param].append(approx)
        print(len(   self.state[param]))
        pass
    def calcHessianApproxQuality(self):
     
        for i, hessian_approx in  enumerate(self.state.values()):
            self.layerCosineSim =  self.layerCosineSim  + [[]] if  len(self.layerCosineSim) - 1 < i else  self.layerCosineSim
            self.layerNMSE =  self.layerNMSE  + [[]] if  len(self.layerNMSE) - 1 < i else  self.layerNMSE

            hessian_flat = hessian_approx[0].view(-1)
            approx_flat = hessian_approx[1].view(-1)    
            cos_sim = torch.nn.functional.cosine_similarity(hessian_flat, approx_flat, dim=0)
            nmse = torch.mean((hessian_flat - approx_flat)**2) / (torch.var(hessian_flat) + 1e-8)
            #benchmark.add("cosine_sim",cos_sim.item())
            #benchmark.add("nmse",nmse.item())
            self.layerCosineSim[i].append(cos_sim.item())
            self.layerNMSE[i].append(nmse.item())
            #print(f"Cosine Similarity: {cos_sim.item()}")
            #print(f"NMSE: {nmse.item()}")
"""
    def calcHessianApproxQuality(self):
       # benchmark = Benchmark.getInstance(None)
        print(f"state legnth {len(self.state.values())}")

        for i, (w, b) in enumerate(zip(list(self.state.values())[::2], list(self.state.values())[1::2])):
            print(w[1].shape,b[1].shape)
            print(i)

            hess = [torch.cat((w[0], b[0])),torch.cat((w[1].view(-1), b[1].view(-1)))]
            hessian_flat = hess[0].view(-1)
            approx_flat = hess[1].view(-1)    
            cos_sim = torch.nn.functional.cosine_similarity(hessian_flat, approx_flat, dim=0)
            nmse = torch.mean((hessian_flat - approx_flat)**2) / (torch.var(hessian_flat) + 1e-8)
            print(cos_sim)
            #benchmark.add(f"cosine_sim_layer_{i}",cos_sim.item())
            #benchmark.add(f"nmse_layer_{i}",nmse.item())
        raise Exception()
"""
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1)  
        self.fc1 = nn.Linear(8 * 7 * 7, 32)  
        self.fc2 = nn.Linear(32, 10)        
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 8 * 7 * 7)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Lade MNIST-Datensatz


def train(optim):
    print(optim)
    benchmark = Benchmark.getInstance(None)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    num_epochs = 3

    epoch_bar = tqdm(total=num_epochs, desc='Training Progress', unit='epoch')
    # Instanziiere das Modell und verschiebe es auf das Gerät
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Verlustfunktion und Optimierer
    optim_class = globals()[optim]
    optimizer = optim_class(model.parameters())
    model.train()
    ha = HessianAccumulator(model)

   
    for epoch in range(num_epochs):
        
        running_loss =0
        for images, labels in train_loader:
            running_loss += 1
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
           # ha.calcHessianFull(mode,images,labels)
            raise Exception()
           # loss = calc_loss(images,labels,params_vector)
            #hessian(calc_loss,params_vector)

            outputs = model(images)
            loss = criterion(outputs,labels)
           # ha.calcHessianDiagonal(loss)
            model.zero_grad()  # Clear gradients before the backward pass
         
            loss.backward(create_graph=True)  # Normal backward pass
            optimizer.step(ha)  # Update model parameters
            #ha.calcHessianApproxQuality()
            epoch_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            epoch_bar.update(1.0/len(train_loader))
            benchmark.add("loss",loss.item())
            
           
    #benchmark.add("cosine_sim",ha.layerCosineSim)
    #benchmark.add("nsme",ha.layerNMSE)

conf = getConfig()
world_size = torch.cuda.device_count()
base_path = createNewRun(f"./runs/hessian-approx")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
optims = ["Apollo"]
logger = Logger(base_path=base_path,rank=device,world_size=world_size)
for optim in optims:
    logger.setup(optim=optim)
    train(optim)
    logger.trash()
    logger.save(optim=optim)

