import torch
import torch.nn.functional as F

from torch.optim import Optimizer
class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Inputs: logits from the model (raw scores before softmax)
        Targets: ground truth labels
        """
        # Apply softmax to logits to get probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create one-hot encoded labels
        one_hot_targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute the negative log-likelihood
        loss = -torch.sum(one_hot_targets * log_probs, dim=1)
        
        # Return the mean loss
        print(log_probs.mean())
        print(torch.tensor([5.5]).to('cuda').requires_grad_())
        return loss.mean()

# Example usage
criterion = CustomCrossEntropyLoss()

class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,weight_decouple=False,weight_decay=0):
        # Initialize the parameter groups and defaults
        defaults = dict(lr=lr, beta1=beta1,beta2=beta2, eps=eps,weight_decouple=weight_decouple, weight_decay=weight_decay)
        print(defaults)
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
             
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
               # self.compute_diagonal_hessian()
        return loss

    def calcHessian(self,grad,param):
        hessian_diag = []

        for param in model.parameters():
            if param.requires_grad:
                for i in range(grad.numel()):
                    grad_i = grad.flatten()[i]

                    if not grad_i.requires_grad:
                        grad_i.requires_grad_(True)
                    print("HI",grad_i.requires_grad,param.requires_grad)
           

                    grad2_i = torch.autograd.grad(grad_i, param)[0].flatten()[i]
                    hessian_diag.append(grad2_i)
        
        hessian_diag = torch.stack(hessian_diag)
        print(hessian_diag.shape)
        return hessian_diag

    def calcHessianApproxQuality(self,hessian,approx):
        #benchmark = Benchmark.getInstance(None)

        hessian_flat = hessian.view(-1)
        approx_flat = approx.view(-1)    
        cos_sim = torch.nn.functional.cosine_similarity(hessian_flat, approx_flat, dim=0)
        nmse = torch.mean((hessian - approx)**2) / (torch.var(hessian) + 1e-8)
        print(f"Cosine Similarity: {cos_sim.item()}")
        print(f"NMSE: {nmse.item()}")
        #benchmark.add("cosine_sim",cos_sim.item())
        #benchmark.add("nmse",nmse.item())

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
def calcHessian(loss,model):
    hessian_diag = []
    for param in model.parameters():
        if param.requires_grad:
            grad = torch.autograd.grad(loss, param, create_graph=True)[0]
            for i in range(grad.numel()):
                grad_i = grad.flatten()[i]
                grad2_i = torch.autograd.grad(grad_i, param, retain_graph=True)[0].flatten()[i]
                hessian_diag.append(grad2_i)
    
    hessian_diag = torch.stack(hessian_diag)
    hessian_diag = hessian_diag.detach()
    return hessian_diag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Einfaches CNN-Modell für MNIST mit etwa 10.000 Parametern
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  # 6 Filter, 3x3 Kernel
        self.conv2 = nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1)  # 8 Filter, 3x3 Kernel
        self.fc1 = nn.Linear(8 * 7 * 7, 32)  # Fully Connected Layer mit 32 Neuronen
        self.fc2 = nn.Linear(32, 10)         # 10 Output-Klassen für MNIST
        
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
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Instanziiere das Modell und verschiebe es auf das Gerät
model = SimpleCNN().to(device)

# Verlustfunktion und Optimierer
optimizer = Adam(model.parameters(), lr=0.01)
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
       
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        print(calcHessian(loss,model).shape)
        optimizer.zero_grad()  # Clear gradients before the backward pass
        loss.backward()  # Normal backward pass
        optimizer.step()  # Update model parameters
        
       
      
        
     
        
       
       
        
        #running_loss += loss.item()
       
    # Berechne die Diagonale der Hesse-Matrix am Ende jeder Epoche
       
    
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    #print(f"Diagonale der Hesse-Matrix nach Epoche {epoch+1}: {hessian_diag}")
    #print(f"Norm der Hesse-Diagonale: {torch.norm(hessian_diag).item()}")


print("Training abgeschlossen.")
print("Anzahl der Parameter im Modell:", sum(p.numel() for p in model.parameters()))