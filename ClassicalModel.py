import torch
import torch.nn as nn
import numpy    as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class Neuralnetwork(nn.Module):
    def __init__(self, neuronio, activation, input_=1, output_=1):
        super().__init__()        
        # input camada linear
        self.hidden_layers = nn.ModuleList([nn.Linear(input_, neuronio[0])])
        # camadas do meio
        self.hidden_layers.extend([nn.Linear(neuronio[_], neuronio[_+1]) for _ in range(len(neuronio)-1)])
        # Última camada linear
        self.output_layer = nn.Linear(neuronio[-1], output_)
        
        # Função de ativação
        self.activation_ = activation
        
            
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation_(layer(x))            
        return self.output_layer(x) 

    
class ClassicalModel:
    def __init__(self, epochs, lr, neuronio, activation,step_size_=500, device="cpu",output_=1,input_=1):

        self.epochs = epochs
        self.lr     = lr
        #self.neurons= neurons
        self.device = device
        
        self.model  = Neuralnetwork(neuronio=neuronio, output_=output_, input_=input_, activation=activation).to(device=device)
        self.opt    = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.lr_decai = StepLR(self.opt,step_size=step_size_,gamma=0.9)
        self.best_loss = 100000
        self.loss_dt= []
     
    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.loss_dt = checkpoint['loss']

    def prepare_data(self,x_,y_):
        self.inputs = torch.tensor(x_,dtype=torch.float32).to(self.device)
        self.data = torch.tensor(y_,dtype=torch.float32).to(self.device)
        
    def train_step(self,checkpoint_path, save):
        self.model.train()
        output  = self.model(self.inputs)      
        loss    = torch.mean((output- self.data)**2)
        ############################################################################
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_dt.append(loss.item())
        if self.best_loss > loss.item() and save ==True:
            self.best_loss = loss.item()
            self.save_checkpoint(checkpoint_path)

    def plot_loss(self):
        plt.plot(self.loss_dt)
        plt.yscale("log")
        plt.show()
        
    def Number_of_parameter(self):
        # Calculando o número total de parâmetros
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params
        
    def train(self,save=False,checkpoint_path=None):   
        for _ in tqdm(range(self.epochs)):
            self.train_step(save=save,checkpoint_path=checkpoint_path)
      
    def save_checkpoint(self,checkpoint_path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'loss':self.loss_dt
        }, checkpoint_path)
            
    def evaluate(self,input):
        entradas = torch.tensor(input,dtype=torch.float32).to(self.device)
        self.model.eval()
        return self.model(entradas)