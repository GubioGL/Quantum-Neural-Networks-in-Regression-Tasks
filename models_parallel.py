import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import sys
import os
# Adiciona o diretório pai ao sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gates as ops

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class Network(nn.Module):
    def __init__(self, D,number_of_layers):
        super(Network, self).__init__()

        self.D = D
        self.block1 = nn.Sequential(
                    ops.Rotation(D=D),
                    ops.Squeezing(D=D),
                    ops.Rotation(D=D),
                    ops.Displacement(D=D),
                    ops.Kerr(D=D),
                )
        if number_of_layers==2:
            self.block1 = nn.Sequential(
                    ops.Rotation(D=D),
                    ops.Squeezing(D=D),
                    ops.Rotation(D=D),
                    ops.Displacement(D=D),
                    ops.Kerr(D=D),
                )
            self.block2 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.camadas = [self.block2]
        elif number_of_layers==3:
            self.block1 = nn.Sequential(
                    ops.Rotation(D=D),
                    ops.Squeezing(D=D),
                    ops.Rotation(D=D),
                    ops.Displacement(D=D),
                    ops.Kerr(D=D),
                )
            self.block2 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block3 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.camadas = [self.block2,self.block3]
        elif number_of_layers==4:
            self.block1 = nn.Sequential(
                    ops.Rotation(D=D),
                    ops.Squeezing(D=D),
                    ops.Rotation(D=D),
                    ops.Displacement(D=D),
                    ops.Kerr(D=D),
                )
            self.block2 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block3 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block4 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.camadas = [self.block2,self.block3,self.block4]
        elif number_of_layers==5:
            self.block1 = nn.Sequential(
                    ops.Rotation(D=D),
                    ops.Squeezing(D=D),
                    ops.Rotation(D=D),
                    ops.Displacement(D=D),
                    ops.Kerr(D=D),
                )
            self.block2 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block3 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block4 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.block5 = nn.Sequential(
                ops.Rotation(D=D),
                ops.Squeezing(D=D),
                ops.Rotation(D=D),
                ops.Displacement(D=D),
                ops.Kerr(D=D),
            )
            self.camadas = [self.block2,self.block3,self.block4,self.block5]
        a = ops.Destroy(D=D).U       
        self.register_buffer('a', a)
        
        N = torch.kron(a+torch.conj(a.T), torch.eye(2))
        self.register_buffer('N', N)

        S = ops.PrepareState(D=2)
        plus =  (1/np.sqrt(2))*(S('0')+S('1'))
        self.register_buffer('plus', plus)

        S = ops.PrepareState(D=D)
        zero = S('0')
        self.register_buffer('zero', zero)

        list_ops = []
        for i in range(D):
            fock = S(f'{i}')
            fock = torch.kron(torch.matmul(fock,fock.T),torch.eye(2))
            list_ops.append(fock)
            
        self.list_ops = list_ops
        self.number_of_layers=number_of_layers
    def forward(self, x):
        x = self.block1(x)
        if self.number_of_layers !=1:
            for camada in self.camadas:
                x =camada(x)
        return x

    def encoder(self, x):
        #print(x,x.shape)
        op = (x.unsqueeze(2))*(self.a.T - self.a)
        #op = ((x.unsqueeze(2))*(self.a.T - self.a)).permute(1,2,0)
        state = torch.kron(torch.matmul(torch.matrix_exp(op), self.zero), self.plus) 
        return state

    def expect(self, x):
        return  torch.matmul(torch.conj(x.T), torch.matmul(self.N, x))
    
    def expect2(self, x):
        exp_list = []
        for i in range(self.D):
            exp = torch.matmul(torch.conj(x.T), torch.matmul(self.list_ops[i], x))
            exp_list.append((exp.item().real))
        return  exp_list
    
class Train:
    def __init__(self,number_of_layers, D, lr=0.01, device='cpu', retrain_model=False, epochs=100,step_size_=500):
        self.device     = device
        self.model      = Network(D=D,number_of_layers=number_of_layers).to(self.device)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.lr_decai   = StepLR(self.optimizer,step_size=step_size_,gamma=0.9)
        self.retrain_model = retrain_model
        self.epochs     = epochs
        self.LOSS       = []
        self.best_loss = 100000

        if self.retrain_model:
            self.load_checkpoint()

    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.LOSS = checkpoint['loss']
        
    def prepare_data(self,x_,y_):
        self.X_ = torch.tensor(x_,dtype=torch.float32).to(self.device).view(-1,1)
        self.Y_ = torch.tensor(y_,dtype=torch.float32).to(self.device)
    
    def print_parametr(self):
        layer0 = self.model.block1
        print("1° layer")
        print("Angle:",layer0[0].angle.item())
        print("Angle:",layer0[1].angle.item(),"r:",layer0[1].r.item())
        print("Angle:",layer0[2].angle.item())
        print("Angle:",layer0[3].angle.item(),"r:",layer0[3].r.item())
        print("kappa:",layer0[4].kappa.item())
        print("2° layer")
        layer1 = self.model.block2
        print("Angle:",layer1[0].angle.item())
        print("Angle:",layer1[1].angle.item(),"r:",layer1[1].r.item())
        print("Angle:",layer1[2].angle.item())
        print("Angle:",layer1[3].angle.item(),"r:",layer1[3].r.item())
        print("kappa:",layer1[4].kappa.item())
    
    def train(self,save,checkpoint_path=None):
        for _ in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()

            state   = self.model.encoder(self.X_).squeeze().T
            output  = self.model(state)
            exp     = self.model.expect(output).diag()
            loss    = torch.mean(abs((self.Y_ - exp)**2))
            
            loss.backward()
            self.optimizer.step()
            self.lr_decai.step()          
            self.LOSS.append(loss.item())
            if self.best_loss > loss.item() and save ==True:
                self.best_loss = loss.item()
                self.save_checkpoint(checkpoint_path)

    def train_state(self,Y_,save=False,checkpoint_path=None):
        
        Y_ = torch.tensor(Y_,dtype=torch.complex64).to(self.device)
        
        
        for _ in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()
            
            state   = self.model.encoder(self.X_).squeeze().T # D(x)|0> = |x>
            state   = self.model(state)
            #print(Y_.shape,state.shape)
            #fideliade = 
            loss    = torch.mean(abs((Y_ - state)**2))
            
            loss.backward()
            self.optimizer.step()
            self.lr_decai.step()          
            self.LOSS.append(loss.item())
            if self.best_loss > loss.item() and save ==True:
                self.best_loss = loss.item()
                self.save_checkpoint(checkpoint_path)

    def plots_loss(self,save_=False,path=""):
        plt.subplots(figsize=(4,4))
        plt.plot(self.LOSS)
        plt.yscale("log")
        if save_==True:
            plt.savefig(path)
        plt.show()
        
    def save_checkpoint(self,checkpoint_path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss':self.LOSS
        }, checkpoint_path)
        
    def evaluate(self,input):
        
        x_ = torch.tensor(input,dtype=torch.float32).to(self.device).view(-1,1)
        
        state   = self.model.encoder(x_).squeeze().T
        output  = self.model(state)
        exp     = self.model.expect(output).diag()
        y_predict = exp.detach().cpu().numpy().real
        return y_predict
    
    def plot_evaluate(self,x_,y_,save_=False,path=""):
        
        x_ = torch.tensor(x_,dtype=torch.float32).to(self.device).view(-1,1)
        y_ = torch.tensor(y_,dtype=torch.float32).to(self.device)
        
        state   = self.model.encoder(x_).squeeze().T
        output  = self.model(state)
        exp     = self.model.expect(output).diag()
        y_predict = exp.detach().cpu().numpy().real
            

        plt.plot(x_.cpu().numpy(), y_.cpu().numpy(),"ko", label='expected')
        plt.plot(x_.cpu().numpy(), y_predict,"r.", label='prediction')
        plt.xlabel("y (output)")
        plt.xlabel("x (input)")
        plt.legend()
        if save_==True:
            plt.savefig(path)
        plt.show()
        
    def plots_probabilidades(self,x_,logscaler=False):
        x_ = torch.tensor(x_).to(self.device)
        Y_predict = []
        for k in range(len(x_)):
            state  = self.model.encoder(x_[k])
            output = self.model(state)
            exp    = self.model.expect2(output)
            plt.plot(exp,".--",label = f"P(x={x_[k]:.2f}) {(1- sum(exp[-6:])):.4f}")
            
        plt.ylabel("Probabilit")
        plt.xlabel("Fock base")
        plt.legend()
        if logscaler==True:
            plt.yscale("log")
            plt.show()   
        plt.show()                
    
