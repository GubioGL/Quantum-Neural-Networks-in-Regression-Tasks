import torch
import sys
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from math import log as log
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Function


pi = np.pi

class PrepareState(nn.Module):
    def __init__(self, D, device='cpu'):
        super(PrepareState, self).__init__()

        self.device = device
        self.fock = torch.zeros((D, D, 1), dtype=torch.complex64, device=self.device)
        for i in range(D):
            self.fock[i][i] = 1.0

    def forward(self, string):
        state = torch.eye(1, dtype=torch.complex64, device=self.device)
        st = ''
        for i in range(len(string)):
            s = string[i]
            if s.isdigit() is False: 
                state = torch.kron(state, self.fock[int(st)])
                st = ''
            elif i == len(string)-1:
                st += s
                state = torch.kron(state, self.fock[int(st)])
            else:
                st += s

        return state


class Destroy(nn.Module):

    def __init__(self, D):
        super(Destroy, self).__init__()

        self.U = torch.zeros((D, D), dtype=torch.complex64)
        for i in range(D-1):
            self.U[i][i+1] = (i+1)**0.5

    def forward(self, x):
        U = torch.kron(self.U, torch.eye(2, dtype=torch.complex64, device=x.device))

        return torch.matmul(U, x)


class Create(nn.Module):

    def __init__(self, D):
        super(Create, self).__init__()

        U = torch.zeros((D, D), dtype=torch.complex64)
        for i in range(D-1):
            U[i+1][i] = (i+1)**0.5

        self.register_buffer('U', U)

    def forward(self, x):
        U = torch.kron(self.U, torch.eye(2, dtype=torch.complex64, device=x.device))

        return torch.matmul(U, x)


class Displacement(nn.Module):
    def __init__(self, r=None, angle=False, D=1):
        super(Displacement, self).__init__()

        self.D  = D

        if angle is None:
            self.angle = nn.Parameter(torch.rand(1))
        elif angle is False:
            angle = torch.zeros(1)
            self.register_buffer('angle', angle)

        if r is None:
            self.r = nn.Parameter(torch.rand(1)*.5)
        else:
            self.r = torch.tensor([r])

        a = Destroy(self.D).U
        b = Create(self.D).U

        self.register_buffer('a', a)
        self.register_buffer('b', b)

    def forward(self, x):

        z = self.r*torch.exp(1j*self.angle*pi)
        op = (z * self.b - torch.conj(z) * self.a)
        U = torch.kron(torch.matrix_exp(op), torch.eye(2, device=x.device))

        return torch.matmul(U, x)


class Squeezing(nn.Module):

    def __init__(self, r=None, angle=False, D=2):
        super(Squeezing, self).__init__()

        self.D = D

        if r is None:
            self.r = nn.Parameter(torch.rand(1)*.5)
        else:
            self.r = torch.tensor(r)

        if angle is None:
            self.angle = nn.Parameter(torch.randn(1)*pi)
        elif angle is False:
            self.angle = torch.zeros(1)
        else:
            self.angle = torch.tensor(angle)

        a = Destroy(self.D).U
        self.register_buffer('a', a)


    def forward(self, x):
        self.angle = self.angle.to(x.device)
        z = self.r*torch.exp(1j*self.angle*pi)
        op = (1 / 2.0) * z.conj() * (torch.matmul(self.a, self.a)) - (1 / 2.0) * z * (torch.matmul(self.a.T, self.a.T))
        U = torch.kron(torch.matrix_exp(op), torch.eye(2, device=x.device))

        return torch.matmul(U, x)


class Rotation(nn.Module):
    def __init__(self, angle=None, D=1):
        super(Rotation, self).__init__()

        self.D = D

        if angle is None:
            self.angle = nn.Parameter(torch.rand(1)*pi)
        else:
            self.angle = angle

        a = Destroy(self.D).U
        self.register_buffer('a', a)

    def forward(self, x):
        op = 1j*self.angle*torch.matmul(self.a.T, self.a)*pi
        M = torch.matrix_exp(op)
        U = torch.kron(M, torch.eye(2, device=x.device))

        return torch.matmul(U, x)


class Kerr(nn.Module):
    def __init__(self, kappa=None, D=2):
        super(Kerr, self).__init__()

        self.D = D

        sigmap = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        sigmam = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        sigmax = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        nu_1x = 6*np.pi
        eta_x = 0.2
        F_omega = 15 #setup of Rabi frequencies
        Omega_1x = nu_1x/F_omega

        a = torch.kron(Destroy(D=D).U, torch.eye(2, dtype=torch.complex128))
        
        Na = torch.matmul(torch.conj(a.T), a)
        x_a = a + torch.conj(a.T)

        sig_p = torch.kron(torch.eye(D, dtype=torch.complex128), sigmap)
        sig_m = torch.kron(torch.eye(D, dtype=torch.complex128), sigmam)
        H0 = nu_1x*Na
        H02 = -((eta_x**4)/2 - eta_x**2)*(Omega_1x/2)*Na
        
        Op_x    =   (1j*eta_x*x_a)
        Opd_x   =   (-1j*eta_x*x_a)
        H       =   (1/2)*Omega_1x*torch.matmul(torch.matrix_exp(Op_x), sig_p)
        Hd      =   (1/2)*Omega_1x*torch.matmul(torch.matrix_exp(Opd_x), sig_m)
        Xi      =   (1/8)*Omega_1x*(eta_x**4)

        M = int(np.ceil(1/Xi))
        if kappa is None:
            self.kappa = nn.Parameter(2*np.pi*torch.rand(1))
        else:
            self.kappa = nn.Parameter(kappa*torch.ones(1))

        self.v = 1.64*np.linspace(0,M,10*M)[-1]

        self.register_buffer('H0', H0)
        self.register_buffer('H', H)
        self.register_buffer('Hd', Hd)
        self.register_buffer('H02', H02)
        

    def forward(self, x):
        x = x.type(torch.complex128)

        op1 = -1j*self.kappa*(self.H0-self.H-self.Hd)*self.v
        op2 = 1j*self.H0*self.kappa*self.v
        op3 = 1j*self.H02*self.kappa*self.v

        U1 = torch.matrix_exp(op1)
        U2 = torch.matrix_exp(op2)
        U3 = torch.matrix_exp(op3)
        
        x = torch.matmul(U1, x)
        x = torch.matmul(U2, x)
        x = torch.matmul(U3, x)

        x = x.type(torch.complex64)
        return x


class Kerr0(nn.Module):
    def __init__(self, kappa=None, D=1):
        super(Kerr0, self).__init__()
        
        self.D = D
        if kappa is None:
            self.kappa = nn.Parameter(torch.randn(1))
        else:
            self.kappa = nn.Parameter(kappa*torch.ones(1, dtype=torch.complex64))

        a = Destroy(self.D).U
        b = Create(self.D).U
        n = torch.matmul(b, a)
        H = torch.matmul(n, n)
        self.register_buffer('H', H)
        self.register_buffer('a', a)

    def forward(self, x):
        op = 1j*self.kappa*self.H
        M = torch.matrix_exp(op)
        U = torch.kron(M, torch.eye(2, device=x.device, dtype=torch.complex64))

        x = torch.matmul(U, x)

        return x