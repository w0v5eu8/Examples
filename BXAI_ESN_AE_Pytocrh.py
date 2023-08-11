import numpy as np
from scipy import linalg 
import torch
from sklearn.utils import check_array
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

class ESN():
    def __init__(self, n_latent=1, 
                 resSize=1500, damping=0.3, spectral_radius=None, n_readout=1,
                 weight_scaling=1.25,initLen=0, random_state=42,inter_unit=torch.tanh, learning_rate=0.1, zero_per =0.9994):
        
        self.resSize=resSize 
        self.n_latent= n_latent # number of latent node
        self.n_readout=n_readout  # number of readouts
        self.damping = damping  # How much to reflect previous data
        self.spectral_radius=spectral_radius # It must be less than 1. The default can be obtained through Eigen value.
        self.weight_scaling=weight_scaling
        self.initLen=initLen # where to start
        self.random_state=random_state
        self.inter_unit=inter_unit # 내부에서 사용할 함수
        self.learning_rate = learning_rate
        self.zero_per = zero_per # Zero abundance of reservoir weight
        torch.manual_seed(random_state) # torch에서 random값 고정
        self.out = None
           

    def encoder_fit(self,input,epoch=150):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        with torch.no_grad():
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W1 = values.view(self.resSize, self.resSize)
            W1=W1.to(device)
            
            num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+n_feature))
            values = values[perm]
            self.Win1 = values.view(self.resSize, 1+n_feature)
            self.Win1=self.Win1.to(device)
            
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W2 = values.view(self.resSize, self.resSize)
            W2=W2.to(device)
            
            num_zeros = int(self.resSize*(1+self.n_latent)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+self.n_latent) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+self.n_latent))
            values = values[perm]
            self.Win2 = values.view(self.resSize, 1+self.n_latent)
            self.Win2=self.Win2.to(device)
            
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
            
        rhoW1 = max(abs(linalg.eig(W1)[0]))
        rhoW2 = max(abs(linalg.eig(W2)[0]))
        
        if self.spectral_radius == None:
            self.W1= W1*(self.weight_scaling/rhoW1)
            self.W2= W2*(self.weight_scaling/rhoW2)
        else:
            self.W1= W1*(self.weight_scaling/self.spectral_radius)
            self.W2= W2*(self.weight_scaling/self.spectral_radius)
        
        Yt=torch.DoubleTensor(input[:,self.initLen:]).to(device)
        
       
        x1 = torch.zeros((self.resSize,1)).type(torch.double).to(device)
        x2 = torch.zeros((self.resSize,1)).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
      
        
       
        self.x1=x1
        self.out = input[:,n_input-1] #generative mode를 위한 input의 last value를 저장
       
        WL= torch.rand(self.n_latent,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        Wout= torch.rand(self.n_readout,1+self.n_latent +self.resSize, dtype=torch.double,requires_grad=True).to(device)
        criterion = torch.nn.MSELoss()
        parameters=[WL,Wout]
        optimizer = optim.Adam([WL,Wout], self.learning_rate)
        latent = torch.zeros((self.n_latent,n_input)).type(torch.double).to(device)
        Y = torch.zeros((self.n_readout,n_input)).type(torch.double).to(device)
        
        for i in range(epoch):
            
            for t in range(n_input):
                u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device)
                x1 = (1-self.damping)*x1 + self.damping*self.inter_unit( torch.matmul( self.Win1, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W1, x1 ) )
                la= torch.matmul( WL, torch.vstack([torch.DoubleTensor([1]),u,x1]).detach()).to(device) 
                la=la.reshape(-1)
                if t >= self.initLen:
                    latent[:,t-self.initLen] = la
            

                m = torch.DoubleTensor(latent[:,t]).reshape(-1,1).to(device)
                if self.n_readout > 1:
                    m = m.reshape(self.n_latent,1)
                x2 = (1-self.damping)*x2+ self.damping*self.inter_unit( torch.matmul( self.Win2, torch.vstack([torch.DoubleTensor([1]),m]) ) + torch.matmul( self.W2, x2 ) )
                y = torch.matmul( Wout, torch.vstack([torch.DoubleTensor([1]),m,x2]).detach()).to(device) 
                y=y.reshape(-1)
                
                if t >= self.initLen:
                    Y[:,t-self.initLen] = y
            self.x1=x1
            self.x2=x1
            loss = criterion(Y , Yt) 
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step() 
            print(i,loss.item())
        self.last = Yt[:,n_input-5]    
        self.WL = WL
        self.latent = latent
        self.Wout = Wout
        return self
    
    def encoder_predict(self,input):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape        
        Yt=torch.DoubleTensor(input[:,self.initLen:]).to(device)
        
       
        x1 = torch.zeros((self.resSize,1)).type(torch.double).to(device)
        x2 = torch.zeros((self.resSize,1)).type(torch.double).to(device) # x의 크기는 n_레저버 * 1

        latent = torch.zeros((self.n_latent,n_input)).type(torch.double).to(device)
        Y = torch.zeros((self.n_readout,n_input)).type(torch.double).to(device)
        for t in range(n_input):
            u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device)
            x1 = (1-self.damping)*x1 + self.damping*self.inter_unit( torch.matmul( self.Win1, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W1, x1 ) )
            la= torch.matmul( self.WL, torch.vstack([torch.DoubleTensor([1]),u,x1]).detach()).to(device) 
            la=la.reshape(self.n_latent)
            if t >= self.initLen:
                latent[:,t-self.initLen] = la
            

            m = torch.DoubleTensor(latent[:,t]).reshape(-1,1).to(device)
            if self.n_readout > 1:
                m = m.reshape(self.n_readout,1)
            x2 = (1-self.damping)*x2+ self.damping*self.inter_unit( torch.matmul( self.Win2, torch.vstack([torch.DoubleTensor([1]),m]) ) + torch.matmul( self.W2, x2 ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),m,x2]).detach()).to(device) 
            y=y.reshape(-1)
            if t >= self.initLen:
                Y[:,t-self.initLen] = y
        return Y
    def future_fit(self, input,epoch=150):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        with torch.no_grad():
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W1 = values.view(self.resSize, self.resSize)
            W1=W1.to(device)
            
            num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+n_feature))
            values = values[perm]
            self.Win1 = values.view(self.resSize, 1+n_feature)
            self.Win1=self.Win1.to(device)
            
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W2 = values.view(self.resSize, self.resSize)
            W2=W2.to(device)
            
            num_zeros = int(self.resSize*(1+self.n_latent)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+self.n_latent) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+self.n_latent))
            values = values[perm]
            self.Win2 = values.view(self.resSize, 1+self.n_latent)
            self.Win2=self.Win2.to(device)
            
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
            
        rhoW1 = max(abs(linalg.eig(W1)[0]))
        rhoW2 = max(abs(linalg.eig(W2)[0]))
        
        if self.spectral_radius == None:
            self.W1= W1*(self.weight_scaling/rhoW1)
            self.W2= W2*(self.weight_scaling/rhoW2)
        else:
            self.W1= W1*(self.weight_scaling/self.spectral_radius)
            self.W2= W2*(self.weight_scaling/self.spectral_radius)
        
        Yt=torch.DoubleTensor(input[:,self.initLen+1:]).to(device)
        
       
        x1 = torch.zeros((self.resSize,1)).type(torch.double).to(device)
        x2 = torch.zeros((self.resSize,1)).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
      
        
       
        self.out = input[:,n_input-1] #generative mode를 위한 input의 last value를 저장
       
        WL= torch.rand(self.n_latent,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        Wout= torch.rand(self.n_readout,1+self.n_latent +self.resSize, dtype=torch.double,requires_grad=True).to(device)
        criterion = torch.nn.MSELoss()
        parameters=[WL,Wout]
        optimizer = optim.Adam([WL,Wout], self.learning_rate)
        latent = torch.zeros((self.n_latent,n_input-1)).type(torch.double).to(device)
        Y = torch.zeros((self.n_readout,n_input-1)).type(torch.double).to(device)
        
        for i in range(epoch):
            
            for t in range(n_input-1):
                u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device)
                x1 = (1-self.damping)*x1 + self.damping*self.inter_unit( torch.matmul( self.Win1, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W1, x1 ) )
                la= torch.matmul( WL, torch.vstack([torch.DoubleTensor([1]),u,x1]).detach()).to(device) 
                la=la.reshape(-1)
                if t >= self.initLen:
                    latent[:,t-self.initLen] = la
            

                m = torch.DoubleTensor(latent[:,t]).reshape(-1,1).to(device)
                if self.n_readout > 1:
                    m = m.reshape(self.n_latent,1)
                x2 = (1-self.damping)*x2+ self.damping*self.inter_unit( torch.matmul( self.Win2, torch.vstack([torch.DoubleTensor([1]),m]) ) + torch.matmul( self.W2, x2 ) )
                y = torch.matmul( Wout, torch.vstack([torch.DoubleTensor([1]),m,x2]).detach()).to(device) 
                y=y.reshape(-1)
                
                if t >= self.initLen:
                    Y[:,t-self.initLen] = y
            
            loss = criterion(Y , Yt) 
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step() 
            print(i,loss.item()) 
        self.x1=x1
        self.x2=x1    
        self.WL = WL
        self.latent = latent
        self.Wout = Wout
        return self
    def future_predict(self, outLen):
        x1=self.x1
        x2=self.x2

        latent = torch.zeros((self.n_latent,outLen)).type(torch.double).to(device)
        Y = torch.zeros((self.n_readout,outLen)).type(torch.double).to(device)
        u = torch.DoubleTensor(self.out.reshape(-1,1)).to(device)
        for t in range(outLen):
            x1 = (1-self.damping)*x1 + self.damping*self.inter_unit( torch.matmul( self.Win1, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W1, x1 ) )
            la= torch.matmul( self.WL, torch.vstack([torch.DoubleTensor([1]),u,x1]).detach()).to(device) 
            la=la.reshape(self.n_latent)
            if t >= self.initLen:

                    latent[:,t-self.initLen] = la
            

            m = torch.DoubleTensor(latent[:,t]).reshape(-1,1).to(device)
            if self.n_readout > 1:
                m = m.reshape(self.n_readout,1)
            x2 = (1-self.damping)*x2+ self.damping*self.inter_unit( torch.matmul( self.Win2, torch.vstack([torch.DoubleTensor([1]),m]) ) + torch.matmul( self.W2, x2 ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),m,x2]).detach()).to(device) 
            y=y.reshape(-1)
            if t >= self.initLen:
                Y[:,t-self.initLen] = y
                u=y
        return Y
        
    def classify_fit(self,labels, n_classes, epoch=150):
        
        n_feature, n_input = self.latent.shape 
        if n_classes == None:
           self.n_classes = len(np.unique(labels))
        else:
            self.n_classes = n_classes
        self.n_classes = n_classes    
        
        with torch.no_grad():
            if self.W3 == None:
                num_zeros = int(self.resSize * self.resSize * self.zero_per)
                values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
                perm = torch.randperm(self.resSize * self.resSize)
                values = values[perm]
                W3 = values.view(self.resSize, self.resSize)
                W3=W3.to(device)
            if self.Win3 == None:
                num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
                values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
                perm = torch.randperm(self.resSize * (1+n_feature))
                values = values[perm]
                self.Win3 = values.view(self.resSize, 1+n_feature)
                self.Win3=self.Win1.to(device)
        
        # Modify Yt to have a shape of (n_labels, n_sequences)
        Yt = F.onehot(labels, num_classes = self.n_classes).to(device)
        Yt = Yt.type(torch.float)
        Yt= Yt.reshape(self.n_Classes, n_input)
        Y = torch.zeros((self.n_classes, n_input )).type(torch.double).to(device)
        cWout = torch.rand(self.n_classes, 1 + n_feature + self.resSize, dtype=torch.double, requires_grad=True).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        parameters = [cWout]
        optimizer = optim.Adam(parameters, self.learning_rate)
        for i in range(epoch):

            for t in range(n_input):
                u = torch.DoubleTensor(np.array(self.latent[:, t].reshape(-1,1))).to(device)
                x = (1 - self.damping) * x + self.damping * self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]), u])) + torch.matmul(self.W, x))
                y = torch.matmul(cWout, torch.vstack([torch.DoubleTensor([1]), u, x])).to(device)

                if t >= self.initLen:
                    probabilities = F.softmax(y, dim=0)
                    Y[:, t] = torch.argmax(probabilities).item()
           

            loss = criterion(Y, Yt)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(i, loss.item())
        if self.W3 == None:
            self.W3 = W3
        self.cWout = cWout
        return self
    
    def classify_predict(self):
      
        n_feature, n_input = self.latent.shape

        x = torch.zeros((self.resSize, 1)).type(torch.double).to(device)
        predictions = []

        for t in range(n_input):
            u = torch.DoubleTensor(np.array(input[:, t].reshape(-1,1))).to(device)
            x = (1 - self.damping) * x + self.damping * self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]), u])) + torch.matmul(self.W, x))
            y = torch.matmul(self.cWout, torch.vstack([torch.DoubleTensor([1]), u, x])).to(device)
            probabilities = F.softmax(y, dim=0)
            predictions.append(torch.argmax(probabilities).item())

        return np.array(predictions)