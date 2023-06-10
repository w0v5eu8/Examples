import numpy as np
from scipy import linalg 
import torch
from sklearn.utils import check_array
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

class ESN():
    def __init__(self, n_readout, 
                 resSize=1500, damping=0.3, spectral_radius=None,
                 weight_scaling=1.25,initLen=0, random_state=42,inter_unit=torch.tanh, learning_rate=1e-1,zero_per = 0.9994):
        
        self.resSize = resSize
        self.n_readout =n_readout # number of readouts
        self.damping = damping  # How much to reflect previous data
        self.spectral_radius = spectral_radius # It must be less than 1. The default can be obtained through Eigen value.
        self.weight_scaling = weight_scaling
        self.initLen=initLen # where to start
        self.random_state = random_state
        self.inter_unit = inter_unit
        self.zero_per = zero_per # Zero abundance of reservoir weight
        self.learning_rate = learning_rate
        self.Win = None # Presence or absence of input weight
        self.W = None # Presence or absence of reservoir weight
        torch.manual_seed(random_state) # fix random seed value 
        self.out = None
        
        
    def fit(self,input,epoch=150):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        with torch.no_grad():
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W = values.view(self.resSize, self.resSize)
            W=W.to(device)
            
            num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+n_feature))
            values = values[perm]
            self.Win = values.view(self.resSize, 1+n_feature)
            self.Win=self.Win.to(device)
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
        rhoW = max(abs(linalg.eig(W)[0]))
        if self.spectral_radius == None:
            self.W= W*(self.weight_scaling/rhoW)
        else:
            self.W= W*(self.weight_scaling/self.spectral_radius)
        
        Yt=torch.DoubleTensor(input[:,self.initLen:]).to(device)
        
       
        x = torch.zeros((self.resSize,1)).type(torch.double)    # Hourly reservoir weight
        x=x.to(device)
        
        
        self.out = input[:,n_input-1] # Stores the last value of the input for generative mode
       
        #### train the output by ridge regression
        # reg = 1e-8  # regularization coefficient
        #### direct equations from texts:
        # X_T = X.T
        # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        # reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        '''
        reg = 1e-8
        Wout = linalg.solve(torch.matmul(self.X,self.X.T) + reg*torch.eye(1+n_feature+self.resSize), torch.matmul(X,Yt.T)).
        Wout=np.array(Wout)
        Wout=torch.DoubleTensor(Wout).to(device)
        self.Wout=torch.DoubleTensor(Wout)
        '''
        
        Wout= torch.rand(self.n_readout,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        criterion = torch.nn.MSELoss()
        parameters=[Wout]
        optimizer = optim.Adam(parameters, self.learning_rate)
        Y = torch.zeros((self.n_readout,n_input)).type(torch.double).to(device)
        
        for i in range(epoch):
            
            for t in range(n_input):
                u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device)
                if n_feature >1:
                    u=u.reshape(n_feature,1)
                x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
                y = torch.matmul( Wout, torch.vstack([torch.DoubleTensor([1]),u,x]).detach()).to(device) 
                y=y.reshape(-1)
                if t >= self.initLen:
                    Y[:,t-self.initLen] = y
                    
            loss = criterion(Y,Yt) 
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step() 
            print(i,loss.item())
        self.x =x    
        self.Wout = Wout
        
        return self    
    
    def score(self, y_pred ,y_true):
        score = r2_score(y_true, y_pred)
        return score
        
    def predict(self,input):    
        #Use after learning Wout first. After fixing Wout, input data can be predicted without learning
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        x = torch.zeros((self.resSize,1)).type(torch.double)    # x의 크기는 n_레저버 * 1
        x=x.to(device)
        n_feature, n_input = input.shape
        Y = torch.zeros((self.n_readout,n_input)).type(torch.double).to(device)
        
     
        for t in range(n_input):
            u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device)
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x]).detach()).to(device) 
            y=y.reshape(-1)
            if t >= self.initLen:
                Y[:,t-self.initLen] = y
        return Y 
    
    def future_fit(self,input,epoch =150):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        with torch.no_grad():
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W = values.view(self.resSize, self.resSize)
            W=W.to(device)
            
            num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * (1+n_feature))
            values = values[perm]
            self.Win = values.view(self.resSize, 1+n_feature)
            self.Win=self.Win.to(device)
 
            
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
        rhoW = max(abs(linalg.eig(W)[0]))
        if self.spectral_radius == None:
            self.W= W*(self.weight_scaling/rhoW)
        else:
            self.W= W*(self.weight_scaling/self.spectral_radius)
        Wout= torch.rand(self.n_readout,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        criterion = torch.nn.MSELoss()
        parameters=[Wout]
        optimizer = optim.Adam(parameters, self.learning_rate)
        Y = torch.zeros((self.n_readout,n_input-1)).type(torch.double).to(device)
        
        Yt=torch.DoubleTensor(input[:,self.initLen+1:]).to(device)
        self.out = input[:,n_input-1] # Stores the last value of the input for generative mode
        x = torch.zeros((self.resSize,1)).type(torch.double)    # x의 크기는 n_레저버 * 1
        x=x.to(device)
        for i in range(epoch):
            for t in range(n_input-1):
                u=torch.DoubleTensor(np.array(input[:,t].reshape(-1,1))).to(device) # input에서 값을 하나씩 들고온다
            
                x = (1-self.damping)*x + self.damping*self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]),u])) + torch.matmul( self.W, x ))
                y = torch.matmul(Wout, torch.vstack([torch.DoubleTensor([1]),u,x]).detach()).to(device) 
                y=y.reshape(-1)
                if t >= self.initLen:
                    Y[:,t-self.initLen] = y
            self.x= x
            loss = criterion(Y,Yt) 
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step() 
            print(i,loss.item())
        self.Wout = Wout    
        return self
                              
                                 
    def future_predict(self,outLen):    
        #It is possible to predict the time after input data in generative mode
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        x=self.x
       
        Y = torch.zeros((self.n_readout,outLen)).to(device)
        u = torch.DoubleTensor(self.out).to(device)
     
        for t in range(outLen):
            
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x]).detach()).to(device) 
            Y[:,t] = y
            # generative mode:
            u = y

        return Y

