import numpy as np
from scipy import linalg 
import torch
from sklearn.utils import check_array
import torch.nn as nn
import torch.optim as optim


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

class ESN(nn.Module):
    def __init__(self, n_readout, 
                 resSize=1500, damping=0.3, spectral_radius=None,
                 weight_scaling=1.25,initLen=0, random_state=42,inter_unit=torch.tanh, learning_rate=1e-1,zero_per = 0.9994, l_a = 'gd',n_feature = 1):
        super(ESN, self).__init__()
        
        self.resSize = resSize # reservoir weight's size
        self.n_readout =n_readout # number of readouts
        self.damping = damping  # How much to reflect previous data
        self.spectral_radius = spectral_radius # It must be less than 1. The default can be obtained through Eigen value.
        self.weight_scaling = weight_scaling
        self.initLen=initLen # esn의 메모리 효과를 학습을 위한 길이 설정
        self.random_state = random_state # value of random seed
        self.inter_unit = inter_unit ## 내부에서 사용할 함수
        self.zero_per = zero_per # Zero abundance of reservoir weight
        self.learning_rate = learning_rate # learning rate
        torch.manual_seed(random_state) # fix random seed value 
        self.out = None # Stores the last value of the input for generative mode
        self.l_a = l_a # learning argorithm
        self._init(n_feature) # 고정된 Win, W, Wout을 initialize here
        
    def _init(self,n_feature):
        with torch.no_grad():
            self.one = torch.DoubleTensor([[1]]).to(device)
            # 아래는 W와 ,Win을 sparse 하게 구현하기 위한 알고리즘
            num_zeros = int(self.resSize * self.resSize * self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * self.resSize - num_zeros,dtype=torch.double) * 2 - 1))
            perm = torch.randperm(self.resSize * self.resSize)
            values = values[perm]
            W = values.view(self.resSize, self.resSize)
            
            
            num_zeros = int(self.resSize*(1+n_feature)*self.zero_per)
            values = torch.cat((torch.zeros(num_zeros, dtype=torch.double), torch.rand(self.resSize * (1+n_feature) - num_zeros, dtype=torch.double) * 2 - 1))
            perm = torch.randperm((1+n_feature) * self.resSize )
            values = values[perm]
            self.Win = values.view(1+n_feature, self.resSize)
            self.Win=nn.Parameter(self.Win.to(device))
        print('Computing spectral radius...')
        
        
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
        rhoW = max(abs(linalg.eig(W)[0]))
        if self.spectral_radius == None:
            self.W= W*(self.weight_scaling/rhoW)
        else:
            self.W= W*(self.weight_scaling/self.spectral_radius) #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        self.W=nn.Parameter(self.W.to(device))
        self.Wout = nn.Parameter(torch.rand(1+n_feature+self.resSize , self.n_readout, dtype=torch.double,requires_grad=True).to(device))
        print('done.')
        
    def fit(self,input,output,epoch=100): #input data를 입력하면 output을 regression 할 수 있도록 학습
   
        if input.ndim==1: # input의 shape을 확인 하고 학습에 적당한 형태로 변환
            input=input.reshape(-1,1)
        input = check_array(input, ensure_2d=True)
        
        if output.ndim==1:  # output의 shape을 확인 하고 학습에 적당한 형태로 변환
            output=output.reshape(-1,1)
        output = check_array(output, ensure_2d=True)
        n_input, n_feature= input.shape
        
            
        Yt=torch.tensor(output[self.initLen: , :],dtype=torch.double, device=device, requires_grad=False) #target data
        
        self.out = input[n_input-1,:] # Stores the last value of the input for generative mode
        
        
        
        if self.l_a == "gd": # gd를 이용한 Wout을 학습하는 esn
            
            criterion = torch.nn.MSELoss()
            parameters=[self.Wout]
            optimizer = optim.Adam(parameters, self.learning_rate)


            for i in range(epoch): 
                # x는 reservoir weight로 time step에 따라 변함
                x = torch.zeros((1,self.resSize),requires_grad = False).type(torch.double).to(device) # 매 에포크 마다 초기화, x의 크기는 n_레저버 * 1
                Y = torch.zeros((n_input-self.initLen , self.n_readout),requires_grad = False).type(torch.double).to(device) # prediction data
                for t in range(n_input):

                    u = torch.tensor(np.array(input[t,:].reshape(1,-1)), dtype=torch.double, device=device, requires_grad=False)
                    x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x,self.W ) )
                    y = torch.matmul(torch.hstack([self.one,u,x]), self.Wout)
                    if t >= self.initLen:
                        Y[t-self.initLen, :] = y.reshape(-1)

                loss = criterion(Y,Yt) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                print("epoch: {}, lost:{}".format(i,loss.item()))
                del x, Y,y,u
                torch.cuda.empty_cache()
               
            
        elif self.l_a =="inverse_matrix": # inver matrix를 이용한 기존 esn
         
            
            
            X = torch.zeros((n_input-self.initLen,1+n_feature+self.resSize),dtype = torch.double, device=device)
            x = torch.zeros((1,self.resSize),requires_grad = False).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
            for t in range(n_input):
                u = torch.tensor(np.array(input[t,:].reshape(1,-1)), dtype=torch.double, device=device) # input에서 값을 하나씩 들고온다

                x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x, self.W ) )
                    # x에 전체노드에서 소실률에 의거해 위의 식에 따라 계산된 weight값을 저장한다 
                if t >= self.initLen:
                    X[t-self.initLen, :] = torch.hstack([self.one,u,x])[0,: ]  # X에 1,u,x를 쌓아 저장한다 
                    
                self.X=X
                self.x=x # genereative mode에 필요해서 저장
                self.out = input[n_input-1, :] #generative mode를 위한 input의 last value를 저장

                #### train the output by ridge regression
                # reg = 1e-8  # regularization coefficient
                #### direct equations from texts:
                # X_T = X.T
                # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
                # reg*np.eye(1+inSize+resSize) ) )
                # using scipy.linalg.solve:
                # inverse matrix를 이용한 ridge regression을 이용하여 Wout을 구할 수 있음
                reg = 1e-8
                self.Wout = nn.Parameter(torch.linalg.solve(torch.matmul(self.X.T,self.X) + reg*torch.eye(1+n_feature+self.resSize).to(device), torch.matmul(Yt.T,X).T))

                
                
        return self    
    
        
    def predict(self,input):    
        #Use after learning Wout first. After fixing Wout, input data can be predicted without learning
        if input.ndim==1:
            input=input.reshape(-1,1)
        input = check_array(input, ensure_2d=True)
        x = torch.zeros((1,self.resSize),requires_grad = False).type(torch.double).to(device) # x의 크기는 n_레저버 * 1  
        n_input,n_feature, = input.shape
        Y = torch.zeros((n_input,self.n_readout)).type(torch.double).to(device)

        for t in range(n_input):
            u=torch.tensor(np.array(input[t,:].reshape(1,-1)),dtype=torch.double, device=device, requires_grad=False)
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x, self.W ) )
            y = torch.matmul(torch.hstack([self.one,u,x]), self.Wout) 
         
            Y[t-self.initLen , :] = y.reshape(-1)
        return Y 
    
    def future_fit(self,input,output,epoch =150):
        if input.ndim==1: # input의 shape을 확인 하고 학습에 적당한 형태로 변환
            input=input.reshape(-1,1)
        if output.ndim==1:  # output의 shape을 확인 하고 학습에 적당한 형태로 변환
            output=output.reshape(-1,1)
            
        output = check_array(output, ensure_2d=True)
        input = check_array(input, ensure_2d=True)
        n_input, n_feature= input.shape      
        Yt=torch.tensor(output[self.initLen+1: , :],dtype=torch.double, device=device, requires_grad=False) #target data
        self.out = input[n_input-1,:] # Stores the last value of the input for generative mode

        if self.l_a == "gd": # gd를 이용한 Wout을 학습하는 esn
            
            criterion = torch.nn.MSELoss()
            parameters=[self.Wout]
            optimizer = optim.Adam(parameters, self.learning_rate)
            
            for i in range(epoch):
                Y = torch.zeros((self.n_readout,n_input-self.initLen-1),requires_grad= False).type(torch.double).to(device)
                x = torch.zeros((self.resSize,1),requires_grad = False).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
                for t in range(n_input-1):
                    u = torch.tensor(np.array(input[t,:].reshape(1,-1)), dtype=torch.double, device=device, requires_grad=False)
                    x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x, self.W ) )
                    y = torch.matmul(torch.hstack([self.one,u,x]), self.Wout)

                    if t >= self.initLen:
                            Y[t-self.initLen, :] = y.reshape(-1)

                self.x= x
                loss = criterion(Y,Yt) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                print("epoch: {}, lost:{}".format(i,loss.item()))
                if i == epoch-1:
                    del u,y
                else:
                    del u,y,x
        elif self.l_a =="inverse_matrix": # inver matrix를 이용한 기존 esn
            
            X = torch.zeros((n_input-self.initLen-1,1+n_feature+self.resSize),dtype = torch.double, device=device)
            x = torch.zeros((1,self.resSize),requires_grad = False).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
            for t in range(n_input-1):
                u = torch.tensor(np.array(input[t,:].reshape(1,-1)), dtype=torch.double, device=device) # input에서 값을 하나씩 들고온다

                x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x, self.W ) )
                    # x에 전체노드에서 소실률에 의거해 위의 식에 따라 계산된 weight값을 저장한다 
                if t >= self.initLen:
                    X[t-self.initLen, :] = torch.hstack([self.one,u,x])[0,: ]  # X에 1,u,x를 쌓아 저장한다
                    
            self.X=X
            self.x=x # genereative mode에 필요해서 저장
            self.out = input[n_input-1, :] #generative mode를 위한 input의 last value를 저장

            #### train the output by ridge regression
            # reg = 1e-8  # regularization coefficient
            #### direct equations from texts:
            # X_T = X.T
            # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
            # reg*np.eye(1+inSize+resSize) ) )
            # using scipy.linalg.solve:
            reg = 1e-8
            self.Wout = nn.Parameter(torch.linalg.solve(torch.matmul(self.X.T,self.X) + reg*torch.eye(1+n_feature+self.resSize).to(device), torch.matmul(Yt.T,self.X).T)) 
            #self.Wout = Wout    
        return self
                              
                                 
    def future_predict(self,outLen):    
        #It is possible to predict the time after input data in generative mode
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        x=self.x # 학습에서 마지막 reservoir weight를 들고와서 사용
        y = torch.zeros((1,self.n_readout)).to(device)
        Y = torch.zeros((outLen, self.n_readout)).to(device)
        u = torch.DoubleTensor(self.out).to(device).unsqueeze(0)
        
        
        for t in range(outLen):
            #print("시작")
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul(torch.hstack([self.one,u]), self.Win)  + torch.matmul( x, self.W ) )
                    # x에 전체노드에서 소실률에 의거해 위의 식에 따라 계산된 weight값을 저장한다 
            
            y = torch.matmul(torch.hstack([self.one,u,x]), self.Wout)
            y=y.reshape(-1)
            Y[t,:] = y
            # generative mode:
            u = y.reshape(1,-1)

        return Y
