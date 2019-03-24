import torch
from torch.nn import Linear,Tanh,Sequential
import torch.optim as optim

import pandas as pd
from sklearn import preprocessing



df = pd.read_csv('nn_goaline.csv')
X=df[df.columns[:-1]]
Y=df[df.columns[-1]]

#Coloca os dados em escala 0 a 1
minmax_scale = preprocessing.MinMaxScaler().fit(X.astype(float))
X=minmax_scale.transform(X)





#Define a Rede
#----------------------------------------------------------------
D_in=len(X[0])
D_hidden=4
D_out=1

#model=Linear(D_in,D_out)
#model = Sequential( Linear(D_in,D_hidden),Linear(D_hidden,D_out) )
model = Sequential( Linear(D_in,D_hidden),Tanh(),Linear(D_hidden,D_out),Tanh() )
    
minibatch_size=2000

D_train=20000
D_test=10000
D_step=1000

n_step=0

#pode ser otimizado se utilizar "torch.from_numpy"
X_train=torch.Tensor(X[n_step*D_step:n_step*D_step+D_train])
Y_train=torch.Tensor(Y[n_step*D_step:n_step*D_step+D_train].values)
X_test=torch.Tensor(X[n_step*D_step+D_train:n_step*D_step+D_train+D_test])
Y_test=torch.Tensor(Y[n_step*D_step+D_train:n_step*D_step+D_train+D_test].values)



#Inicia os parametros
#with torch.no_grad():
#    for param in model.parameters(): param.uniform_(-1,1)
        


optimizer=optim.Adam(model.parameters(),0.1)


#X_train=torch.


#Loop de Treino
total_loss=0
total_lucro=0
losses={}
#----------------------------------------------------------------   
for i in range(0,1000):
    # Seleciona um minibatch aleatĂ³rio a partir do treino
    minibatch_index=torch.randperm(X_train.size(0))
    minibatch_index=minibatch_index[:minibatch_size]
    minibatch_x=X_train[minibatch_index]
    minibatch_y=Y_train[minibatch_index]

    #Forward pass
    output=model(minibatch_x)
    
    #loss=(output-minibatch_y).pow(2).sum()
    loss=-(output.clamp(min=0)*minibatch_y).mean()
    # Backward pass
    loss.backward()
    total_loss+=loss.item()
    total_lucro+=(output.clamp(min=0.0, max=1.0)*minibatch_y).sum().item()
    # Optimizer pass
    optimizer.step()
    optimizer.zero_grad()
    
    if i>0 and i%50==0:
        losses[i]=total_loss/i
        
        #loss, e o lucro em unidades
        print(losses[i], total_lucro/i )
        

