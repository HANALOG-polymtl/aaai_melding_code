''' Script to perform the budget allocation experiment'''

import numpy as np
import torch
import torch.nn as nn
import time
import pickle
from tqdm import tqdm
from functools import partial
import random
import argparse

from coverage import optimize_coverage_multilinear, CoverageInstanceMultilinear, dgrad_coverage, hessian_coverage
from submodular import ContinuousOptimizer

st =  time.time() #start time of the computation

##USEFUL PARAMETERS 
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=2) #change the number of layers while running the script

args = parser.parse_args()
num_layers = args.layers

num_items = 100
num_targets = 500
num_iters = 40
use_hessian = True
test_pct = 0.2
num_instances = 1000
total_instances = 1000

kvals = [5, 10, 20] #budget variation

activation = 'relu'
intermediate_size = 200
learning_rate = 1e-3

##RESULT INITIALISATION
opt_vals = {} #objective value of its decision evaluated using the true parameters
mse_vals = {} #MSE
algs = ['diffopt', 'twostage', 'opt', 'random'] 

for alg in algs:
    opt_vals[alg] = np.zeros((30, len(kvals)))
    mse_vals[alg] = np.zeros((30, len(kvals)))

##LOADING THE DATA 
instances_load = random.sample(range(total_instances), num_instances)
with open('benchmarks_release/budget_allocation_data.pickle', 'rb') as f:
        Pfull, wfull = pickle.load(f, encoding='bytes')
Ps= [torch.from_numpy(Pfull[i]).float() for i in instances_load]

##TRAINING
def make_fc():
    '''This function builds a fully connected neural network'''
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        net_layers.append(nn.Linear(intermediate_size, num_targets))
        net_layers.append(nn.ReLU())
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, num_targets), nn.ReLU())

def get_test_mse(net, test, loss_fn, Ps):
    '''Compute the MSE for the test set'''
    loss = 0
    for i in test:
        pred = net(data[i])
        loss += loss_fn(pred, Ps[i])
    return loss/len(test)
    
#def get_train_mse(net, train, loss_fn):
#    '''Compute the MSE for the train set'''
#    loss = 0
#    for i in train:
#        pred = net(data[i])
#        loss += loss_fn(pred, Ps[i])
#    return loss/len(train)

#def get_test_mse_random():
#    '''Compute a random MSE'''
#    loss = 0
#    train_sum = 0
#    for i in train:
#        train_sum += Ps[0].sum()
#    train_sum /= len(train)
#    for i in test:
#        pred = torch.rand(num_items, num_targets).float()
#        pred *= train_sum/pred.sum()
#        loss += loss_fn(pred, Ps[i])
#    return loss/len(test)

#def get_opt(instances, Ps, w, opt, optfunc, dgrad, hessian, verbose=True, max_x=1):
#    '''Computes the optimal x for given parameters'''
#    val = 0.
#    for i in instances:
#        x = opt.apply(Ps[i], optfunc, dgrad, verbose, hessian, max_x)
#        val += CoverageInstanceMultilinear().apply(x, Ps[i], w)
#    return val/len(instances)

def eval_opt(instances, Ps, w, opt, optfunc, dgrad, hessian, verbose=True, max_x=1, net=None):
    '''Computes and evaluates the optimal x'''
    val = 0.
    for i in instances:
        if net is not None:
            pred = net(data[i])
        else:
            pred=Ps[i]
        x = opt.apply(pred, optfunc, dgrad, verbose, hessian, max_x)
        val += CoverageInstanceMultilinear().apply(x, Ps[i], w)
    return val/len(instances)

def get_rand(instances, num_items, Ps, w, k):
    '''Random approach'''
    val = 0
    for _ in range(100):
        for i in instances:
            x = np.zeros(num_items)
            x[random.sample(range(num_items), k)] = 1
            x = torch.from_numpy(x).float()
            val += CoverageInstanceMultilinear().apply(x, Ps[i], w)
    return val/(100*len(instances))

#All results are averaged over 30 random splits
idx = 0
for idx in tqdm(range(30)):
    #print("Random split n°: ", idx)

    ###TRAIN TEST RANDOM SPLIT
    test = random.sample(range(num_instances), int(test_pct*num_instances))
    train = [i for i in range(num_instances) if i not in test]
    
    w = np.ones(num_targets, dtype=np.float32) 
    num_features = int(num_targets)
    true_transform = nn.Sequential(
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_features),
            nn.ReLU(),    
        )
    
    data = [torch.from_numpy(true_transform(P).detach().numpy()).float() for P in Ps] 
    #f_true = [CoverageInstanceMultilinear(P, w, True) for P in Ps] #TODO: think of another way of storing these, for the class to work well (deprecated)
    
    ###BUILD FC NETWORKS FOR THE 2 METHODS
    net = make_fc()
    net_two_stage = make_fc()
    
    loss_fn = nn.MSELoss()
    
      
    ###FILLING RESULTS FOR ALL METHODS
    for kidx, k in enumerate(kvals):
        optfunc = partial(optimize_coverage_multilinear, w = w, k=k, c = 0.95)
        dgrad = partial(dgrad_coverage, w = w)
        if use_hessian:
            hessian = partial(hessian_coverage, w = w)
        else:
            hessian = None

        opt = ContinuousOptimizer()
        
        opt_vals['opt'][idx, kidx] = eval_opt(test, Ps, w, opt, optfunc, dgrad, hessian, False, 0.95)
        opt_vals['random'][idx, kidx] = get_rand(test, num_items, Ps, w, k)

        optimizer = torch.optim.Adam(net_two_stage.parameters(), lr=learning_rate)
        #print('Training two stage')
        for t in range(4001):
            i = random.choice(train)
            pred = net_two_stage(data[i])
            loss = loss_fn(pred, Ps[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        opt_vals['twostage'][idx, kidx] = eval_opt(test, Ps, w, opt, optfunc, dgrad, hessian, False, 0.95, net=net_two_stage)
        mse_vals['twostage'][idx, kidx] = get_test_mse(net_two_stage, test, loss_fn, Ps) 

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for t in range(num_iters):
            i = random.choice(train)
            pred = net(data[i])
            x = opt.apply(pred, optfunc, dgrad, False, hessian, 0.95)
            loss = -CoverageInstanceMultilinear().apply(x, Ps[i], w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        opt_vals['diffopt'][idx, kidx] = eval_opt(test, Ps, w, opt, optfunc, dgrad, hessian, False, 0.95, net=net)
        mse_vals['diffopt'][idx, kidx] = get_test_mse(net, test, loss_fn, Ps)
        #for alg in algs:
        #    print("Results for method:", alg)
        #    print(opt_vals[alg][idx, kidx])

pickle.dump(opt_vals, open('output/evaluation_synthetic_full_{}_opt.pickle'.format(num_layers), 'wb'))
pickle.dump(mse_vals, open('output/evaluation_synthetic_full_{}_mse.pickle'.format(num_layers), 'wb'))
    