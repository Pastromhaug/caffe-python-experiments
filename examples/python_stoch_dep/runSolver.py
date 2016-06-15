from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from PIL import Image
import sys
from myLayers import *
from buildNetSolver import *
sys.setrecursionlimit(150000)

# helper function for common structures
def log():
    print ('device: ', device)
    print ('stages: ', stages)
    print ('deathRate: ', deathRate)
    print ('niter: ', niter)
    print ('lr: ', lr)
    print ('real: ', real)

def sample_gates():
    for i in addtables:
        if np.random.rand(1)[0] < solver.net.layers[i].deathRate:
            solver.net.layers[i].gate = False
        else:
            solver.net.layers[i].gate = True

def show_gates():
    a = []
    for i in addtables:
        a.append(solver.net.layers[i].gate)
        a.append(solver.net.layers[i].deathRate)
    print(a)

# if __name__ == '__main__':
if True:

    device = 1
    niter = 200000
    N=18
    stages = [2, N+1, N, N]
    deathRate = 0
    lr = 0.1
    real = True


    make_net(stages, device)
    make_solver(niter=niter)

    # TRAINING THE NET
    # execfile("examples/resnet_cifar/generate_final_proto.py")
    # date = time.strftime('%Y_%m_%d_%H',time.localtime(time.time()))
    #
    # caffe.set_device(device)
    # caffe.set_mode_gpu()
    # solver = None
    # solver = caffe.get_solver('examples/resnet_cifar/solver.prototxt')
    #
    # # to keep the same init with torch code
    # std = 1./np.sqrt(solver.net.params['InnerProduct1'][0].shape[1])
    # # solver.net.params['InnerProduct1'][0].data[...] = np.random.uniform(-std, std, solver.net.params['InnerProduct1'][0].shape)
    # # solver.net.params['InnerProduct1'][1].data[...] = np.random.uniform(-std, std, solver.net.params['InnerProduct1'][1].shape)
    #
    #
    # addtables = []
    # for i in range(len(solver.net.layers)):
    #     if type(solver.net.layers[i]).__name__ == 'RandAdd':
    #         addtables.append(i)
    # for i in range(len(addtables)):
    #     solver.net.layers[addtables[i]].deathRate = float(i+1)/len(addtables) * deathRate
    #     solver.net.layers[addtables[i]].train = True
    #     solver.test_nets[0].layers[addtables[i]].deathRate = float(i+1)/len(addtables) * deathRate
    #     solver.test_nets[0].layers[addtables[i]].train = False
    #
    #
    #
    # batch_size = 128
    # iter_per_epoch = int(np.ceil(50000/batch_size))
    #
    # train_loss = np.zeros(int(np.ceil(niter / iter_per_epoch)) + 1)
    # test_error = np.zeros(int(np.ceil(niter / iter_per_epoch)) + 1)
    # loss = 0
    #
    # time_last = datetime.datetime.now()
    # sample_gates()
    #
    # solver.step(1)
    # log()
    # print ('Iteration\tEpoch\tTest Accuracy\tTraining Loss\tTime')
    # for it in range(1, niter):
    #
    #     if it % iter_per_epoch == 0:
    #         time_now = datetime.datetime.now()
    #         delta_time = (time_now - time_last).seconds
    #         time_last = time_now
    #
    #         epoch = it / iter_per_epoch
    #         correct = 0
    #
    #         for test_it in range(100):
    #             solver.test_nets[0].forward()
    #             correct += sum(solver.test_nets[0].blobs['InnerProduct1'].data.argmax(1)
    #                 == solver.test_nets[0].blobs['Data2'].data)
    #         test_error[epoch] = 1 - correct / 1e4
    #         train_loss[epoch] = loss / iter_per_epoch
    #         loss = 0
    #         print('%d\t\t%d\t\t%0.2f\t\t%0.5f\t\t%ds'% (it, epoch, test_error[epoch]*100, train_loss[epoch], delta_time))
    #         np.savetxt('examples/resnet_cifar/results/%s_%d_%d_%d_%d_%.2f_%d_%.1f' % (date, niter, stages[1], stages[2], stages[3], lr, niter, deathRate),
    #             np.column_stack((test_error, train_loss)))
    #
    #     sample_gates()
    #     solver.step(1)
    #     loss += solver.net.blobs['SoftmaxWithLoss1'].data

class RandAdd(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 2
        self.train = False
        self.gate = False
        self.deathRate = 0
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        #bottom[0] is skip connection
        if self.train:
            if self.gate:
                top[0].data[...] = bottom[0].data + bottom[1].data
            else:
                top[0].data[...] = bottom[0].data
        else:
            top[0].data[...] = bottom[0].data + bottom[1].data * (1- self.deathRate)
            # print('test')

    def backward(self, top, propagate_down, bottom):
        if self.train:
            if self.gate:
                bottom[0].diff[...] = top[0].diff
                bottom[1].diff[...] = top[0].diff
            else:
                bottom[0].diff[...] = top[0].diff
                bottom[1].diff[...] = np.zeros(bottom[0].diff.shape)
        else:
            print("No backward during testing!")
