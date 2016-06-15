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
sys.setrecursionlimit(150000)

# helper function for common structures
def log():
    print ('device: ', device)
    print ('stages: ', stages)
    print ('deathRate: ', deathRate)
    print ('niter: ', niter)
    print ('lr: ', lr)
    print ('real: ', real)

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def conv_factory_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

#written by me
def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_relu(bottom, 3, num_filter, 1, 1)
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1)
    addition = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, in_place=True)
    return relu

def residual_factory2(bottom, num_filter):
    conv1 = conv_factory_relu(bottom, 3, num_filter, 1, 1)
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1)
    addition = L.Python(bottom, conv2, module='resnet_oc', ntop=1, layer='RandAdd')
    relu = L.ReLU(addition, in_place=True)
    return relu

#written by me
def residual_factory_padding1(bottom, num_filter, stride, batch_size, feature_size):
    conv1 = conv_factory_relu(bottom, ks=3, nout=num_filter, stride=stride, pad=1)
    conv2 = conv_factory(conv1, ks=3, nout=num_filter, stride=1, pad=1)
    pool1 = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    print("hey 1")
    padding = L.Input(input_param=dict(shape=dict(dim=[batch_size, num_filter/2, feature_size, feature_size])))
    print("hey 1 after")
    concate = L.Concat(pool1, padding, axis=1)
    addition = L.Eltwise(concate, conv2, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, in_place=True)
    return relu

def residual_factory_padding2(bottom, num_filter, stride, batch_size, feature_size):
    conv1 = conv_factory_relu(bottom, ks=3, nout=num_filter, stride=stride, pad=1)
    conv2 = conv_factory(conv1, ks=3, nout=num_filter, stride=1, pad=1)
    pool1 = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    print("hey 2")
    padding = L.Input(input_param=dict(shape=dict(dim=[batch_size, num_filter/2, feature_size, feature_size])))
    print("hey 2 after")
    concate = L.Concat(pool1, padding, axis=1)
    addition = L.Python(concate, conv2, module='resnet_oc', ntop=1, layer='RandAdd')
    relu = L.ReLU(addition, in_place=True)
    return relu

def resnet(leveldb, batch_size=128, stages=[2, 2, 2, 2], first_output=16):
    feature_size=32
    data, label = L.Data(source=leveldb, backend=P.Data.LEVELDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=feature_size, mirror=True))
    residual = conv_factory_relu(data, 3, first_output, stride=1, pad=1)

    st = 0
    for i in stages[1:]:
        st += 1
        for j in range(i):
            if j==i-1:
                first_output *= 2
                feature_size /= 2
                if i==0:#never called
                    residual = residual_factory_proj(residual, first_output, 1)

                # bottleneck layer, but not at the last stage
                elif st != 3:
                    if real:
                        residual = residual_factory_padding1(residual, num_filter=first_output, stride=2,
                            batch_size=batch_size, feature_size=feature_size)
                    else:
                        residual = residual_factory_padding2(residual, num_filter=first_output, stride=2,
                            batch_size=batch_size, feature_size=feature_size)
            else:
                if real:
                    residual = residual_factory1(residual, first_output)
                else:
                    residual = residual_factory2(residual, first_output)


    glb_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True);
    fc = L.InnerProduct(glb_pool, num_output=10,bias_term=True, weight_filler=dict(type='msra'))
    loss = L.SoftmaxWithLoss(fc, label)
    return to_proto(loss)

def make_net(stages, device):

    with open('examples/python_stoch_dep/residual_train.prototxt', 'w') as f:
        train_net = resnet('/scratch/pas282/caffe/examples/cifar10/cifar10_train_leveldb_padding3', stages=stages, batch_size=128)
        print(str(train_net), file=f)

    with open('examples/python_stoch_dep/residual_test.prototxt', 'w') as f:
        test_net = resnet('/scratch/pas282/caffe/examples/cifar10/cifar10_test_leveldb_padding3', stages=stages, batch_size=100)
        print(str(test_net), file=f)

def make_solver(niter=20000, lr = 0.1):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'examples/python_stoch_dep/residual_train.prototxt'
    s.test_net.append('examples/python_stoch_dep/residual_test.prototxt')
    s.test_interval = 10000
    s.test_iter.append(100)

    s.max_iter = niter
    s.type = 'Nesterov'

    s.base_lr = lr
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'examples/python_stoch_dep/solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))


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
