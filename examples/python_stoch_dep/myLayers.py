from caffe import layers as L, params as P

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
    padding = L.Input(input_param=dict(shape=dict(dim=[batch_size, num_filter/2, feature_size, feature_size])))
    concate = L.Concat(pool1, padding, axis=1)
    addition = L.Eltwise(concate, conv2, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, in_place=True)
    return relu

def residual_factory_padding2(bottom, num_filter, stride, batch_size, feature_size):
    conv1 = conv_factory_relu(bottom, ks=3, nout=num_filter, stride=stride, pad=1)
    conv2 = conv_factory(conv1, ks=3, nout=num_filter, stride=1, pad=1)
    pool1 = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    padding = L.Input(input_param=dict(shape=dict(dim=[batch_size, num_filter/2, feature_size, feature_size])))
    concate = L.Concat(pool1, padding, axis=1)
    addition = L.Python(concate, conv2, module='resnet_oc', ntop=1, layer='RandAdd')
    relu = L.ReLU(addition, in_place=True)
    return relu
