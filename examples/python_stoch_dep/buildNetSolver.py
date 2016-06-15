
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
