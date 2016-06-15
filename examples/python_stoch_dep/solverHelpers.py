import numpy as np

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
