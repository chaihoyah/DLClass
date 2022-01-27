import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        self.filter_width = filter_width
        self.filter_height = filter_height
        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def conv(self, x, y,fh,fw):
        sliding_wind = view_as_windows(x,(len(x),len(x[0]),fh,fw))
        sliding_wind = np.squeeze(sliding_wind, axis=0)
        sliding_wind = np.squeeze(sliding_wind, axis=0)
        sliding_wind = np.transpose(sliding_wind,(2,0,1,3,4,5))
        out_h = len(sliding_wind[0])
        out_w = len(sliding_wind[0][0])
        sliding_wind = sliding_wind.reshape((len(x),out_h,out_w,-1))
        tmp = np.reshape(y,(len(y),-1,1))
        return np.transpose(np.squeeze(sliding_wind.dot(tmp),axis=4),(0,3,1,2))

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        out = self.conv(x,self.W,self.filter_height,self.filter_width)+self.b
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        dLdy_padded = np.pad(dLdy,((0,0),(0,0),(2,2),(2,2)))
        flipped_filt = self.W[:,:,::-1,::-1]
        flipped_filt = np.transpose(flipped_filt,(1,0,2,3))
        dLdx = self.conv(dLdy_padded,flipped_filt,self.filter_height,self.filter_width)
        x_transpose = np.transpose(x,(1,0,2,3))
        dLdY_transpose = np.transpose(dLdy,(1,0,2,3))
        print(x.shape)
        print(dLdy.shape)
        dLdW = self.conv(x_transpose,dLdY_transpose,len(x[0][0][0])-self.filter_height+1,len(x[0][0])-self.filter_width+1)
        dLdW = np.transpose(dLdW,(1,0,2,3))
        dLdb = np.sum(dLdy,axis=2)
        dLdb = np.sum(dLdb,axis=2)
        dLdb = np.sum(dLdb,axis=0)
        dLdb = np.reshape(dLdb,(1,-1,1,1))
        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        sliding_wind = view_as_windows(x,(len(x),len(x[0]),self.pool_size,self.pool_size),step=self.stride)
        sliding_wind = np.squeeze(sliding_wind,0)
        sliding_wind = np.squeeze(sliding_wind,0)
        sliding_wind = np.transpose(sliding_wind,(2,3,0,1,4,5))
        sliding_wind = np.max(sliding_wind,axis=4)
        sliding_wind = np.max(sliding_wind,axis=4)
        out = sliding_wind
        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        max = self.forward(x)
        mask = np.equal(x,max.repeat(2,axis=3).repeat(2,axis=2)).astype(int)
        dLdy_extended = dLdy.repeat(2,axis=3).repeat(2,axis=2)
        dLdx = mask*dLdy_extended
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')