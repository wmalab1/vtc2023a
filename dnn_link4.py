import tensorflow as tf
import numpy as np
import time
import datetime
from sklearn.utils import shuffle
import os

print("===== np version: %s =====" %np.__version__)
print("===== tf version: %s =====" %tf.__version__)
print("===== Is GPU available?: %s =====" %tf.test.is_gpu_available())

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def file_write(file_name, write_type, data):
    with open(file_name, write_type) as f:
        if data.shape[0] == data.size:
            for i in range(data.shape[0]):
                f.write('%10.10g\n' % (data[i]))
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write('%10.10g    ' % (data[i, j]))
                f.write('\n')
    f.close()

def file_read(file_name, data_type):
    f = open(file_name, 'r')
    data = f.readlines()
    w = np.zeros_like(data_type)
    for i, line in enumerate(data):
        line = line.rstrip("\n")
        line = line.rstrip()
        if data_type.shape[0] == data_type.size:
            w[i] = float(line)
        else:
            try:
                a = line.split()
                a = [float(j) for j in a]
                w[i,:] = a

            except ValueError as e:
                print(e)
                print ("on line %d" %i)
                print(data_type.shape)
                print("a.shape: %s" %len(a))
                print("w.shape: %s" %w[i,:].shape)
    f.close()
    return w

class Dnn:
    def __init__(self, mode_shuffle='disable',batch_size=100, n_epoch=2000, layer_dim_list=[16, 16, 3], max_pkt_size=2^16,
                 num_pkt=3, D_step=256, r=[1,2,3,4], c=[1,4/3,2], num_link=2, N=0, nr=2, nt=2):

        self.mode_shuffle = mode_shuffle
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list

        self.max_pkt_size = max_pkt_size
        self.num_pkt = num_pkt

        self.D_step = D_step

        self.R = r
        self.C =c

        self.num_link = num_link
        self.N = N

        self.Nr = nr
        self.Nt = nt

        self.num_R = r.shape[1]
        self.num_C = c.shape[1]

        self.R_max = 4
        self.Cx3_max = 6

    def Linear_interpolation(self, SINR, Pout_stbc_tf):
        small_v1 = tf.cast(tf.math.floor(SINR), tf.float64)
        big_v2 = tf.cast(tf.math.ceil(SINR), tf.float64)
        S_idx1 = tf.cast(small_v1 - (-200), tf.int32)
        S_idx2 = tf.cast(big_v2 - (-200), tf.int32)

        Pout = tf.where(tf.equal(small_v1, big_v2),
                        tf.gather_nd(Pout_stbc_tf, [S_idx1]),
                        ((big_v2 - SINR) / (big_v2 - small_v1)) * tf.gather_nd(Pout_stbc_tf, [S_idx1]) +
                        ((SINR - small_v1) / (big_v2 - small_v1)) * tf.gather_nd(Pout_stbc_tf, [S_idx2])
                        )
        return Pout

    def Extrapolation(self, SINR, Pout_stbc_tf):
        last_v1 = 60
        last_v2 = 59

        last_p1 = Pout_stbc_tf[-1]
        last_p2 = Pout_stbc_tf[-2]

        log_Pout = ((log10(last_p1) - log10(last_p2)) / (last_v1 - last_v2)) * (SINR - last_v2) + log10(last_p2)
        Pout = pow(10, log_Pout)

        return Pout

    def Calc_distort(self, input_dr_link1, input_dr_link2, input_dr_link3, input_dr_link4,input_h,
                     R_link1, R_link2, R_link3, R_link4, Cx3_link1, Cx3_link2, Cx3_link3, Cx3_link4, power, Pout_all_stbc ):

        D = np.array([input_dr_link1, input_dr_link2, input_dr_link3, input_dr_link4])
        h = np.reshape(input_h,[self.num_link, self.num_link])
        R_link = np.array([R_link1, R_link2, R_link3, R_link4])
        Cx3_link = np.array([Cx3_link1, Cx3_link2, Cx3_link3, Cx3_link4])

        E_D = 0
        SINR = np.zeros([self.num_link])
        for i in range(self.num_link):
            den = self.N
            for j in range(self.num_link):
                if j != i:
                    den += power[j] * h[i, j]
            SINR[i] = power[i] * h[i, i] / den

        SINR_dB = 10 * np.log10(SINR)

        Pout = np.zeros([self.num_link, self.num_pkt])
        C_idx = -1
        R_idx = -1

        for i in range(self.num_link):
            for a in range(self.num_pkt):
                if Cx3_link[i,a] == 3:
                    C_idx = 0
                elif Cx3_link[i, a] == 4:
                    C_idx = 1
                elif Cx3_link[i, a] == 6:
                    C_idx = 2

                R_idx = (R_link[i,a]).astype('int32')

                if SINR_dB[i]<-200:
                    Pout[i,a] = 1
                elif -200 <= SINR_dB[i] <= 60:
                    small_v1 = (np.floor(SINR_dB[i])).astype('float64')
                    big_v2 = (np.ceil(SINR_dB[i])).astype('float64')

                    S_idx1 = (small_v1 - (-200)).astype('int32')
                    S_idx2 = (big_v2 - (-200)).astype('int32')

                    if S_idx1 == S_idx2:
                        Pout[i, a] = Pout_all_stbc[C_idx,S_idx1,R_idx]
                    else:
                        w1 = (big_v2 - SINR_dB[i]) / (big_v2 - small_v1)
                        w2 = (SINR_dB[i] - small_v1) / (big_v2 - small_v1)
                        Pout[i,a] = w1 * Pout_all_stbc[C_idx, S_idx1, R_idx ] + w2 * Pout_all_stbc[C_idx, S_idx2, R_idx]
                else:
                    log_Pout = ((np.log10(Pout_all_stbc[C_idx,-1,R_idx] ) - np.log10(Pout_all_stbc[C_idx,-2,R_idx])) / (60 - 59)) * (SINR_dB[i] - 59) + np.log10(Pout_all_stbc[C_idx,-2,R_idx])
                    Pout[i,a] = pow(10, log_Pout)

        link_idx = np.arange(0, self.num_link, 1).reshape([-1, 1])

        for success_pkt in range(self.num_pkt, 0, -1):
            total_bits = self.max_pkt_size * (np.sum(R_link[:, 0:success_pkt]* Cx3_link[:,0:success_pkt], axis=1, keepdims=True) / (self.R_max* self.Cx3_max))

            D_idx1 = (total_bits / self.D_step).astype('int32')
            D_idx2 = D_idx1 + 1

            small_v = (D_idx1 * self.D_step).astype('float64')
            big_v = (D_idx2 * self.D_step).astype('float64')

            w1 = (big_v - total_bits) / (big_v - small_v)
            w2 = (total_bits - small_v) / (big_v - small_v)

            if len(np.where(R_link1 == self.R_max)[0]) == self.num_pkt:
                D_idx_link1 = (total_bits[0] / self.D_step).astype('int32')
                D_idx2[0] = D_idx_link1

            if len(np.where(R_link2 == self.R_max)[0]) == self.num_pkt:
                D_idx_link2 = (total_bits[1] / self.D_step).astype('int32')
                D_idx2[1] = D_idx_link2

            if len(np.where(R_link3 == self.R_max)[0]) == self.num_pkt:
                D_idx_link3 = (total_bits[2] / self.D_step).astype('int32')
                D_idx2[2] = D_idx_link3

            if len(np.where(R_link4 == self.R_max)[0]) == self.num_pkt:
                D_idx_link4 = (total_bits[3] / self.D_step).astype('int32')
                D_idx2[3] = D_idx_link4

            Distortion = w1 * D[link_idx, D_idx1] + w2 * D[link_idx, D_idx2]

            if success_pkt == self.num_pkt:
                value = Distortion
            else:
                value = Distortion * np.reshape(Pout[:, success_pkt], [self.num_link, 1])

            E_D = E_D + value
            E_D = E_D * (1 - np.reshape(Pout[:, success_pkt - 1], [self.num_link, 1]))

        E_D = E_D + np.reshape(D[:, 0], [self.num_link, 1]) * np.reshape(Pout[:, 0], [self.num_link, 1])  # f(0)*P1에 해당

        return np.reshape(E_D, [1, self.num_link]), np.reshape(SINR,[1,self.num_link])

    def train_dnn(self, input_dr, input_h, num_pl_per_img, lr, Pout_all_stbc, file_path):

        self.weights = []
        self.biases = []

        dr_pair = int(input_dr.shape[0]/self.num_link)

        num_batch = int(dr_pair / self.batch_size)

        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)

        input_dr_not_scaled = input_dr
        input_dr_log = np.log10(input_dr)

        max_dr_log = np.max(input_dr_log)
        min_dr_log = np.min(input_dr_log)

        temp = (input_dr_log - min_dr_log) / (max_dr_log - min_dr_log)
        input_dr = 2 * temp - 1

        input_dr_link1 = input_dr[:dr_pair, :]
        input_dr_link2 = input_dr[dr_pair:2*dr_pair, :]
        input_dr_link3 = input_dr[2*dr_pair:3*dr_pair, :]
        input_dr_link4 = input_dr[3*dr_pair:, :]

        input_dr_not_scaled_link1 = input_dr_not_scaled[:dr_pair]
        input_dr_not_scaled_link2 = input_dr_not_scaled[dr_pair:2*dr_pair, :]
        input_dr_not_scaled_link3 = input_dr_not_scaled[2*dr_pair:3*dr_pair, :]
        input_dr_not_scaled_link4 = input_dr_not_scaled[3*dr_pair:, :]

        input_h_not_scaled = input_h
        input_h_log = np.log10(input_h)

        avg_h_log = np.mean(input_h_log)
        std_h_log = np.std(input_h_log)

        input_h = (input_h_log - avg_h_log) / std_h_log

        with open(file_path + '\input_scaling_param.dat','w') as f:
            f.write('   %.20g\n   %.20g\n   %.20g\n   %.20g\n' % (max_dr_log, min_dr_log, avg_h_log, std_h_log))

        with tf.device('/CPU:0'):
            tf.reset_default_graph()
            x_ph = tf.placeholder(tf.float64, shape=[None, input_dr.shape[1]*self.num_link+self.num_link**2])

            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input_dr.shape[1]*self.num_link+self.num_link**2
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i-1]
                    out_dim = self.layer_dim_list[i]

                weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=tf.sqrt(2.0 / tf.cast(in_dim, tf.float64)), seed=seed_weight * (i * i + 1), dtype=tf.float64), dtype=tf.float64)
                bias = tf.Variable(tf.zeros(out_dim, dtype=tf.float64), dtype=tf.float64)

                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list)-1:
                    out_layer = tf.nn.relu(mult)

                else:
                    mult_link1 = mult[:, : self.num_pkt*(self.num_R + self.num_C)]
                    mult_link2 = mult[:, self.num_pkt*(self.num_R + self.num_C) : 2*self.num_pkt*(self.num_R + self.num_C)]
                    mult_link3 = mult[:, 2*self.num_pkt*(self.num_R + self.num_C): 3*self.num_pkt*(self.num_R + self.num_C)]
                    mult_link4 = mult[:, 3*self.num_pkt*(self.num_R + self.num_C): 4*self.num_pkt*(self.num_R + self.num_C)]
                    mult_pw = mult[:,- self.num_link :]

                    output_pw = tf.nn.sigmoid(mult_pw) + 1e-100

                    output_prob_R_link1 = tf.concat(
                        [
                            tf.nn.softmax(mult_link1[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_R_link2 = tf.concat(
                        [
                            tf.nn.softmax(mult_link2[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_R_link3 = tf.concat(
                        [
                            tf.nn.softmax(mult_link3[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)
                    output_prob_R_link4 = tf.concat(
                        [
                            tf.nn.softmax(mult_link4[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_R = tf.stack([output_prob_R_link1, output_prob_R_link2, output_prob_R_link3, output_prob_R_link4])

                    output_prob_C_link1 = tf.concat(
                        [
                            tf.nn.softmax(mult_link1[:,self.num_R * self.num_pkt + self.num_C * j : self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C_link2 = tf.concat(
                        [
                            tf.nn.softmax(mult_link2[:,self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C_link3 = tf.concat(
                        [
                            tf.nn.softmax(mult_link3[:,self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)
                    output_prob_C_link4 = tf.concat(
                        [
                            tf.nn.softmax(mult_link4[:,self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C = tf.stack([output_prob_C_link1, output_prob_C_link2, output_prob_C_link3, output_prob_C_link4])

                    output_R = []
                    for i in range(self.num_link):
                        r_temp = tf.concat(
                            [
                                tf.reshape(tf.argmax(output_prob_R[i, :, self.num_R * j:self.num_R * (j + 1)], axis=1),[-1, 1]) + 1
                                for j in range(self.num_pkt)
                            ], axis=1)
                        output_R.append(r_temp)

                    output_R = tf.stack([output_R[i] for i in range(self.num_link)])

                    output_C = []
                    for i in range(self.num_link):
                        c_temp = tf.concat(
                            [
                                tf.reshape(tf.argmax(output_prob_C[i, :, self.num_C * j:self.num_C * (j + 1)], axis=1), [-1, 1])
                                for j in range(self.num_pkt)
                            ], axis=1)
                        output_C.append(c_temp)
                    output_C = tf.stack([output_C[i] for i in range(self.num_link)])

                self.weights.append(weight)
                self.biases.append(bias)

            temp = (x_ph[:, :-(self.num_link ** 2)] + 1) / 2
            temp = temp * (max_dr_log - min_dr_log) + min_dr_log
            D = pow(10, temp)

            D = tf.reshape(D, [self.num_link, self.batch_size, -1])

            H = tf.reshape(x_ph[:,-(self.num_link**2):], [-1, self.num_link, self.num_link])
            temp = H * std_h_log + avg_h_log
            H = pow(10, temp)

            SINR = []

            for i in range(self.num_link):
                den = self.N # noise power
                for j in range(self.num_link):
                    if j != i:
                        den += output_pw[:,j] * H[:,i,j]
                SINR_per_link = output_pw[:,i] * H[:,i,i] / den
                SINR.append(SINR_per_link)

            SINR = tf.convert_to_tensor(SINR)

            Pout_all_stbc_tf = tf.convert_to_tensor(Pout_all_stbc)

            Pout = []
            SINR_dB = 10*log10(SINR)

            for i in range(self.num_link):
                Pout_each_link = []
                for a in range(self.num_pkt):
                    Pout_temp = []
                    for j in range(self.batch_size):
                        Pout_temp_per_imgs = 0
                        for C_idx in range(self.num_C):
                            for R_idx in range(self.num_R):
                                tmp = tf.cond(tf.less(SINR_dB[i,j],-200),
                                                             lambda: tf.constant(1, tf.float64),
                                                             lambda: tf.cond(tf.math.logical_and(tf.greater_equal(SINR_dB[i, j], -200),tf.less_equal(SINR_dB[i, j], 60)),
                                                                              lambda: self.Linear_interpolation(SINR_dB[i,j], Pout_all_stbc_tf[C_idx, :, R_idx + 1]),
                                                                              lambda: self.Extrapolation(SINR_dB[i,j], Pout_all_stbc_tf[C_idx, :,R_idx + 1])) )

                                Pout_temp_per_imgs = tf.cast(tmp, tf.float64) * output_prob_R[i,j,self.num_R * a + R_idx] * output_prob_C[i,j,self.num_C * a + C_idx] + Pout_temp_per_imgs
                        Pout_temp.append(Pout_temp_per_imgs)
                    Pout_temp = tf.transpose(Pout_temp)
                    Pout_each_link.append(Pout_temp)

                Pout_each_link = tf.transpose(Pout_each_link)
                Pout.append(Pout_each_link)

            Pout = tf.stack([Pout[i] for i in range(self.num_link)])

            R = []
            Cx3 = []
            for i in range(self.num_link):
                R_temp = 1 * output_prob_R[i,:, 0:output_prob_R.shape[1]:self.num_R] + 2 * output_prob_R[i,:,1:output_prob_R.shape[1]:self.num_R] \
                      + 3 * output_prob_R[i,:, 2:output_prob_R.shape[1]:self.num_R] + 4 * output_prob_R[i,:,3:output_prob_R.shape[1]:self.num_R]
                Cx3_temp =  3 * output_prob_C[i,:, 0:output_prob_C.shape[1]:self.num_C] + 4 * output_prob_C[i,:, 1:output_prob_C.shape[1]:self.num_C] + 6 * output_prob_C[i,:, 2:output_prob_C.shape[1]:self.num_C]

                R.append(R_temp)
                Cx3.append(Cx3_temp)

            R = tf.stack([R[i] for i in range(self.num_link)])
            Cx3 = tf.stack([Cx3[i] for i in range(self.num_link)])

            link_idx = np.arange(0, self.num_link, 1).reshape(-1,1)
            link_idx = np.tile(link_idx[:,np.newaxis,:], (1,self.batch_size,1))

            sample_idx = np.arange(0, self.batch_size, 1).reshape([self.batch_size, 1])
            sample_idx = np.tile(sample_idx, (self.num_link,1,1))

            E_D = tf.zeros([self.num_link, self.batch_size, 1], dtype=tf.float64)

            for success_pkt in range(self.num_pkt, 0, -1):
                total_bits = self.max_pkt_size * (tf.reduce_sum(R[:, :, 0:success_pkt] * Cx3[:, :, 0:success_pkt], -1, keep_dims=True) / (self.R_max * self.Cx3_max))

                D_idx1 = tf.cast(total_bits / self.D_step, tf.int32)
                D_idx2 = D_idx1 + 1

                small_v1 = tf.cast(D_idx1 * self.D_step, tf.float64)
                big_v2 = tf.cast(D_idx2 * self.D_step, tf.float64)

                w1 = (big_v2 - total_bits) / (big_v2 - small_v1)
                w2 = (total_bits - small_v1) / (big_v2 - small_v1)

                idx1 = tf.concat([link_idx, sample_idx, D_idx1], -1)
                idx2 = tf.concat([link_idx, sample_idx, D_idx2], -1)

                if success_pkt == self.num_pkt:
                    Distortion = []
                    for i in range(self.num_link):
                        for j in range(self.batch_size):
                            Distortion_temp = tf.cond(total_bits[i,j, 0] < self.max_pkt_size * self.num_pkt,
                                                      lambda: w1[i,j, 0] * tf.gather_nd(D[i,j, :], [D_idx1[i,j, 0]]) + w2[i,j, 0] * tf.gather_nd(D[i,j, :], [D_idx2[i,j, 0]]),
                                                      lambda: tf.gather_nd(D[i,j, :], [D_idx1[i,j, 0]]))
                            Distortion.append(Distortion_temp)

                    value = tf.reshape(Distortion, [self.num_link, self.batch_size, 1])
                else:
                    Distortion = w1 * tf.reshape(tf.gather_nd(D, idx1), [self.num_link, self.batch_size, 1])\
                                 + w2 * tf.reshape(tf.gather_nd(D, idx2), [self.num_link, self.batch_size, 1])
                    value = Distortion * tf.reshape(Pout[:,:, success_pkt], [self.num_link, self.batch_size, 1])

                E_D = (E_D + value) * (1 - tf.reshape(Pout[:,:, success_pkt - 1], [self.num_link, self.batch_size, 1]))

            E_D = E_D + tf.reshape(D[:, :, 0], [self.num_link, self.batch_size, 1]) * tf.reshape(Pout[:,:, 0], [self.num_link, self.batch_size, 1])  # E_D: num_link x batch_size x 1

            psnr = 10 * log10(255 * 255 / E_D)

            loss =  -1 * tf.reduce_mean(input_tensor= psnr)

            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            train = optimizer.minimize(loss)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            start_time_sec = time.time()
            start_time = datetime.datetime.now()
            print('======== Start Time: %s ========\n' % start_time)

            loss_per_h_1batch = np.zeros([num_pl_per_img])
            loss_per_batch = np.zeros([num_batch])
            loss_per_epoch = np.zeros([self.n_epoch])

            R_prob_link1_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_R])
            R_prob_link2_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_R])
            R_prob_link3_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_R])
            R_prob_link4_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_R])

            C_prob_link1_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt*self.num_C])
            C_prob_link2_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_C])
            C_prob_link3_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_C])
            C_prob_link4_per_epoch = np.zeros([self.n_epoch, num_pl_per_img, num_batch, self.num_pkt * self.num_C])


            for e in range(self.n_epoch):
                if (self.mode_shuffle == 'enable'):
                    input_shuffle_dr_link1 = shuffle(input_dr_link1, random_state=seed_shuffle*e)
                    input_shuffle_dr_link2 = shuffle(input_dr_link2, random_state=seed_shuffle*e)
                    input_shuffle_dr_link3 = shuffle(input_dr_link3, random_state=seed_shuffle * e)
                    input_shuffle_dr_link4 = shuffle(input_dr_link4, random_state=seed_shuffle * e)
                    input_shuffle_h = shuffle(input_h, random_state=seed_shuffle*e)
                else:
                    input_shuffle_dr_link1 = input_dr_link1
                    input_shuffle_dr_link2 = input_dr_link2
                    input_shuffle_dr_link3 = input_dr_link3
                    input_shuffle_dr_link4 = input_dr_link4
                    input_shuffle_h = input_h

                for j in range(num_batch):
                    input_batch_link1 = input_shuffle_dr_link1[j * self.batch_size: (j + 1) * self.batch_size]
                    input_batch_link2 = input_shuffle_dr_link2[j * self.batch_size: (j + 1) * self.batch_size]
                    input_batch_link3 = input_shuffle_dr_link3[j * self.batch_size: (j + 1) * self.batch_size]
                    input_batch_link4 = input_shuffle_dr_link4[j * self.batch_size: (j + 1) * self.batch_size]

                    input_h_random = input_shuffle_h[j*num_pl_per_img : (j+1)*num_pl_per_img, :]

                    for h in range(input_h_random.shape[0]):
                        h_vec = np.tile(input_h_random[h, :], (self.batch_size,1))
                        input_batch = np.concatenate((input_batch_link1, input_batch_link2, input_batch_link3, input_batch_link4, h_vec),axis=1)  # 옆으로 붙여짐(열의 방향으로)

                        sess.run(train,feed_dict={x_ph: input_batch})
                        loss_per_h_1batch[h] = sess.run(loss, feed_dict={x_ph: input_batch})
                    loss_per_batch[j] = np.mean(loss_per_h_1batch)
                loss_per_epoch[e] = np.mean(loss_per_batch)

                if (e + 1) % 10 == 0:
                    now_time = datetime.datetime.now()
                    remain_time = (now_time - start_time) * self.n_epoch / (e + 1) - (now_time - start_time)

                    print('epoch= %6d | lr=%g | loss= %8.10g | remain = %s(h:m:s)'% (e + 1, lr, loss_per_epoch[e], remain_time))

            ww, bb = sess.run([self.weights, self.biases])

            dnn_R_link1_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_R_link2_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_R_link3_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_R_link4_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])

            dnn_Cx3_link1_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_Cx3_link2_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_Cx3_link3_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_Cx3_link4_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])

            distort_dnn_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])
            psnr_dnn_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])

            dnn_power_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])

            for m in range(dr_pair):
                input_link1 = input_dr_link1[m * self.batch_size: (m + 1) * self.batch_size]
                input_link2 = input_dr_link2[m * self.batch_size: (m + 1) * self.batch_size]
                input_link3 = input_dr_link3[m * self.batch_size: (m + 1) * self.batch_size]
                input_link4 = input_dr_link4[m * self.batch_size: (m + 1) * self.batch_size]

                all_h = input_h[m*num_pl_per_img : (m+1)*num_pl_per_img, :]

                for h in range(num_pl_per_img):
                    h_vec = np.tile(all_h[h, :], (self.batch_size,1))
                    input_concat = np.concatenate((input_link1, input_link2, input_link3, input_link4, h_vec),axis=1)
                    dnn_power_per_sample[h] = sess.run(output_pw, feed_dict={x_ph: input_concat})

                    dnn_R_link1_per_sample[h] = sess.run(output_R[0, :, :], feed_dict={x_ph: input_concat})
                    dnn_R_link2_per_sample[h] = sess.run(output_R[1, :, :], feed_dict={x_ph: input_concat})
                    dnn_R_link3_per_sample[h] = sess.run(output_R[2, :, :], feed_dict={x_ph: input_concat})
                    dnn_R_link4_per_sample[h] = sess.run(output_R[3, :, :], feed_dict={x_ph: input_concat})

                    dnn_Cx3_link1_per_sample[h] = sess.run(output_C[0, :, :], feed_dict={x_ph: input_concat})
                    dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 2] = 6
                    dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 1] = 4
                    dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 0] = 3

                    dnn_Cx3_link2_per_sample[h] = sess.run(output_C[1, :, :], feed_dict={x_ph: input_concat})
                    dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 2] = 6
                    dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 1] = 4
                    dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 0] = 3

                    dnn_Cx3_link3_per_sample[h] = sess.run(output_C[2, :, :], feed_dict={x_ph: input_concat})
                    dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 2] = 6
                    dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 1] = 4
                    dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 0] = 3

                    dnn_Cx3_link4_per_sample[h] = sess.run(output_C[3, :, :], feed_dict={x_ph: input_concat})
                    dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 2] = 6
                    dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 1] = 4
                    dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 0] = 3


                    distort_dnn_per_sample[h, m, :], Pout_after = self.Calc_distort(input_dr_not_scaled_link1[m, :],input_dr_not_scaled_link2[m, :],input_dr_not_scaled_link3[m, :],input_dr_not_scaled_link4[m, :],
                                                                                    input_h_not_scaled[m*num_pl_per_img+h, :],
                                                                                    dnn_R_link1_per_sample[h, m, :],dnn_R_link2_per_sample[h, m, :],dnn_R_link3_per_sample[h, m, :],dnn_R_link4_per_sample[h, m, :],
                                                                                    dnn_Cx3_link1_per_sample[h,m,:], dnn_Cx3_link2_per_sample[h,m,:],dnn_Cx3_link3_per_sample[h,m,:], dnn_Cx3_link4_per_sample[h,m,:],
                                                                                    dnn_power_per_sample[h,m,:], Pout_all_stbc)

                    psnr_dnn_per_sample[h, m, :] = 10 * np.log10(255 * 255 /distort_dnn_per_sample[h, m, :])

        sess.close()

        wb_path = file_path +'./weight_bias'
        if not os.path.exists(wb_path):
            os.makedirs(wb_path)

        for i in range(len(self.layer_dim_list)):
            file_write(wb_path + '\W' + str(i + 1) + '_lr' + str(format(lr,"1.0e")) + '.dat', 'w', ww[i][:, :])
            file_write(wb_path + '\B' + str(i + 1) + '_lr' + str(format(lr,"1.0e")) + '.dat', 'w', bb[i][:])

        lr_path = file_path + '\lr=' + str(lr)
        if not os.path.exists(lr_path):
            os.makedirs(lr_path)

        epoch = np.arange(0, self.n_epoch, 1)
        with open(lr_path + '\loss_per_epoch_train.dat', 'w') as f4:
            for j in range(self.n_epoch):
                f4.write('%d\t %8.10g\n' % (epoch[j] + 1, loss_per_epoch[j]))
            f4.close()

        for m in range(dr_pair):
            with open(lr_path + '\psnr_dnn_train.dat', 'a') as f_p:
                for h in range(num_pl_per_img):
                    f_p.write('%g\t' % (lr))
                    for i in range(self.num_link):
                        f_p.write('%10.10g  ' %psnr_dnn_per_sample[h, m, i])
                    f_p.write('\n')

        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now() - start_time))

    def test_dnn(self, input_dr, input_h, num_pl_per_img, lr, Pout_all_stbc, file_path):
        start_time = datetime.datetime.now()
        print('======== Start Time: %s ========\n' % start_time)

        dr_pair = int(input_dr.shape[0] / self.num_link)

        with open(file_path + 'input_scaling_param.dat', 'r') as f:
            lines = f.readlines()

        max_dr_log = float(lines[0].strip())
        min_dr_log = float(lines[1].strip())

        avg_h_log = float(lines[2].strip())
        std_h_log = float(lines[3].strip())

        input_dr_not_scaled = input_dr
        input_dr_log = np.log10(input_dr)

        temp = (input_dr_log - min_dr_log) / (max_dr_log - min_dr_log)
        input_dr = 2 * temp - 1

        input_dr_link1 = input_dr[:dr_pair, :]
        input_dr_link2 = input_dr[dr_pair:2 * dr_pair, :]
        input_dr_link3 = input_dr[2 * dr_pair:3 * dr_pair, :]
        input_dr_link4 = input_dr[3 * dr_pair:, :]

        input_dr_not_scaled_link1 = input_dr_not_scaled[:dr_pair]
        input_dr_not_scaled_link2 = input_dr_not_scaled[dr_pair:2 * dr_pair, :]
        input_dr_not_scaled_link3 = input_dr_not_scaled[2 * dr_pair:3 * dr_pair, :]
        input_dr_not_scaled_link4 = input_dr_not_scaled[3 * dr_pair:, :]

        input_h_not_scaled = input_h
        input_h_log = np.log10(input_h)

        input_h = (input_h_log - avg_h_log) / std_h_log

        with tf.device('/CPU:0'):
            tf.reset_default_graph()
            x_ph = tf.placeholder(tf.float64, shape=[None, input_dr.shape[1] * self.num_link + self.num_link ** 2])

            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input_dr.shape[1]*self.num_link+self.num_link**2
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i-1]
                    out_dim = self.layer_dim_list[i]

                weight = np.zeros([in_dim, out_dim], dtype=np.float64)
                bias= np.zeros(out_dim, dtype=np.float64)

                wb_path = file_path +'./weight_bias'
                weight = file_read(wb_path + "\W" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", weight)
                bias = file_read(wb_path + "\B" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", bias)

                weight = tf.convert_to_tensor(weight)
                bias = tf.convert_to_tensor(bias)

                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list) - 1:
                    out_layer = tf.nn.relu(mult)
                else:
                    mult_link1 = mult[:, : self.num_pkt * (self.num_R + self.num_C)]
                    mult_link2 = mult[:,self.num_pkt * (self.num_R + self.num_C): 2 * self.num_pkt * (self.num_R + self.num_C)]
                    mult_link3 = mult[:, 2 * self.num_pkt * (self.num_R + self.num_C): 3 * self.num_pkt * (self.num_R + self.num_C)]
                    mult_link4 = mult[:, 3 * self.num_pkt * (self.num_R + self.num_C): 4 * self.num_pkt * (self.num_R + self.num_C)]
                    mult_pw = mult[:, - self.num_link:]

                    output_pw = tf.nn.sigmoid(mult_pw) + 1e-100

                    output_prob_R_link1 = tf.concat(
                        [
                            tf.nn.softmax(mult_link1[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_R_link2 = tf.concat(
                        [
                            tf.nn.softmax(mult_link2[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)


                    output_prob_R_link3 = tf.concat(
                        [
                            tf.nn.softmax(mult_link3[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)
                    output_prob_R_link4 = tf.concat(
                        [
                            tf.nn.softmax(mult_link4[:, self.num_R * j:self.num_R * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_R = tf.stack([output_prob_R_link1, output_prob_R_link2, output_prob_R_link3, output_prob_R_link4])

                    output_prob_C_link1 = tf.concat(
                        [
                            tf.nn.softmax(mult_link1[:,self.num_R * self.num_pkt + self.num_C * j: self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C_link2 = tf.concat(
                        [
                            tf.nn.softmax(mult_link2[:,
                                          self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C_link3 = tf.concat(
                        [
                            tf.nn.softmax(mult_link3[:,self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)
                    output_prob_C_link4 = tf.concat(
                        [
                            tf.nn.softmax(mult_link4[:,self.num_R * self.num_pkt + self.num_C * j:self.num_R * self.num_pkt + self.num_C * (j + 1)])
                            for j in range(self.num_pkt)
                        ]
                        , axis=1)

                    output_prob_C = tf.stack([output_prob_C_link1, output_prob_C_link2, output_prob_C_link3, output_prob_C_link4])

                    output_R = []
                    for i in range(self.num_link):
                        r_temp = tf.concat(
                            [
                                tf.reshape(tf.argmax(output_prob_R[i, :, self.num_R * j:self.num_R * (j + 1)], axis=1),[-1, 1]) + 1
                                for j in range(self.num_pkt)
                            ], axis=1)
                        output_R.append(r_temp)

                    output_R = tf.stack([output_R[i] for i in range(self.num_link)])

                    output_C = []
                    for i in range(self.num_link):
                        c_temp = tf.concat(
                            [
                                tf.reshape(tf.argmax(output_prob_C[i, :, self.num_C * j:self.num_C * (j + 1)], axis=1),[-1, 1])
                                for j in range(self.num_pkt)
                            ], axis=1)
                        output_C.append(c_temp)
                    output_C = tf.stack([output_C[i] for i in range(self.num_link)])

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            dnn_R_link1_per_sample = np.zeros([num_pl_per_img,  dr_pair, self.num_pkt])
            dnn_R_link2_per_sample = np.zeros([num_pl_per_img,  dr_pair, self.num_pkt])
            dnn_R_link3_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_R_link4_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])

            dnn_Cx3_link1_per_sample = np.zeros([num_pl_per_img,  dr_pair, self.num_pkt])
            dnn_Cx3_link2_per_sample = np.zeros([num_pl_per_img,  dr_pair, self.num_pkt])
            dnn_Cx3_link3_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
            dnn_Cx3_link4_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])

            distort_dnn_per_sample = np.zeros([num_pl_per_img,  dr_pair, self.num_link])
            psnr_dnn_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])

            dnn_power_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])
            capa_dnn_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])

            avg_psnr_dnn_per_sample = np.zeros([dr_pair, self.num_link])

            for h in range(num_pl_per_img):
                h_vec = input_h[h::num_pl_per_img, :]
                input_concat = np.concatenate((input_dr_link1, input_dr_link2, input_dr_link3,input_dr_link4, h_vec), axis=1)

                dnn_power_per_sample[h] = sess.run(output_pw, feed_dict={x_ph: input_concat})

                dnn_R_link1_per_sample[h] =  sess.run(output_R[0,:,:], feed_dict={x_ph: input_concat})
                dnn_R_link2_per_sample[h] = sess.run(output_R[1, :, :], feed_dict={x_ph: input_concat})
                dnn_R_link3_per_sample[h] = sess.run(output_R[2, :, :], feed_dict={x_ph: input_concat})
                dnn_R_link4_per_sample[h] = sess.run(output_R[3, :, :], feed_dict={x_ph: input_concat})

                dnn_Cx3_link1_per_sample[h] = sess.run(output_C[0, :, :], feed_dict={x_ph: input_concat})
                dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 2] = 6
                dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 1] = 4
                dnn_Cx3_link1_per_sample[dnn_Cx3_link1_per_sample == 0] = 3

                dnn_Cx3_link2_per_sample[h] = sess.run(output_C[1, :, :], feed_dict={x_ph: input_concat})
                dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 2] = 6
                dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 1] = 4
                dnn_Cx3_link2_per_sample[dnn_Cx3_link2_per_sample == 0] = 3

                dnn_Cx3_link3_per_sample[h] = sess.run(output_C[2, :, :], feed_dict={x_ph: input_concat})
                dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 2] = 6
                dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 1] = 4
                dnn_Cx3_link3_per_sample[dnn_Cx3_link3_per_sample == 0] = 3

                dnn_Cx3_link4_per_sample[h] = sess.run(output_C[3, :, :], feed_dict={x_ph: input_concat})
                dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 2] = 6
                dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 1] = 4
                dnn_Cx3_link4_per_sample[dnn_Cx3_link4_per_sample == 0] = 3

                for m in range(dr_pair):
                    distort_dnn_per_sample[h, m, :], Pout_after = self.Calc_distort(input_dr_not_scaled_link1[m, :],input_dr_not_scaled_link2[m, :],input_dr_not_scaled_link3[m, :],input_dr_not_scaled_link4[m, :],
                                                                                    input_h_not_scaled[m * num_pl_per_img + h, :],
                                                                                    dnn_R_link1_per_sample[h, m, :],dnn_R_link2_per_sample[h, m, :],dnn_R_link3_per_sample[h, m, :],dnn_R_link4_per_sample[h, m, :],
                                                                                    dnn_Cx3_link1_per_sample[h, m, :],dnn_Cx3_link2_per_sample[h, m, :],dnn_Cx3_link3_per_sample[h, m, :],dnn_Cx3_link4_per_sample[h, m, :],
                                                                                    dnn_power_per_sample[h, m, :],Pout_all_stbc)

                    psnr_dnn_per_sample[h, m, :] = 10 * np.log10(255 * 255 / distort_dnn_per_sample[h, m, :])

            for i in range(self.num_link):
                for m in range(dr_pair):
                    avg_psnr_dnn_per_sample[m,i] = np.mean(psnr_dnn_per_sample[:,m,i])

            sess.close()

        lr_path = file_path + '\lr=' + str(lr)

        for m in range(dr_pair):

            with open(lr_path + '\R_link1_dnn_test.dat', 'a') as f, open(lr_path + '\R_link2_dnn_test.dat', 'a') as f2,\
                    open(lr_path + '\R_link3_dnn_test.dat', 'a') as f3, open(lr_path + '\R_link4_dnn_test.dat', 'a') as f4:
                for h in range(num_pl_per_img):
                    f.write('%g\t' % (lr))
                    f2.write('%g\t' % (lr))
                    f3.write('%g\t' % (lr))
                    f4.write('%g\t' % (lr))
                    for k in range(self.num_pkt):
                        f.write('%10.10g  ' %dnn_R_link1_per_sample[h, m, k])
                        f2.write('%10.10g  ' % dnn_R_link2_per_sample[h, m, k])
                        f3.write('%10.10g  ' % dnn_R_link3_per_sample[h, m, k])
                        f4.write('%10.10g  ' % dnn_R_link4_per_sample[h, m, k])
                    f.write('\n')
                    f2.write('\n')
                    f3.write('\n')
                    f4.write('\n')

            with open(lr_path + '\Cx3_link1_dnn_test.dat', 'a') as f5, open(lr_path + '\Cx3_link2_dnn_test.dat', 'a') as f6,\
                    open(lr_path + '\Cx3_link3_dnn_test.dat', 'a') as f7, open(lr_path + '\Cx3_link4_dnn_test.dat', 'a') as f8:
                for h in range(num_pl_per_img):
                    f5.write('%g\t' % (lr))
                    f6.write('%g\t' % (lr))
                    f7.write('%g\t' % (lr))
                    f8.write('%g\t' % (lr))
                    for k in range(self.num_pkt):
                        f5.write('%10.10g  ' % dnn_Cx3_link1_per_sample[h, m, k])
                        f6.write('%10.10g  ' % dnn_Cx3_link2_per_sample[h, m, k])
                        f7.write('%10.10g  ' % dnn_Cx3_link3_per_sample[h, m, k])
                        f8.write('%10.10g  ' % dnn_Cx3_link4_per_sample[h, m, k])

                    f5.write('\n')
                    f6.write('\n')
                    f7.write('\n')
                    f8.write('\n')

            with open(lr_path + '\psnr_dnn_test.dat', 'a') as f_p:
                for h in range(num_pl_per_img):
                    f_p.write('%g\t' % (lr))
                    for i in range(self.num_link):
                        f_p.write('%10.10g  ' %psnr_dnn_per_sample[h, m, i])
                    f_p.write('\n')


            with open(lr_path + '\power_dnn_test.dat', 'a') as f_pw:
                for h in range(num_pl_per_img):
                    f_pw.write('%g\t' % (lr))
                    for i in range(self.num_link):
                        f_pw.write('%10.10g  ' %dnn_power_per_sample[h, m, i])
                    f_pw.write('\n')

            with open(lr_path + '\capa_dnn_test.dat', 'a') as f_capa:
                for h in range(num_pl_per_img):
                    f_capa.write('%g\t' % (lr))
                    for i in range(self.num_link):
                        f_capa.write('%10.10g  ' %capa_dnn_per_sample[h, m, i])
                    f_capa.write('\n')

        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now() - start_time))




