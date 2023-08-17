import os.path
import glob
import csv
import numpy as np
from dnn_link2 import Dnn

num_epoch = 500
Batch_Size = 1

lr = np.array([3e-5])
# lr = np.array([1e-4])
# lr = np.array([3e-4])
# lr = np.array([1e-3])
# lr = np.array([3e-3])
# lr = np.array([6e-3])
# lr = np.array([1e-2])

no_link = 2

no_pkt = 3
# no_pkt = 16

Nt = 2
Nr = 2

R = np.array([1,2,3,4]).reshape([1,-1])
C = np.array([1,4/3,2]).reshape([1,-1])

noise = 1e-12*(10**(-0.7))*4

Pout_all_snr_stbc = np.zeros([C.shape[1], 60-(-200)+1, R.shape[1]+1])
Pout_all_snr_stbc[:,:,0] = np.arange(-200,61,1)

file_list = ['OSTBC', 'DBLAST', 'VBLAST']
for i in range(len(file_list)):
    for r in range(R.shape[1]):
        filename = 'data/Pout/' + file_list[i] + '_K00_R' + str(R[0, r]) + '_2x2.dat'
        with open(filename, 'r') as f:
            rdr = csv.reader(f, delimiter='\t')
            temp = np.array(list(rdr), dtype=np.float64)
            data = temp.reshape([1, 261, 2])
            Pout_all_snr_stbc[i,:,r+1] = data[0,:,1]

image_size = 512*512*1
maximum_pkt_size = image_size / no_pkt

D_STEP = 2**14
STEP = np.arange(0, int(np.ceil(image_size/64)) + 1, int(D_STEP/64))

num_sample_train = 1600
num_sample_test = 400

i = 0
num_total_dr = num_sample_train+num_sample_test
total_input_dr = np.zeros([num_total_dr, STEP.shape[0]])

input_path = 'data/D_R/'
for input_file in glob.glob(os.path.join(input_path, 'distort_a*')):
    with open(input_file, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        temp = np.array(list(rdr), dtype=np.float64)
        total_input_dr[i, :] = temp[STEP, 1]
        i = i + 1
    if i == num_total_dr:
        break

input_dr_train = total_input_dr[:num_sample_train,:]
input_dr_test = total_input_dr[-num_sample_test:,:]

num_pl_per_img_train = 20
num_pl_per_img_test = 20

filename_h = 'data\input_data\channel_' + str(no_link) + 'links_h16000(train)_alpha_3.dat'
with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')
input_h_train = np.reshape(input_h_temp[:16000,:],[-1,4])

filename_h = 'data\input_data\channel_' + str(no_link) + 'links_h4000(test)_alpha_3.dat'
with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')
input_h_test = np.reshape(input_h_temp[:4000,:],[-1,4])


Layer_dim_list = [16, R.shape[1]*no_link*no_pkt+C.shape[1]*no_link*no_pkt + no_link] # link 별 power 할당 추가
dnnObj = Dnn(mode_shuffle = 'disable',batch_size=Batch_Size, n_epoch=num_epoch, layer_dim_list=Layer_dim_list, max_pkt_size=maximum_pkt_size,
             num_pkt=no_pkt, D_step = D_STEP, r=R, c=C, num_link=no_link, N=noise, nr=Nr, nt=Nt)

file_path = '.\\dnn_result\\num_link=' + str(no_link) + ',pkt=' + str(no_pkt) +'\\'+'input_h_train='+str(input_h_train.shape[0])+','+'input_h_test='+str(input_h_test.shape[0])+','+'num_pl_per_img='+str(num_pl_per_img_train)+ '\\trainimg='+ str(num_sample_train) + ',testimg='+ str(num_sample_test) +'\\'

if not os.path.exists(file_path):
    os.makedirs(file_path)

for l_idx in range(lr.shape[0]):
    lr_path = file_path + 'lr=' + str(lr[l_idx]) +'\\'
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
    with open(lr_path+ 'psnr_dnn_train.dat', 'w') as f:
        pass

for j in range(lr.shape[0]):
    dnnObj.train_dnn(input_dr_train, input_h_train, num_pl_per_img_train, lr[j], Pout_all_snr_stbc, file_path )

for l_idx in range(lr.shape[0]):
    lr_path = file_path + 'lr=' + str(lr[l_idx]) +'\\'
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
    with open(lr_path + 'psnr_dnn_test.dat', 'w') as f, open(lr_path + 'power_dnn_test.dat', 'w') as f2:
        pass
    with open(lr_path + 'R_link1_dnn_test.dat', 'w') as f3, open(lr_path + 'R_link2_dnn_test.dat', 'w') as f4, open(lr_path + 'capa_dnn_test.dat', 'w') as f5:
        pass
    with open(lr_path + 'Cx3_link1_dnn_test.dat', 'w') as f6, open(lr_path + 'Cx3_link2_dnn_test.dat', 'w') as f7:
        pass

for j in range(lr.shape[0]):
    dnnObj.test_dnn(input_dr_test, input_h_test, num_pl_per_img_test, lr[j], Pout_all_snr_stbc, file_path)










