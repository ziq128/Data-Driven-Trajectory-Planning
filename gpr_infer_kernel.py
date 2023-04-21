import numpy as np
import pandas as pd
import random
import sys
import time
from GPR_MPC_FINAL import Data_Pre, Logger
from scipy.stats import norm
import GPy
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error


plt.rc('font', family='Times New Roman')
from matplotlib import rcParams
rcParams['mathtext.default'] = 'regular'
rcParams.update({'font.size': 15})

# data_init = pd.read_csv("data_init.csv")
# data = pd.read_csv("df_training_3s_2.csv")
#
# datapre = Data_Pre.Data_pre(data, data_init)
# df_training, SPAT, out = datapre.main()
#
# df_training.to_csv('df_3s_for_kernel.csv')


df_training = pd.read_csv("../GPR_MPC_FINAL/df_3s_for_kernel.csv")
# df_training = pd.read_csv("df_5s_for_kernel.csv")

seed = 12
seed = 42

np.random.seed(seed)

def gpr_model(lenSca, variance, dim = 2):
    ker = GPy.kern.RBF(input_dim=dim, lengthscale=lenSca, variance=variance, ARD=True) + GPy.kern.White(2)

    # create simple GP model
    m = GPy.models.GPRegression(X_train_0, Y_train_0, ker)
    m.optimize(messages=True, max_f_eval=1000)
    #fig.savefig('GPR_event0_1Input', dpi=80)
    return m

def gpr_predict(vi, yi, m):
    inputArr = np.expand_dims(np.array([vi, (600 - yi) / 1000]), 0)
    mu_dMin, sigma_dMs2 = m.predict(inputArr)
    return mu_dMin, sigma_dMs2

#https://nipunbatra.github.io/blog/ml/2020/03/26/gp.html#Disclaimer
#increasing the scale means that a point becomes more correlated with a further away point.
#increasing the length scale would mean making the regression smoother
#
# vehicle_test, df_training, out = datapre.main()

df_training['DIS'] = (600 - df_training[['H_local_y']]) / 1000
Host_veh = df_training[['Host_vel']] / 3.6
Preced_veh = df_training[['Preceding_vel']] / 3.6
df_training['vel_diff'] = Preced_veh['Preceding_vel'] - Host_veh['Host_vel']
vel = df_training['Host_vel'].values

# safe distance standard

D_diff_tr = np.zeros(len(vel))
for i in range(len(vel)):
    if vel[i] < 40:
        D_min_standard = 30
    elif 40 <= vel[i] < 50:
        D_min_standard = 50
    else:
        D_min_standard = vel[i]

    D_diff_tr[i] = (df_training['Distance_Diff_Ts'].values[i] - D_min_standard) / D_min_standard

# df_training['D_diff_tr'] = df_training['Distance_Diff']
df_training['D_diff_tr'] = D_diff_tr

training_scale = 0.1
# training_scale = 0.3
# training_scale = 0.5
# training_scale = 0.7
test_scale = 0.1

num_tr = random.sample(list(df_training.index), int(np.floor(len(vel) * training_scale)))

ts = set(list(df_training.index)) - set(num_tr)

num_ts = random.sample(list(ts), int(np.floor(len(vel) * test_scale)))

df_tr = df_training.reindex(num_tr)

df_ts = df_training.reindex(num_ts)

#X_train_0 = df_tr[['DIS']]
X_train_0 = df_tr[['vel_diff', 'DIS']]
Y_train_0 = df_tr[['D_diff_tr']]


X_test_0 = df_ts[['vel_diff', 'DIS']]
Y_test_0 = df_ts[['D_diff_tr']]
'''
plt.plot(X_train_0, Y_train_0, 'kx', mew = 2, label='Train Points')
plt.plot(X_test_0, Y_test_0, 'mo', mew = 2, label='Test Points')
'''
plt.scatter(X_train_0['DIS'], X_train_0['vel_diff'], s=Y_train_0['D_diff_tr'])

plt.show()

ls = [0.05, 0.25, 0.5, 1., 2., 4.]
va = [0.01, 0.05, 0.25, 0.5, 1., 2., 4.]

# error = np.zeros([len(ls), len(va)])
# for i in range(len(ls)):
#     for j in range(len(va)):
#         gprM = gpr_model(lenSca=ls[i], variance=va[j])
#         Y_pre = gprM.predict(np.array(X_test_0))
#         error[i, j] = mean_absolute_error(Y_test_0, Y_pre[0].flatten())
#
#
# print(np.min(error))
# x, y = np.where(error == np.min(error))
#
# lenScale = ls[x[0]]
# variance = va[y[0]]
#
# print(lenScale, variance)

# # 5s
# lenScale = 0.25
# variance = 4
# 3s
lenScale = 0.25
variance = 4

start1 = time.time()
# plot
m = gpr_model(lenSca=lenScale, variance=variance)
end1 = time.time()

training_time = end1-start1

start2 = time.time()
Y_pre = m.predict(np.array(X_test_0))
end2 = time.time()


error = mean_absolute_error(Y_test_0, Y_pre[0].flatten())

sys.stdout = Logger.Logger('datalog.txt')

pred_time = 3

print("Predicton time: %d, error is (%.3f) with (%.2f) training,(%.2f) test, training time:(%.2f), randomseed:(%.2f)"
      %(pred_time, error, training_scale, test_scale, training_time, seed))

fig1 = GPy.plotting.plotting_library().figure()
m.plot(figure=fig1)


slices = [0.08]

figure = GPy.plotting.plotting_library().figure(1, 1, figsize=(10, 6))

for i, y in zip(range(1), slices):
    # if i == 0:
    #     m.plot(figure=figure, fixed_inputs=[(1, y)], row=(i + 1), plot_data=True, ylabel="Distance_Diff",
    #            title="slice at distance = 0.05km")
    # elif i == 1:
    #     m.plot(figure=figure, fixed_inputs=[(1, y)], row=(i + 1), plot_data=True, ylabel="Distance_Diff",
    #            title="slice at distance = 0.08km")
    # else:
    m.plot(figure=figure, fixed_inputs=[(1, y)], row=(i + 1), plot_data=True, fontsize=40, xlabel='Relative Velocity(m/s)',
               ylabel="Normalized Relative Distance", title="Slice at distance = 80m & prediction time = 3s")

#figure.savefig('GPR_model_3s_slice_2.png', dpi=100, format='png')
figure.savefig('GPR_model_3s_slice_tr10.png', dpi=100, format='png')

