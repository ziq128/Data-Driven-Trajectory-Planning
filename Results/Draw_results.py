import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GPR_MPC_FINAL import Data_Pre
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rc('font', family='Times New Roman')
from matplotlib import rcParams
rcParams['mathtext.default'] = 'regular'
rcParams.update({'font.size': 18})
# fs = 13

fuelopt_5 = np.load('../../GPR_MPC_FINAL/Results/Fuel_opt0.05.npy')
fuelopt_15 = np.load('../../GPR_MPC_FINAL/Results/Fuel_opt0.15.npy')
fuelopt_25 = np.load('../../GPR_MPC_FINAL/Results/Fuel_opt0.25.npy')
fuelopt_35 = np.load('../../GPR_MPC_FINAL/Results/Fuel_opt0.35.npy')
# fuelopt_45 = np.load('Results/Fuel_opt0.45.npy')

fuelorig_5 = np.load('../../GPR_MPC_FINAL/Results/Fuel_orig0.05.npy')
fuelorig_15 = np.load('../../GPR_MPC_FINAL/Results/Fuel_orig0.15.npy')
fuelorig_25 = np.load('../../GPR_MPC_FINAL/Results/Fuel_orig0.25.npy')
fuelorig_35 = np.load('../../GPR_MPC_FINAL/Results/Fuel_orig0.35.npy')
# fuelorig_45 = np.load('Results/Fuel_orig0.45.npy')


data_5 = pd.read_csv('../../GPR_MPC_FINAL/Results/data_beta0.05.csv')
data_15 = pd.read_csv('../../GPR_MPC_FINAL/Results/data_beta0.15.csv')
data_25 = pd.read_csv('../../GPR_MPC_FINAL/Results/data_beta0.25.csv')
data_35 = pd.read_csv('../../GPR_MPC_FINAL/Results/data_beta0.35.csv')
# data_45 = pd.read_csv('Results/data_beta0.45.csv')

data = pd.read_csv("../../GPR_MPC_FINAL/df_training_5s_2.csv")
data_init = pd.read_csv("../../GPR_MPC_FINAL/data_init.csv")
datapre = Data_Pre.Data_pre(data, data_init)
df_Training, SPAT, out = datapre.main()

figure1 = plt.figure(figsize=(12, 8), dpi=100)
ax1 = figure1.add_subplot(2, 2, 1)
t_test = len(eval(data_5['sHost'][0]))
t = np.linspace(0, t_test - 1, t_test)
t_sp = np.linspace(0,  len(SPAT[0]) - 1,  len(SPAT[0]))

l1 = ax1.plot(t, eval(data_5['sLead_new'][0]), color='coral', linestyle='--')
l2 = ax1.plot(t, eval(data_5['sTest_new'][0])[:t_test], 'k')
l3 = ax1.plot(t, eval(data_5['sHost'][0]), 'g')
l4 = ax1.plot(t, eval(data_15['sHost'][0]), 'royalblue')
l5 = ax1.plot(t, eval(data_25['sHost'][0]), 'violet')
# ax1.plot(t, eval(data_35['sHost'][0]), 'gold')
l6 = ax1.plot(t_sp, 610 * SPAT[0], 'r', label='Traffic light', linewidth=2)
ax1.set_xlabel('Time(secs)')
ax1.set_ylabel('Location(m)')
ax1.title.set_text('Trajectories of vehicle0')

ax2 = figure1.add_subplot(2, 2, 2)
t_test = len(eval(data_5['sHost'][1]))
t = np.linspace(0, t_test - 1, t_test)
t_sp = np.linspace(0,  len(SPAT[1]) - 1,  len(SPAT[1]))
l1 = ax2.plot(t, eval(data_5['sLead_new'][1]), color='coral', linestyle='--')
l2 = ax2.plot(t, eval(data_5['sTest_new'][1])[:t_test], 'k')
l3 = ax2.plot(t, eval(data_5['sHost'][1]), 'g')
l4 = ax2.plot(t, eval(data_15['sHost'][1]), 'royalblue')
l5 = ax2.plot(t, eval(data_25['sHost'][1]), 'violet')
# ax2.plot(t, eval(data_35['sHost'][1]), 'gold')
l6 = ax2.plot(t_sp, 610 * SPAT[1], 'r', label='Traffic light', linewidth=2)
ax2.set_xlabel('Time(secs)')
ax2.set_ylabel('Location(m)')
ax2.title.set_text('Trajectories of vehicle1')

ax3 = figure1.add_subplot(2, 2, 3)

t_test = len(eval(data_5['sHost'][2]))
t = np.linspace(0, t_test - 1, t_test)
t_sp = np.linspace(0,  len(SPAT[2]) - 1,  len(SPAT[2]))

l1 = ax3.plot(t, eval(data_5['sLead_new'][2]), color='coral', linestyle='--')
l2 = ax3.plot(t, eval(data_5['sTest_new'][2])[:t_test], 'k')
l3 = ax3.plot(t, eval(data_5['sHost'][2]), 'g')
l4 = ax3.plot(t, eval(data_15['sHost'][2]), 'royalblue')
l5 = ax3.plot(t, eval(data_25['sHost'][2]), 'violet')
# ax3.plot(t, eval(data_35['sHost'][2]), 'gold')
l6 = ax3.plot(t_sp, 610 * SPAT[2], 'r', label='Traffic light', linewidth=2)

ax3.set_xlabel('Time(secs)')
ax3.set_ylabel('Location(m)')
ax3.title.set_text('Trajectories of vehicle2')

ax4 = figure1.add_subplot(2, 2, 4)

t_test = len(eval(data_5['sHost'][3]))
t = np.linspace(0, t_test - 1, t_test)
t_sp = np.linspace(0,  len(SPAT[3]) - 1,  len(SPAT[3]))

ax4.plot(t[:-1], eval(data_5['sLead_new'][3]), color='coral', linestyle='--', label='Safety constraints')
ax4.plot(t, eval(data_5['sTest_new'][3])[:t_test], 'k', label='Host Vehicle Original Trajectory')
ax4.plot(t, eval(data_5['sHost'][3]), 'g', label=r'Optimized Trajectory with $\beta$ = 0.05')
ax4.plot(t, eval(data_15['sHost'][3]), 'royalblue', label=r'Optimized Trajectory with $\beta$ = 0.15')
ax4.plot(t, eval(data_25['sHost'][3]), 'violet', label=r'Optimized Trajectory with $\beta$ = 0.25')
# ax4.plot(t, eval(data_35['sHost'][3]), 'gold', label=r'Optimized Trajectory with $\beta$ = 0.35')
ax4.plot(t_sp, 610 * SPAT[3], 'r', label='Traffic light', linewidth=2)
# ax4.legend()
ax4.set_xlabel('Time(secs)')
ax4.set_ylabel('Location(m)')
ax4.title.set_text('Trajectories of vehicle3')

plt.tight_layout()

lines, labels = figure1.axes[-1].get_legend_handles_labels()

figure1.legend(lines, labels, loc='lower right', prop ={'size': 10})
figure1.show()

figure1.savefig('plot2.png')



'''
COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]
result = np.vstack((fuelorig_5[0:4], fuelopt_5[0:4], fuelopt_15[0:4],
                     fuelopt_25[0:4]))
result = result.T

x_tickets = np.array(['Original', r'$\beta$ = 0.05', r'$\beta$ = 0.15',
                   r'$\beta$ = 0.25'])
x = np.linspace(0, x_tickets.shape[0] - 1, x_tickets.shape[0])

y_tickets = np.array(['vehicle0', 'vehicle1', 'vehicle2', 'vehicle3'])
y = np.linspace(0, y_tickets.shape[0] - 1, y_tickets.shape[0])

xx, yy = np.meshgrid(x, y, copy=False)

color_list = []
for i in range(len(y)):
    c = COLOR[i]
    color_list.append([c] * len(x))
color_list = np.asarray(color_list)


# print(color_list)
xx_flat, yy_flat, result_flat, color_flat = \
    xx.ravel(), yy.ravel(), result.ravel(), color_list.ravel()
# print(xx_flat)
# print(yy_flat)

figure2 = plt.figure(figsize=(5, 5), dpi=150)
ax = figure2.add_subplot(111, projection='3d')
ax.bar3d(xx_flat - 0.35, yy_flat - 0.35, 0, 0.7, 0.7, result_flat,
         color=color_flat,
         edgecolor='black',
         shade='True')

# 座标轴名
ax.set_ylabel("Vehicles")
ax.set_xlabel(r"$Trajectory$")
ax.set_zlabel("Fuel Consumption(ml/m)")

# 座标轴范围
ax.set_zlim((0, 6))

# 座标轴刻度标签
# 似乎要 `set_*ticks` 先，再 `set_*ticklabels`
# has to call `set_*ticks` to mount `ticklabels` to corresponding `ticks` ?
ax.set_xticks(x)
ax.set_xticklabels(x_tickets)
ax.set_yticks(y)
ax.set_yticklabels(y_tickets)


# 保存
# plt.tight_layout()
figure2.savefig("bar3d.png", bbox_inches='tight', pad_inches=0)
plt.close()
'''