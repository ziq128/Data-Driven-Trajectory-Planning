import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import json  # get additional solution information

from GPR_MPC_FINAL import MPC_Frame, Data_Pre, Logger
from scipy.stats import norm
from more_itertools import chunked

plt.rc('font', family='Times New Roman')
from matplotlib import rcParams
rcParams['mathtext.default'] = 'regular'
fs = 13

data = pd.read_csv("df_training_5s_2.csv")
data_init = pd.read_csv("./data_init.csv")
datapre = Data_Pre.Data_pre(data, data_init)
df_Training, SPAT, out = datapre.main()

timestep = 5

mpcframe = MPC_Frame.MPC_Frame(event=0, timestep=timestep, df=df_Training.sample(n=5000))

sHost, vHost, uHost, sTest_new, sLead_new, sPreced_new = [[] for x in range(6)]
Fuel_opt_test = np.zeros(len(SPAT))
Fuel_orig_test = np.zeros(len(SPAT))

# gprM = mpcframe.gpr_model(lenSca=.25, variance= 4)
gprM = mpcframe.gpr_model2(lenSca=0.25, variance=4)

beta = 0.15
n = 0

SP = []

for i in range(len(SPAT)):
    print('vehicle ' + str(i))

    if out['sLead'][i] == []:
        continue

    t_test = int(np.rint(len(out['sLead'][i]) / 10))

    # s_test is the original trajectory of host vehicle
    # slead
    slead_new = out['sLead'][i][::10]
    vlead_new = [sum(x) / len(x) for x in chunked(out['vLead'][i], 10)]
    alead_new = np.diff(np.array(vlead_new)/3.6)

    if np.isnan(slead_new).any():
        continue
    else:
        pass

    s_test_new = out['s_test'][i][::10]
    v_test_new = [sum(x) / len(x) for x in chunked(out['vel_test'][i], 10)]

    Fuel_orig_test[i] = mpcframe.fv(v_test_new[:t_test + 1]) / (s_test_new[t_test + 1]-s_test_new[0])

    s0 = out['s0_host'][i]
    v0 = out['v0_host'][i] / 3.6
    a0 = v_test_new[1]/3.6 - v_test_new[0]/3.6

    cycles = t_test - timestep

    u_cont = np.ones(cycles)

    a = np.ones(cycles + 1)
    v = np.ones(cycles + 1)
    s = np.ones(cycles + 1)

    v[0] = v0
    s[0] = s0
    a[0] = a0

    d = np.zeros(cycles + 1)

    sp = np.zeros(cycles + 1)

    Dmin = np.zeros(cycles + 1)

    # print(cycles)
    for j in range(cycles + 1):
        print('cycle ' + str(j))
        print('cycles ' + str(cycles))
        mu_dMin, sigma_dMs2 = mpcframe.gpr_predict(vlead_new[j] / 3.6 - v[j], s[j], gprM)
        dmin = norm.ppf(1 - beta, loc=mu_dMin, scale=np.sqrt(sigma_dMs2))
        d[j] = dmin[0][0]
        print(d[j])

        Dmin[j] = d[j] - v[j]*timestep - 0.5*a[j]*(timestep**2)

        # mpc_model = mpcframe.mpc(slead_new[j + timestep] - 400, d[j], v[j], s[j], s_test_new[j + timestep] - 400)
        mpc_model = mpcframe.modelopt(s[j] + d[j] - s0, v[j], s[j] - s0, s_test_new[j + timestep] - 400)
        # mpc_model = mpcframe.modelopt(s[j] + mu_dMin, v[j], s[j], s_test_new[j + timestep] - 400)

        if j == cycles:
            # mpc_model.options.CV_TYPE = 2
            mpc_model.fix_final(mpc_model.v, vlead_new[-1] / 3.6)
            # mpc_model.fix_final(mpc_model.s, s_test_new[-1] - s0)
            mpc_model.s.SP = slead_new[-1] - s0
            sp[j] = mpc_model.s.SP
            try:
                mpc_model.solve(disp=False)
            except:
                mpc_model.open_folder()
                mpc_model.options.SOLVER = 3
                mpc_model.solve(disp=False)

            u_cont = np.append(u_cont, mpc_model.u.VALUE[1:])
            v = np.append(v, mpc_model.v.VALUE[1:])
            a = np.append(a, mpc_model.a.VALUE[1:])
            # if mpc_model.s.value[0] < 20:
            s = np.append(s, mpc_model.s.PRED[1:] + s0)
            # else:
            #     s = np.append(s, mpc_model.s.value[1:])
            print('cycle ' + str(j))
            print('cycles ' + str(cycles))
            print(mpc_model.s.value[1:] + s0)

            break
        else:
            # mpc_model.s.SP = slead_new[j] + vlead_new[j] * timestep - 400 - d[j] / 5
            mpc_model.s.SP = slead_new[j + timestep] - s0 - d[j]
            sp[j] = mpc_model.s.SP
            # fig2
            # mpc_model.s.SP = slead_new[j] + vlead_new[j] * timestep + 0.5 * alead_new[j]*(timestep**2) - Dmin[j]  # fig

        try:
            mpc_model.solve(disp=False)
        except:
            mpc_model.open_folder()
            mpc_model.options.SOLVER = 3
            mpc_model.solve(disp=False)
            # print('Not successful')
            # from gekko.apm import get_file
            #
            # f = get_file(mpc_model.server, mpc_model.model_name, 'infeasibilities.txt')
            # f = f.decode().replace('\r', '')
            # with open('infeasibilities.txt', 'w') as fl:
            #     fl.write(str(f))

        with open(mpc_model.path + '//results.json') as f:
            results = json.load(f)

        u_cont[j] = mpc_model.u.NEWVAL
        # print(u_cont[j])

        a[j + 1], v[j + 1], s[j + 1] = mpcframe.process(v[j], s[j], u_cont[j])
        print(a[j + 1], v[j + 1], s[j + 1])

    Fuel_opt_test[i] = mpcframe.fuel(v, a, s) / (s[-1]-s0)

    uHost.append(list(u_cont))
    vHost.append(list(v))
    sHost.append(list(s))
    SP.append(list(sp + s0))


    sLead_new.append(list(slead_new))
    sTest_new.append(list(s_test_new))
    # sPreced_new.append(list(s_preced_new))

    plt.figure()
    plt.plot(SP[i])
    plt.plot(sLead_new[i], 'r--', label='Safety constraints ')
    plt.plot(s_test_new[:t_test + 1], 'k', label='Host Vehicle Original Trajectory')
    plt.plot(sHost[i], 'g', label='Host Vehicle Optimized Trajectory')
    plt.plot(610 * np.array(out['SPAT'][i]), 'r', label='Traffic light', linewidth=2)
    plt.xlabel('Time(secs)')
    plt.ylabel('Location(m)')
    plt.legend()
    plt.savefig('./fig10/vehicle_' + str(i) + 'png')
    print('vehicle' + str(i) + 'fuel_opt: %f' % (Fuel_opt_test[i]))
    print('vehicle' + str(i) + 'fuel_orig: %f' % (Fuel_orig_test[i]))


df = pd.DataFrame({'uHost': uHost, 'vHost': vHost, 'sHost': sHost,
                   'sLead_new': sLead_new, 'sTest_new': sTest_new })

df.to_csv('data_beta0.15.csv')
np.save('Results/Fuel_opt0.15.npy', Fuel_opt_test)
np.save('Results/Fuel_orig0.15.npy', Fuel_orig_test)



plt.figure()
vehicle_numbers = ["vehicle{}".format(index) for index in range(len(SPAT))]
Fuel_opt = list(Fuel_opt_test)
Fuel_orig = list(Fuel_orig_test)

bar_width = 0.3
index_idm = np.arange(len(vehicle_numbers))
index_mpc = index_idm + bar_width
plt.bar(index_idm, height=Fuel_orig, width=bar_width, color='b', label='Host Vehicle')
plt.bar(index_mpc, height=Fuel_opt, width=bar_width, color='g', label='Host Vehicle with GPR_MPC')
plt.ylabel('Fuel_consumption(mL)')
plt.legend()
plt.savefig('./fig10/fuel.png')
plt.show()


