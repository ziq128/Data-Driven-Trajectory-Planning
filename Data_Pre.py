import pandas as pd
import numpy as np
import random

# random.seed(12)
# random.seed(1)
random.seed(123)

class Data_pre:
    def __init__(self, data, data_init):
        self.data = data
        self.data_init = data_init

        self.b0 = 0.1569
        self.b1 = 2.4508 * 10e-2
        self.b2 = -7.415 * 10e-4
        self.b3 = 5.975 * 10e-5
        self.c0 = 0.07224
        self.c1 = 9.681 * 10e-2
        self.c2 = 1.075 * 10e-3
        self.ts = 0.1

    def get_training_df(self, ID_num_tr):
        df_training = pd.DataFrame()
        for i in range(len(ID_num_tr)):
            df_tr = self.data[(self.data['Host_ID'] == ID_num_tr[i])]
            df_train = df_tr[
                ['Host_vel', 'H_local_y', 'Preceding_vel', 'P_local_y_ts', 'Distance_Diff', 'Distance_Diff_Ts']]
            df_training = df_training.append(df_train)

        return df_training

    def get_test_df(self, ID_num_ts):
        df_test = pd.DataFrame()
        for i in range(len(ID_num_ts)):
            df_ts = self.data_init[(self.data_init['Vehicle_ID'] == ID_num_ts[i])]
            df_ts_2 = df_ts[(df_ts['Local_Y'] >= 600 - 211)]
            df_test = df_test.append(df_ts_2)

        return df_test

    def fv(self, v, a):
        if a >= 0:
            fuel = self.b0 + self.b1 * v + self.b2 * (v ** 2) + self.b3 * (v ** 3) + a * (
                    self.c0 + self.c1 * v + self.c2 * (v ** 2))
        else:
            fuel = self.b0 + self.b1 * v + self.b2 * (v ** 2) + self.b3 * (v ** 3)
        return fuel

    def fuel(self, v, a, s):
        n = len(s)
        fuel = np.zeros(n)
        for i in range(n - 1):
            if a[i] >= 0:
                fuel[i] = self.b0 + self.b1 * (v[i] * 3.6) + self.b2 * ((v[i] * 3.6) ** 2) + self.b3 * (
                            (v[i] * 3.6) ** 3) + a[i] * (
                                  self.c0 + self.c1 * (v[i] * 3.6) + self.c2 * ((v[i] * 3.6) ** 2))
            else:
                fuel[i] = self.b0 + self.b1 * (v[i] * 3.6) + self.b2 * ((v[i] * 3.6) ** 2) + self.b3 * (
                            (v[i] * 3.6) ** 3)

        fuel_ml = 0
        for t in range(len(v)):
            fuel_ml = fuel_ml + fuel[t] * self.ts

        return fuel_ml

    def get_init_fv(self, df=None):
        vel = df['v_Vel'].values
        s = df['Local_Y'].values
        v0_host = vel[0]
        s0_host = df['Local_Y'].values[0]
        acc = np.zeros(len(vel))
        fuel = np.zeros(len(vel))
        for i in range(len(vel) - 1):
            acc[i] = ((vel[i + 1] / 3.6) - (vel[i] / 3.6)) / self.ts
            fuel[i] = self.fv(vel[i], acc[i])

        fuel_ml = 0
        for t in range(len(acc)):
            fuel_ml = fuel_ml + fuel[t] * self.ts

        return vel, acc, s, fuel_ml, v0_host, s0_host


    def get_preceding_cons(self, df=None):
        sLead = list()
        vLead = list()
        preceding_ID = list()

        for row in df.itertuples():
            time = getattr(row, 'Global_Time')
            frame_ID = getattr(row, 'Frame_ID')
            if getattr(row, 'Preceding') == 0:
                pre_ID = preceding_ID[-1]
            else:
                pre_ID = getattr(row, 'Preceding')

            preceding_ID.append(pre_ID)
            preced_traj = self.data_init[(self.data_init['Vehicle_ID'] == pre_ID) & (self.data_init['Global_Time'] == time)
                                         & (self.data_init['Frame_ID'] == frame_ID) & (self.data_init['Local_Y'] >= 600 - 211)]
            if preced_traj.shape[0] == 1:
                slead = preced_traj['Local_Y'].values
                vlead = preced_traj['v_Vel'].values

                sLead.append(slead[0])
                vLead.append(vlead[0])
            else:
                sLead.append(np.nan)
                vLead.append(np.nan)

        return sLead, vLead

    @staticmethod
    def get_sPat(df=None):
        G_TR_NB = [630, 1629, 2618, 3628, 4585, 5625, 6582, 7613, 8612, 9578, 10610, 11507, 12618, 13606, 14614]
        R_TR_NB = [989, 1988, 2987, 3987, 4985, 5984, 6983, 7982, 8981, 9979, 10978, 11977, 12976, 13974, 14972]
        # find  SPAT information
        G_start = df.loc[df['Frame_ID'].isin(G_TR_NB[j] for j in range(0, len(G_TR_NB)))]
        R_start = df.loc[df['Frame_ID'].isin(R_TR_NB[j] for j in range(0, len(R_TR_NB)))]

        if R_start.shape[0] == 0 and G_start.shape[0] == 1:
            G_turn = G_start['Frame_ID'].values[0]
            cycle_st = int((G_turn - df['Frame_ID'].min()) / 10)
            index = G_TR_NB.index(G_turn)
            duration = int(np.rint((R_TR_NB[index] - G_turn) / 10))
            sPaT = list(np.ones(cycle_st)) + list(np.zeros(duration)) + list(np.ones(5))

        elif R_start.shape[0] == 1 and G_start.shape[0] == 1:
            G_turn = G_start['Frame_ID'].values[0]
            R_turn = R_start['Frame_ID'].values[0]
            if G_turn > R_turn:
                duration = int((G_turn - R_turn) / 10)
                cycle_st = int(np.rint((R_turn - df['Frame_ID'].min()) / 10))
                index = R_TR_NB.index(R_turn)
                nxt_cycle_st = int(np.rint((R_TR_NB[index + 1] - G_turn) / 10))
                sPaT = list(np.zeros(cycle_st)) + list(np.ones(duration)) + list(np.zeros(nxt_cycle_st)) + list(
                    np.ones(5))
            else:
                duration = int((R_turn - G_turn) / 10)
                cycle_st = int((G_turn - df['Frame_ID'].min()) / 10)
                index = G_TR_NB.index(G_turn)
                nxt_cycle_st = int(np.rint((G_TR_NB[index + 1] - R_turn) / 10))

                sPaT = list(np.ones(cycle_st)) + list(np.zeros(duration)) + list(np.ones(nxt_cycle_st)) + list(
                    np.zeros(5))

        elif R_start.shape[0] == 2 and G_start.shape[0] == 1:
            G_turn = G_start['Frame_ID'].values[0]
            R_turn = R_start['Frame_ID'].values[0]
            duration = int((G_turn - R_turn) / 10)
            cycle_st = int(np.rint((R_turn - df['Frame_ID'].min()) / 10))
            index = R_TR_NB.index(R_turn)
            nxt_cycle_st = int(np.rint((G_turn - R_TR_NB[index + 1]) / 10))
            nxt_R_st = R_start['Frame_ID'].values[1]
            second_R_duration = int(np.rint((df['Frame_ID'].max() - nxt_R_st) / 10))
            sPaT = list(np.zeros(cycle_st)) + list(np.ones(duration)) + list(np.zeros(nxt_cycle_st)) + list(
                np.ones(second_R_duration))

        elif R_start.shape[0] == 0 and G_start.shape[0] == 0:
            R_turn = min(R_TR_NB, key=lambda x: abs(x - df['Frame_ID'].max()))
            duration = int((R_turn - df['Frame_ID'].min()) / 10)
            sPaT = list(np.zeros(duration)) + list(np.ones(5))

        # elif R_start.shape[0] == 1 and G_start.shape[0] == 0:
        #     R_turn = R_start['Frame_ID'].values[0]
        #     index = R_TR_NB.index(R_turn)
        #     duration = int((df['Frame_ID'].max() - R_turn) / 10)
        #     cycle_st = int((R_turn - df['Frame_ID'].min()) / 10)
        #     sPaT = list(np.zeros(cycle_st)) + list(np.ones(duration)) + list()

        else:
            R_turn = R_start['Frame_ID'].values[0]
            cycle_st = int((R_turn - df['Frame_ID'].min()) / 10)
            index = R_TR_NB.index(R_turn)
            duration = int(np.rint((G_TR_NB[index + 1] - R_turn) / 10))
            nxt_cycle_st = int(np.rint((R_TR_NB[index + 1] - G_TR_NB[index + 1]) / 10))

            sPaT = list(np.zeros(cycle_st)) + list(np.ones(duration)) + list(np.zeros(nxt_cycle_st)) + list(np.ones(5))

        df_spat = pd.Series(sPaT)
        df_spat[df_spat == 0] = np.nan
        sPaT = df_spat.values

        return sPaT

    def main(self):
        vehicle_list = self.data['Host_ID'].unique()
        data_range = len(vehicle_list)
        data_training_range = np.floor(data_range * 0.7)

        ID_num_training = random.sample(list(vehicle_list), int(data_training_range))
        ID_num_test = list(set(vehicle_list) - set(ID_num_training))

        df_Training = self.get_training_df(ID_num_training)
        # Calculate the fuel consumption of the original data
        df_Test = self.get_test_df(ID_num_test)

        vehicle_test = df_Test['Vehicle_ID'].unique()

        veh_test, vel_test, acc_test, s_test, fuel_ml, SPAT, sLead, vLead, v0_host, s0_host= [[] for x in range(10)]

        for j in range(len(vehicle_test)):
            vehicle_test_value = df_Test[(df_Test['Vehicle_ID'] == vehicle_test[j])]
            lane_id = vehicle_test_value['Lane_ID'].unique()
            Total_frame_id = vehicle_test_value['Total_Frames'].unique()
            if len(lane_id) > 2 or len(Total_frame_id) > 1:
                continue


            Vel, Acc, S, Fuel_ml, V0_test, S0_test = self.get_init_fv(vehicle_test_value)
            pre_veh_list = vehicle_test_value['Preceding'].unique()

            if pre_veh_list[0] == 0:
                continue
            else:
                Slead, Vlead = self.get_preceding_cons(vehicle_test_value)

            if np.nan in Slead:
                num = 0
                for i in reversed(Slead):
                    num = num + 1
                    if np.isnan(i):
                        continue
                    else:
                        break

            Slead_new = Slead[:-num]
            Vlead_new = Vlead[:-num]

            sPat = self.get_sPat(vehicle_test_value)

            veh_test.append(vehicle_test_value)
            SPAT.append(sPat)
            fuel_ml.append(Fuel_ml)
            vel_test.append(Vel)
            s_test.append(S)
            sLead.append(list(Slead_new))
            vLead.append(list(Vlead_new))
            v0_host.append(V0_test)
            s0_host.append(S0_test)

        df_Training = self.get_training_df(ID_num_training)

        out = {}
        out['vel_test'] = vel_test
        out['acc_test'] = acc_test
        out['s_test'] = s_test
        out['fuel_ml'] = fuel_ml
        out['sLead'] = sLead
        out['vLead'] = vLead
        out['v0_host'] = v0_host
        out['s0_host'] = s0_host
        out['SPAT'] = SPAT

        return df_Training, SPAT, out















