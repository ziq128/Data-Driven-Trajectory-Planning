import numpy as np
from gekko import GEKKO
import random
import GPy
from scipy.stats import norm
from more_itertools import chunked


class MPC_Frame:
    def __init__(self, event, timestep, df=None):
        self.event = event
        self.timestep = timestep
        self.df = df

        self.M = 1200
        self.Av = 2.5
        self.Cd = 0.32
        self.rho = 1.184
        self.mu = 0.013
        self.g = 9.81

        self.b0 = 0.1569
        self.b1 = 2.4508 * 10e-2
        self.b2 = -7.415 * 10e-4
        self.b3 = 5.975 * 10e-5
        self.c0 = 0.07224
        self.c1 = 9.681 * 10e-2
        self.c2 = 1.075 * 10e-3

        self.uMin = -2
        self.uMax = 3
        self.vMin = 0
        self.vMax = 20

        self.Ts = 1
        self.s0 = 400
        self.Ts_turn = 10
        self.NumberBranches = int(1000)
        self.NumberTrains = int(200)


    def fv(self, v):
        n = len(v)
        fuel = np.zeros(n-1)
        a = np.zeros(n-1)
        for i in range(n-1):
            a[i] = ((v[i+1]/3.6) - (v[i]/3.6))/self.Ts

            if a[i] >= 0:
                fuel[i] = self.b0 + self.b1 * v[i] + self.b2 * (v[i] ** 2) + self.b3 * (v[i] ** 3) + a[i] * (
                        self.c0 + self.c1 * v[i] + self.c2 * (v[i] ** 2))
            else:
                fuel[i] = self.b0 + self.b1 * v[i] + self.b2 * (v[i] ** 2) + self.b3 * (v[i] ** 3)

        fuel_ml = 0
        for t in range(n-1):
            fuel_ml = fuel_ml + fuel[t] * self.Ts

        return fuel_ml



    def fv2(self, v, u, s):
        n = len(s)
        a = np.zeros(n)
        fuel = np.zeros(n)
        for i in range(n - 1):
            a[i] = (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * ((v[i]*3.6) ** 2) - self.mu * self.g + u[i]
            if a[i] >= 0:
                fuel[i] = self.b0 + self.b1 * (v[i]*3.6) + self.b2 * ((v[i]*3.6) ** 2) + self.b3 * ((v[i]*3.6) ** 3) + a[i] * (
                        self.c0 + self.c1 * (v[i]*3.6) + self.c2 * ((v[i]*3.6) ** 2))
            else:
                fuel[i] = self.b0 + self.b1 * (v[i]*3.6) + self.b2 * ((v[i]*3.6) ** 2) + self.b3 * ((v[i]*3.6) ** 3)

        fuel_ml = 0
        for t in range(len(v)):
            fuel_ml = fuel_ml + fuel[t] * self.Ts

        return fuel_ml

    def fv3(self, v, u, s):
        n = len(s)
        a = np.zeros(n)
        fuel = np.zeros(n)
        for i in range(n-1):
            a[i] = (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * ((v[i]) ** 2) - self.mu * self.g + u[i]
            if a[i] >= 0:
                fuel[i] = self.b0 + self.b1 * v[i] + self.b2 * (v[i] ** 2) + self.b3 * (v[i] ** 3) + a[i] * (
                        self.c0 + self.c1 * v[i] + self.c2 * (v[i] ** 2))
            else:
                fuel[i] = self.b0 + self.b1 * v[i] + self.b2 * (v[i] ** 2) + self.b3 * (v[i] ** 3)

        fuel_ml = 0
        for t in range(len(v)):
            fuel_ml = fuel_ml + fuel[t] * self.Ts

        return fuel_ml

    def add_noise(self, slead_vel, slead):
        tf = len(slead)
        t = np.linspace(0, (tf-1)/10, num=tf)
        slead_vel_new = slead_vel - 3.2 * np.sin(0.3*t) + 0.2 * np.random.rand(tf)
        slead_new = np.zeros(tf)
        slead_new[0] = slead[0]
        for i in range(1, tf):
            slead_new[i] = slead_new[i-1] + self.Ts * slead_vel_new[i]

        return slead_new, slead_vel_new

    def distance_list_turn(self, distance_old):
        data_turn = distance_old - self.s0 * np.ones(len(distance_old))
        data_new = [sum(x) / len(x) for x in chunked(data_turn, self.Ts_turn)]
        return data_new

    def gpr_model(self, lenSca, variance):
        self.df['DIS'] = (600 - self.df[['H_local_y']]) / 1000
        Host_veh = self.df[['Host_vel']] / 3.6
        Preced_veh = self.df[['Preceding_vel']] / 3.6
        self.df['vel_diff'] = Preced_veh['Preceding_vel'] - Host_veh['Host_vel']

        # # safe distance standard
        # D_diff_tr = []
        # for row in self.df.itertuples():
        #     vel = getattr(row, 'Host_vel')
        #     rel_dis = getattr(row, 'Distance_Diff')
        #     if vel/3.6 < 2:
        #         D_min_standard = 10
        #     else:
        #         D_min_standard = (vel/3.6) / 2
        #
        #     D_diff_tr.append((rel_dis - D_min_standard) / D_min_standard)
        #
        self.df['D_diff_tr'] = self.df[['Distance_Diff']]
        # self.df['D_diff_tr'] = self.df[['Distance_Diff_Ts']]

        num_tr = random.sample(list(self.df.index), int(np.floor(self.df.shape[0] * 0.3)))

        df_tr = self.df.reindex(num_tr)

        X_train_0 = df_tr[['vel_diff', 'DIS']]
        Y_train_0 = df_tr[['D_diff_tr']]
        # Y_train_0 = df_tr[['Distance_Diff']]

        ker = GPy.kern.RBF(input_dim=2, lengthscale=lenSca, variance=variance, ARD=True) + GPy.kern.White(2)

        # create simple GP model
        m = GPy.models.GPRegression(X_train_0, Y_train_0, ker)
        m.optimize(messages=True, max_f_eval=1000)
        return m

    def gpr_model2(self, lenSca, variance):
        self.df['DIS'] = (600 - self.df[['H_local_y']]) / 1000
        Host_veh = self.df[['Host_vel']] / 3.6
        Preced_veh = self.df[['Preceding_vel']] / 3.6
        self.df['vel_diff'] = Preced_veh['Preceding_vel'] - Host_veh['Host_vel']

        # # safe distance standard
        # D_diff_tr = []
        # for row in self.df.itertuples():
        #     vel = getattr(row, 'Host_vel')
        #     rel_dis = getattr(row, 'Distance_Diff')
        #     if vel/3.6 < 2:
        #         D_min_standard = 10
        #     else:
        #         D_min_standard = (vel/3.6) / 2
        #
        #     D_diff_tr.append((rel_dis - D_min_standard) / D_min_standard)
        #
        self.df['D_diff_tr'] = self.df[['Distance_Diff_Ts']]
        # self.df['D_diff_tr'] = self.df[['Distance_Diff_Ts']]

        num_tr = random.sample(list(self.df.index), int(np.floor(self.df.shape[0] * 0.3)))

        df_tr = self.df.reindex(num_tr)

        X_train_0 = df_tr[['vel_diff', 'DIS']]
        Y_train_0 = df_tr[['D_diff_tr']]
        # Y_train_0 = df_tr[['Distance_Diff']]

        ker = GPy.kern.RBF(input_dim=2, lengthscale=lenSca, variance=variance, ARD=True) + GPy.kern.White(2)

        # create simple GP model
        m = GPy.models.GPRegression(X_train_0, Y_train_0, ker)
        m.optimize(messages=True, max_f_eval=1000)
        return m

    def gpr_predict(self, vi, yi, m):
        inputArr = np.expand_dims(np.array([vi, (600 - yi) / 1000]), 0)
        mu_dMin, sigma_dMs2 = m.predict(inputArr)
        return mu_dMin, sigma_dMs2

    def modelopt(self, sLead, v0, s0, goal):
        # MODEL
        p = GEKKO(remote=False)
        p.time = np.linspace(0, self.timestep, self.timestep+1)

        #p.dmin = p.Param(value=dMin, name='dmin')

        p.u = p.MV(lb=self.uMin, ub=self.uMax, name='Tractive Force')
        p.sLead = p.FV(value=sLead, name="preceding distance")

        p.v = p.CV(value=v0, lb=self.vMin, ub=self.vMax, name="velocity-meter per second")
        p.s = p.CV(value=s0, name="distance")

        p.a = p.Var(name="acceleration")

        p.V = p.Var(name="velocity-kilometer per hour")
        p.obj = p.Var(name="objective")

        p.Equation(p.s.dt() == p.v)
        p.Equation(p.v.dt() == p.a)
        p.Equation(p.a == (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * p.v ** 2 - self.mu * self.g + p.u)

        # p.Equation(p.final * (goal - p.s) <= 0)
        # p.Equation(p.s - p.sLead + p.dmin + p.v*0.5 <= 0)
        # p.Equation(p.s - p.sLead + p.v *0.5 <= 0)
        p.Equation(p.s - p.sLead <= 0)
        # p.Equation(p.final * p.v == 0)
        p.Equation(p.V == p.v * 3.6)
        p.Equation(p.obj == p.if3(p.a, (self.b0 + self.b1 * p.V + self.b2 * (p.V ** 2) + self.b3 * (p.V ** 3)) / p.v,
                                  (self.b0 + self.b1 * p.V + self.b2 * (p.V ** 2) + self.b3 * (p.V ** 3) +
                                   p.a * (self.c0 + self.c1 * p.V + self.c2 * (p.V ** 2))) / p.v))

        p.Minimize(p.obj)

        # p.options.MAX_ITER = 500
        # p.options.SOLVER = 3

        p.options.IMODE = 6
        p.options.CV_TYPE = 2

        p.u.STATUS = 1
        p.s.STATUS = 1

        p.u.FSTATUS = 0
        p.s.FSTATUS = 1
        # p.v.TAU = 5

        p.u.DCOST = 0.05  # Delta cost penalty for MV movement
        p.u.DMAX = 20

        p.s.TR_INIT = 2  # set point trajectory(reference trajectory how to change)
        # p.s.TAU = 3  # time constant of
        p.s.TAU = 5

        return p

    def mpc(self, slead, dmin, v0, s0, goal):
        c = GEKKO(name="control", remote=False)
        c.time = np.linspace(0, self.timestep, self.timestep+1)
        final = np.zeros(self.timestep+1)
        final[-1] = 1
        c.final = c.Param(value=final)

        c.u = c.MV(lb=self.uMin, ub=self.uMax)
        c.sLead = c.FV(value=slead)

        c.v = c.CV(value=v0)
        c.s = c.CV(value=s0)

        c.a = c.Var()

        c.V = c.Var()
        c.obj = c.Var()

        c.Equation(c.s.dt() == c.v)
        c.Equation(c.v.dt() == c.a)
        c.Equation(c.a == (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * c.v ** 2 - self.mu * self.g + c.u)
        # c.Equation(c.s - c.sLead + dmin + c.v*0.5 <= 0)
        c.Equation(c.s - c.sLead + dmin <= 0)
        c.Equation(c.final * (goal - c.s) <= 0)
        c.Equation(c.V == c.v * 3.6)
        c.Equation(c.obj == c.if3(c.a, (self.b0 + self.b1 * c.V + self.b2 * (c.V ** 2) + self.b3 * (c.V ** 3)) / c.v,
                                  (self.b0 + self.b1 * c.V + self.b2 * (c.V ** 2) + self.b3 * (c.V ** 3)
                                   + c.a * (self.c0 + self.c1 * c.V + self.c2 * (c.V ** 2))) / c.v))

        c.Minimize(c.obj)
        # options
        c.options.IMODE = 6
        c.options.CV_TYPE = 1
        # c.options.SOLVER = 3

        # STATUS = 0, optimizer doesn't adjust value
        # STATUS = 1, optimizer can adjust
        c.u.STATUS = 1
        c.s.STATUS = 1

        # FSTATUS = 0, no measurement
        # FSTATUS = 1, measurement used to update model
        c.u.FSTATUS = 0
        c.s.FSTATUS = 1

        c.u.DCOST = 20

        # if CV_TYPE = 2, use SP

        # TR_INIT is an option to specify how the initial conditions of
        # the controlled variable’s (CV) setpoint reference trajectory should change with each cycle.
        # Setpoint trajectory initialization (0=dead-band, 1=re-center with coldstart/out-of-service, 2=re-center always)
        c.s.TR_INIT = 2

        return c


    def process(self, v0, s0, u0):
        Ts = 1
        a = (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * (v0*3.6) ** 2 - self.mu * self.g + u0
        v = v0 + a*Ts
        if v <= 0:
            v_new = 0.100
        else:
            v_new = v
        s = s0 + v_new*Ts
        return a, v_new, s


    def fuel(self, v, a, s):
        n = len(v)
        fuel = np.zeros(n)
        for i in range(n - 1):
            if a[i] >= 0:
                fuel[i] = self.b0 + self.b1 * (v[i]*3.6) + self.b2 * ((v[i]*3.6) ** 2) + self.b3 * ((v[i]*3.6) ** 3) + a[i] * (
                        self.c0 + self.c1 * (v[i]*3.6) + self.c2 * ((v[i]*3.6) ** 2))
            else:
                fuel[i] = self.b0 + self.b1 * (v[i]*3.6) + self.b2 * ((v[i]*3.6) ** 2) + self.b3 * ((v[i]*3.6) ** 3)

        fuel_ml = 0
        for t in range(len(v)):
            fuel_ml = fuel_ml + fuel[t] * self.Ts

        return fuel_ml


    '''
     # def mhe(self, slead, dmin):
        m = GEKKO(name="estimated")
        m.time = np.linspace(0, 1, 11)

        m.u = m.MV()
        m.a = m.Var(name="acceleration")
        m.v = m.Var(lb=self.vMin, ub=self.vMax, name="velocity")
        m.s = m.CV(name="distance")
        m.sLead = m.FV(value=slead, lb=400, ub=630)
        # m.dmin = m.FV(value=dmin)
        m.V = m.Var()
        m.obj = m.Var(name="objective")

        m.Equation(m.s.dt() == m.v)
        m.Equation(m.v.dt() == m.a)
        m.Equation(m.a == (-1 / (2 * self.M)) * self.Cd * self.rho * self.Av * m.v ** 2 - self.mu * self.g + m.u)
        m.Equation(m.s - m.sLead + dmin + m.v * 0.5 <= 0)

        m.Equation(m.V == m.v * 3.6)
        m.Equation(
            m.obj == m.if3(m.a, (self.b0 + self.b1 * m.V + self.b2 * (m.V ** 2) + self.b3 * (m.V ** 3)) / m.v,
                           (self.b0 + self.b1 * m.V + self.b2 * (m.V ** 2) + self.b3 * (m.V ** 3) +
                            m.a * (self.c0 + self.c1 * m.V + self.c2 * (m.V ** 2))) / m.v))

        m.Minimize(m.obj)
        m.options.IMODE = 5
        m.options.EV_TYPE = 1
        m.options.DIAGLEVEL = 0

        # STATUS = 0, optimizer doesn't adjust value
        # STATUS = 1, optimizer can adjust
        m.u.STATUS = 0
        m.sLead.STATUS = 1
        # m.dmin.STATUS = 1
        m.s.STATUS = 1

        # FSTATUS = 0, no measurement
        # FSTATUS = 1, measurement used to update model
        m.u.FSTATUS = 1
        m.sLead.FSTATUS = 0
        # m.dmin.FSTATUS = 0
        m.s.FSTATUS = 1

        m.sLead.DMAX = 5
        # m.dmin.DMAX = 20

        m.s.MEAS_GAP = 5
        # Setpoint  trajectory  initialization(0 = dead-band, 1 = re-centerwith coldstart / out-of-service, 2=re-center always)
        m.s.TR_INIT = 1

        return m
        
        
    def GaussianMethod(self, sLead, vLead, dmin, v0, s0, goal, t_test, alpha):
        BranchPos = np.zeros([t_test, self.NumberBranches])
        BranchSpeed = np.zeros([t_test, self.NumberBranches])
        BranchControl = np.zeros([t_test-1, self.NumberBranches])

        Sigma = np.zeros([t_test, self.NumberBranches])
        n = 0

        Mean = np.zeros([t_test, 1])
        Mean1 = np.zeros([t_test, 1])
        Mean2 = np.zeros([t_test, 1])
        vTarget = np.zeros([t_test, 1])

        for i in range(self.NumberBranches):
            pos = s0
            speed = v0
            posPrev = pos
            speedPrev = speed

            BranchPos[0, i] = s0
            BranchSpeed[0, i] = v0

            for k in range(t_test):
                vTarget[k] = (goal-pos)/(t_test)

                if goal < pos:
                    vTarget[k] = 0

                Mean1[k, i] = ((vTarget[k] - speed)/self.Ts) + (1 / (2 * self.M)) * self.Cd * self.rho * self.Av * (speed ** 2) + self.mu * self.g
                if Mean1[k, i] > 0:
                    Mean1[k, i] = np.min([Mean1[k, i], self.uMax])
                else:
                    Mean1[k, i] = np.min([Mean1[k, i], self.uMin])

                Mean2[k, i] = ((vLead[k] - speed)/self.Ts) + (1 / (2 * self.M)) * self.Cd * self.rho * self.Av * (speed ** 2) + self.mu * self.g
                if Mean2[k, i] > 0:
                    Mean2[k, i] = np.min([Mean2[k, i], self.uMax])
                else:
                    Mean2[k, i] = np.max([Mean2[k, i], self.uMin])

                if sLead[k] - pos >= dmin + alpha * speed:
                    Mean[k, i] = Mean1[k, i]
                else:
                    Mean[k, i] = Mean2[k, i]

                d = sLead[k] - pos
                dConst = dmin

                beta1 = 0.6
                beta2 = 1.4
                Sigma[k, i] = beta1 * np.abs(dConst / (d - beta2 * dConst))

                for j in range(self.NumberTrains):
                    control = Mean[k, i] + Sigma[k, i] * np.random.randn()

                    pos = posPrev + self.Ts*speedPrev
                    a = control - (1 / (2 * self.M)) * self.Cd * self.rho * self.Av * (speed ** 2) - self.mu * self.g
                    speed = speedPrev + self.Ts*a

                    if (sLead[k] - pos >= dmin + alpha * speed) and (self.vMin <= speed <= self.vMax) and (self.uMin <= control<= self.uMax):
                        BranchPos[k+1, i] = pos
                        BranchSpeed[k+1, i] = speed
                        BranchControl[k, i] = control
                        posPrev = pos
                        speedPrev = speed

                        break

            if BranchPos[t_test, i] >= goal:
                n = n + 1
                BranchSuccessPos = np.append(BranchSuccessPos, BranchPos[:, i])
                BranchSuccessSpeed = np.append(BranchSuccessSpeed, BranchSpeed[:, i])
                BranchSuccessControl = np.append(BranchSuccessControl, BranchControl[:, i])

        fuel = np.zeros(n)
        for i in range(n):
            fuel[i] = self.fv2(BranchSuccessSpeed[i], BranchSuccessControl[i],BranchSuccessPos[i])

        MinFuel = np.min(fuel)
        index = np.where(fuel == np.min(fuel))

        Space = BranchSuccessPos[index]
        Velocity = BranchSuccessSpeed[index]
        Control = BranchSuccessControl[index]
        SuccessfulBranches = 100*(n/self.NumberBranches)

        return Space, Velocity, Control, MinFuel, SuccessfulBranches
 '''












