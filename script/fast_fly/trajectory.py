import numpy as np
import casadi as ca
import csv

class Near():
    def __init__(self, path=None):
        
        opt_option = {
                'verbose': False,
                'ipopt.tol': 1e-2,
                'ipopt.acceptable_tol': 1e-2,
                'ipopt.max_iter': 20,
                # 'ipopt.warm_start_init_point': 'yes',
                'ipopt.print_level': 0,
            }

        if path==None:
            t = ca.SX.sym('t')
            traj_ploy = ca.SX.sym("traj_ploy", 12, 2)
            dts = ca.SX.sym("dts",2)
            pos = ca.SX.sym('pos', 3)

            obj = (self.traj_ploy(traj_ploy, dts, t)-pos).T@(self.traj_ploy(traj_ploy, dts, t)-pos)

            nlp_dect = {
                'f': obj,
                'x': t,
                'p': ca.vertcat(traj_ploy[:,0], traj_ploy[:,1], dts, pos)
            }
            self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, opt_option)
        else:
            self._opt_solver = ca.nlpsol('opt', 'ipopt', path, opt_option)

    def solve(self, coef1, coef2, dt1, dt2, pos):
        p = np.zeros(12*2+2+3)
        p[:12] = coef1
        p[12:24] = coef2
        p[24] = dt1
        p[25] = dt2
        p[26:] = pos

        res = self._opt_solver(
            x0=0,
            p=p
        )

        return res['x'].full().flatten()


    def traj_ploy(self, ploy_coef, dts, t):
        p=0

        a0=ploy_coef[:3,0]
        a1=ploy_coef[3:6,0]
        a2=ploy_coef[6:9,0]
        a3=ploy_coef[9:12,0]
        dt = dts[0]+t
        dtdt = dt*dt
        dtdtdt = dtdt*dt
        p+= (t<=0)*( a0+a1*dt+a2*dtdt+a3*dtdtdt )
        
        a0=ploy_coef[:3,1]
        a1=ploy_coef[3:6,1]
        a2=ploy_coef[6:9,1]
        a3=ploy_coef[9:12,1]
        dt = t
        dtdt = dt*dt
        dtdtdt = dtdt*dt
        p+= (t>0)*( a0+a1*dt+a2*dtdt+a3*dtdtdt )

        return p

class Trajectory():
    def __init__(self, csv_f=None, near_path=None):
        if csv_f != None:
            t = []
            pos = []
            vel = []
            quaternion = []
            omega = []
            self._N = 0
            ploynomials = []
            u = []
            acc = []
            with open(csv_f, 'r') as f:
                traj_reader = csv.DictReader(f)
                for s in traj_reader:
                    t.append(float(s['t']))
                    pos.append([ float(s["p_x"]), float(s["p_y"]), float(s["p_z"]) ])
                    vel.append([ float(s["v_x"]), float(s["v_y"]), float(s["v_z"]), np.sqrt(float(s["v_x"])*float(s["v_x"])+float(s["v_y"])*float(s["v_y"])+float(s["v_z"])*float(s["v_z"])) ])
                    quaternion.append([ float(s["q_w"]), float(s["q_x"]), float(s["q_y"]), float(s["q_z"]) ])
                    omega.append([ float(s["w_x"]), float(s["w_y"]), float(s["w_z"]) ])
                    if "u_1" in s.keys():
                        u.append([float(s["u_1"]), float(s["u_2"]), float(s["u_3"]), float(s["u_4"])])
                    if "a_x" in s.keys():
                        acc.append([float(s["a_x"]), float(s["a_y"]), float(s["a_z"])])
            
            self._t = np.array(t)
            self._pos = np.array(pos)
            self._vel = np.array(vel)
            self._quaternion = np.array(quaternion)
            self._omega = np.array(omega)
            self._u = np.array(u)
            self._acc = np.array(acc)
                
            self._N = self._t.shape[0]-1
            assert(self._N>0)
            self._dt = self._t[1:]-self._t[:-1]
            for i in range(self._N):
                a0, a1, a2, a3 = self._ploynomial(self._pos[i], self._pos[i+1], self._vel[i,:3], self._vel[i+1,:3], self._dt[i])
                ploynomials.append( np.concatenate((a0,a1,a2,a3)) )
            self._ploynomials = np.array(ploynomials)

        else:
            self._t = np.array([])
            self._pos = np.array([])
            self._vel = np.array([])
            self._quaternion = np.array([])
            self._omega = np.array([])
            self._u = np.array([])
            self._acc = np.array([])
            self._N = 0
            self._dt = np.array([])
            self._ploynomials = np.array([])
        
        self._near = Near(near_path)
        
    def load_data(self, pos, vel, quat, omega, dt):
        self._pos = pos
        self._quaternion = quat
        self._omega = omega
        self._dt = dt
        self._N = self._pos.shape[0]-1

        self._vel = np.zeros([self._N+1, 4])
        self._vel[0,:3] = vel[0,:]
        self._vel[0,3] = np.linalg.norm(vel[0,:])
        self._t = np.zeros([self._N+1])
        self._t[0] = 0
        ploynomials = []
        for i in range(self._N):
            self._vel[i+1,:3] = vel[i+1,:]
            self._vel[i+1,3] = np.linalg.norm(vel[i+1,:])
            self._t[i+1] = self._t[i]+dt[i]
            
            a0, a1, a2, a3 = self._ploynomial(self._pos[i], self._pos[i+1], self._vel[i,:3], self._vel[i+1,:3], self._dt[i])
            ploynomials.append( np.concatenate((a0,a1,a2,a3)) )
        self._ploynomials = np.array(ploynomials)
        

    def __getitem__(self, idx):
        traj = Trajectory()
        traj._t = self._t[idx]
        traj._pos = self._pos[idx]
        traj._vel = self._vel[idx]
        traj._quaternion = self._quaternion[idx]
        traj._omega = self._omega[idx]
        traj._dt = traj._t[1:] - traj._t[:-1]
        traj._ploynomials = self._ploynomials[idx]
        if self._u.shape[0]!=0:
            traj._u = self._u[idx]
        if self._acc.shape[0]!=0:
            traj._acc = self._acc[idx]
        traj._N = traj._t.shape[0]-1
        return traj

    def _ploynomial(self, p1, p2, v1, v2, dt):
        a0 = p1
        a1 = v1
        a2 = (3*p2-3*p1-2*v1*dt-v2*dt)/dt/dt
        a3 = (2*p1-2*p2+v2*dt+v1*dt)/dt/dt/dt
        return a0, a1, a2, a3
    
    def _seg_pos_vel(self, poly_coef, dt):
        a0 = poly_coef[:3]
        a1 = poly_coef[3:6]
        a2 = poly_coef[6:9]
        a3 = poly_coef[9:12]

        dtdt = dt*dt
        dtdtdt = dtdt*dt
        pos = a0 + a1*dt + a2*dtdt +a3*dtdtdt
        vel = a1 + 2*a2*dt + 3*a3*dtdt
        return pos, vel

    def sample(self, n, pos):
        idx = np.argmin(np.linalg.norm(self._pos-pos, axis=1))
        idx -= 1 #

        traj_seg = np.zeros((n, 3))
        traj_v_seg = np.zeros((n, 3))
        traj_dt_seg = np.zeros(n)
        traj_polynomial_seg = np.zeros((n, 12))
        for i in range(n):
            traj_seg[i,:] = self._pos[(idx+int(i*1.0))%self._N]
            traj_v_seg[i,:] = self._vel[(idx+int(i*1.0))%self._N,:3]
            traj_dt_seg[i] = self._dt[(idx+int(i*1.0))%self._N]
            traj_polynomial_seg[i, :] = self._ploynomials[(idx+int(i*1.0))%self._N]

        return traj_seg, traj_v_seg, traj_dt_seg, traj_polynomial_seg
    
    def sample_dt_reset(self):
        self._sample_time_idx = 0
        self._sample_time = 0
    
    def sample_t(self, time_dt, dt, n, loop=True):
        self._sample_time += time_dt
        
        while self._sample_time - self._dt[self._sample_time_idx]>0:
            self._sample_time -= self._dt[self._sample_time_idx]
            self._sample_time_idx += 1
            if self._sample_time_idx >= self._N:
                if loop:
                    self._sample_time_idx = 0
                else:
                    self._sample_time_idx = self._N-1
                    break
        
        traj_seg = np.zeros((n, 3))
        traj_v_seg = np.zeros((n, 3))
        for i in range(n):
            s_time = self._sample_time + (i+1)*dt
            s_idx = self._sample_time_idx
            while s_time - self._dt[s_idx]>0:
                s_time -= self._dt[s_idx]
                s_idx += 1
                if s_idx >= self._N:
                    if loop:
                        s_idx = 0
                    else:
                        s_idx = self._N-1
                        break
            
            traj_seg[i,:], traj_v_seg[i,:] = self._seg_pos_vel(self._ploynomials[s_idx], s_time)
            
        return traj_seg, traj_v_seg
    
    def distance(self, pos):
        idx = np.argmin(np.linalg.norm(self._pos-pos, axis=1))
        if idx == 0:
            idx = 1
        ddt = self._near.solve(self._ploynomials[idx-1], self._ploynomials[idx], self._dt[idx-1], self._dt[idx], pos)
        idx_dt = 0
        if ddt<0:
            idx_dt = self._dt[idx-1]+ddt
            idx = idx-1
        else:
            idx_dt = ddt
        
        p, v = self._seg_pos_vel(self._ploynomials[idx], idx_dt)
        return np.linalg.norm(p-pos)
    
    def divide_loops(self, pos):
        loop_idx = []
        flag1 = 0
        flag2 = 0
        for i in range(self._N):
            if np.linalg.norm(self._pos[i]-pos)< 1:
                if flag1 == 0:
                    l = np.linalg.norm(self._pos[i] - pos)
                    flag1 = 1
                else:
                    if l<np.linalg.norm(self._pos[i] - pos):
                        if flag2 == 0:
                            loop_idx.append(i)
                            flag2 = 1
                l = np.linalg.norm(self._pos[i] - pos)
            else:
                flag1 = 0
                flag2 = 0

        return loop_idx

class TrajLog():
    def __init__(self, path):
        self._fd = open(path, 'w')
        self._traj_writer = csv.writer(self._fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        labels = ['t',
                  "p_x", "p_y", "p_z",
                  "v_x", "v_y", "v_z",
                  "q_w", "q_x", "q_y", "q_z",
                  "w_x", "w_y", "w_z",
                  "u_1", "u_2", "u_3", "u_4"]
        self._traj_writer.writerow(labels)
    def log(self, t, s, u):
        self._traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])

    def __del__(self):
        self._fd.close()

class StateSave():
    def __init__(self, path):
        self._fd = open(path, 'w')
        self._state_writer = csv.writer(self._fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        labels = ['t',
                  "p_x", "p_y", "p_z",
                  "v_x", "v_y", "v_z",
                  "q_w", "q_x", "q_y", "q_z",
                  "w_x", "w_y", "w_z",
                  "u_1", "u_2", "u_3", "u_4",
                  "a_x", "a_y", "a_z"]
        self._state_writer.writerow(labels)
    def log(self, t, s, u, a):
        self._state_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3], a[0], a[1], a[2]])

    def __del__(self):
        self._fd.close()

class StateLoader():
    def __init__(self, csv_f):
        if csv_f != None:
            t = []
            pos = []
            vel = []
            quaternion = []
            omega = []
            u = []
            acc = []
            self._N = 0
            with open(csv_f, 'r') as f:
                traj_reader = csv.DictReader(f)
                for s in traj_reader:
                    t.append(float(s['t']))
                    pos.append([ float(s["p_x"]), float(s["p_y"]), float(s["p_z"]) ])
                    vel.append([ float(s["v_x"]), float(s["v_y"]), float(s["v_z"])])
                    quaternion.append([ float(s["q_w"]), float(s["q_x"]), float(s["q_y"]), float(s["q_z"]) ])
                    omega.append([ float(s["w_x"]), float(s["w_y"]), float(s["w_z"]) ])
                    u.append([ float(s["u_1"]), float(s["u_2"]), float(s["u_3"]), float(s["u_4"]) ])
                    acc.append([ float(s["a_x"]), float(s["a_y"]), float(s["a_z"]) ])
            
            self._t = np.array(t)
            self._pos = np.array(pos)
            self._vel = np.array(vel)
            self._quaternion = np.array(quaternion)
            self._omega = np.array(omega)
            self._u = np.array(u)
            self._acc = np.array(acc)
            self._N = self._t.shape[0]
