import numpy as np
import casadi as ca

import sys, os
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
sys.path += [BASEPATH]
from quadrotor import QuadrotorModel

# Quaternion Multiplication
def quat_mult(q1,q2):
    ans = ca.vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
           q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
           q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
           q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

# Quaternion-Vector Rotation
def rotate_quat(q1,v1):
    ans = quat_mult(quat_mult(q1, ca.vertcat(0, v1)), ca.vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return ca.vertcat(ans[1,:], ans[2,:], ans[3,:]) # to covert to 3x1 vec

class WayPointOpt():
    def __init__(self, quad:QuadrotorModel, wp_num:int, Ns:list, loop:bool, tol=0.01):
        self._loop = loop
        
        self._quad = quad
        self._ddynamics = self._quad.ddynamics_dt()

        self._tol = tol
        self._wp_num = wp_num
        assert(len(Ns)==wp_num)
        self._Ns = Ns
        self._Herizon = 0
        self._N_wp_base = [0]
        for i in range(self._wp_num):
            self._N_wp_base.append(self._N_wp_base[i]+self._Ns[i])
            self._Herizon += self._Ns[i]
        print("Total points: ", self._Herizon)

        self._X_dim = self._ddynamics.size1_in(0)
        self._U_dim = self._ddynamics.size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub

        self._DTs = ca.SX.sym('DTs', self._wp_num)
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._WPs_p = ca.SX.sym('WPs_p', 3, self._wp_num)
        if self._loop:
            self._X_init = self._Xs[:,-1]
        else:
            self._X_init = ca.SX.sym('X_init', self._X_dim)

        self._cost_Co = ca.diag([0.01,0.01,0.01]) # opt param
        self._cost_WP_p = ca.diag([1,1,1]) # opt param

        self._opt_option = {
            # 'verbose': False,
            'ipopt.tol': 1e-5,
            # 'ipopt.acceptable_tol': 1e-3,
            'ipopt.max_iter': 1000,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        self._opt_t_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0
        }
        self._opt_t_warm_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_frac': 1e-6,
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_frac': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.print_level': 0
        }

        #################################################################
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []

        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []

        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []

        self._nlp_g_orientation = []
        self._nlp_lbg_orientation = []
        self._nlp_ubg_orientation = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_wp_p = []
        self._nlp_lbg_wp_p = []
        self._nlp_ubg_wp_p = []

        self._nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []

        if self._loop:
            self._nlp_p_xinit = []
        else:
            self._nlp_p_xinit = [ self._X_init ]
            self._xinit = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
        self._nlp_p_dt = []
        self._nlp_p_wp_p = []
        
        self._nlp_obj_orientation = 0
        self._nlp_obj_minco = 0
        self._nlp_obj_time = 0
        self._nlp_obj_wp_p = 0
        self._nlp_obj_quat = 0
        self._nlp_obj_dyn = 0

        ###################################################################

        for i in range(self._wp_num):
            self._nlp_x_x += [ self._Xs[:, self._N_wp_base[i]] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, self._N_wp_base[i]] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            self._nlp_x_t += [ self._DTs[i] ]
            self._nlp_lbx_t += [0]
            self._nlp_ubx_t += [0.5]

            if i==0:
                dd_dyn = self._Xs[:,0]-self._ddynamics( self._X_init, self._Us[:,0], self._DTs[0])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_obj_minco += self._Xs[10:13,0].T@self._cost_Co@self._Xs[10:13,0]
            else:
                dd_dyn = self._Xs[:,self._N_wp_base[i]]-self._ddynamics( self._Xs[:,self._N_wp_base[i]-1], self._Us[:,self._N_wp_base[i]], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_obj_minco += self._Xs[10:13,self._N_wp_base[i]].T@self._cost_Co@self._Xs[10:13,self._N_wp_base[i]]
            
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]

            self._nlp_g_wp_p += [ (self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]).T@(self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]) ]
            self._nlp_lbg_wp_p += [0]
            self._nlp_ubg_wp_p += [ self._tol*self._tol ]

            self._nlp_p_dt += [ self._DTs[i] ]
            self._nlp_p_wp_p += [ self._WPs_p[:,i] ]

            # self._nlp_obj_minco += (self._Us[:,self._N_wp_base[i]]).T@self._cost_Co@(self._Us[:,self._N_wp_base[i]])
            self._nlp_obj_time += self._DTs[i]*self._Ns[i]
            self._nlp_obj_wp_p += (self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]).T@self._cost_WP_p@(self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i])
            
            for j in range(1, self._Ns[i]):
                self._nlp_x_x += [ self._Xs[:, self._N_wp_base[i]+j] ]
                self._nlp_lbx_x += self._X_lb
                self._nlp_ubx_x += self._X_ub
                self._nlp_x_u += [ self._Us[:, self._N_wp_base[i]+j] ]
                self._nlp_lbx_u += self._U_lb
                self._nlp_ubx_u += self._U_ub
                
                dd_dyn = self._Xs[:,self._N_wp_base[i]+j]-self._ddynamics( self._Xs[:,self._N_wp_base[i]+j-1], self._Us[:,self._N_wp_base[i]+j], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
                self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]

                # self._nlp_obj_minco += (self._Us[:,self._N_wp_base[i]+j]).T@self._cost_Co@(self._Us[:,self._N_wp_base[i]+j])
                self._nlp_obj_minco += self._Xs[10:13,self._N_wp_base[i]+j].T@self._cost_Co@self._Xs[10:13,self._N_wp_base[i]+j]

    # warm-up solver
    def define_opt(self):
        nlp_dect = {
            'f': 1*self._nlp_obj_dyn + self._nlp_obj_wp_p + 1*self._nlp_obj_minco,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_wp_p+self._nlp_p_dt)),
            # 'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        self._xu0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        for i in range(self._Herizon):
            self._xu0[i*self._X_dim+6] = 1
            self._xu0[self._Herizon*self._X_dim+i*self._U_dim] = -9.81

    def solve_opt(self, xinit, wp_p, dts):
        if self._loop:
            p = np.zeros(3*self._wp_num+self._wp_num)
            p[:3*self._wp_num] = wp_p
            p[3*self._wp_num:3*self._wp_num+self._wp_num] = dts
        else:
            self._xinit = xinit
            p = np.zeros(self._X_dim+3*self._wp_num+self._wp_num)
            p[:self._X_dim] = xinit
            p[self._X_dim:self._X_dim+3*self._wp_num] = wp_p
            p[self._X_dim+3*self._wp_num:] = dts
        res = self._opt_solver(
            x0=self._xu0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            # lbg=(),
            # ubg=(),
            p=p
        )
        self._xu0 = res['x'].full().flatten()
        self._dt0 = dts
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+self._wp_num)
        self._xut0[:(self._X_dim+self._U_dim)*self._Herizon] = self._xu0
        self._xut0[(self._X_dim+self._U_dim)*self._Herizon:] = self._dt0
        return res

    # time-optimal planner solver
    def define_opt_t(self):
        nlp_dect = {
            'f': self._nlp_obj_time,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_wp_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_wp_p)),
        }
        self._opt_t_solver = ca.nlpsol('opt_t', 'ipopt', nlp_dect, self._opt_t_option)
        self._opt_t_solver_warm = ca.nlpsol('opt_t', 'ipopt', nlp_dect, self._opt_t_warm_option)
        self._lam_x0 = np.zeros(self._opt_t_solver.size_in(6)[0])
        self._lam_g0 = np.zeros(self._opt_t_solver.size_in(7)[0])
        
    def solve_opt_t(self, xinit, wp_p, warm=False):
        if self._loop:
            p = np.zeros(3*self._wp_num)
            p = wp_p
        else:
            p = np.zeros(self._X_dim+3*self._wp_num)
            p[:self._X_dim] = xinit
            p[self._X_dim:self._X_dim+3*self._wp_num] = wp_p
        if warm:
            res = self._opt_t_solver_warm(
                x0=self._xut0,
                lam_x0 = self._lam_x0,
                lam_g0 = self._lam_g0,
                lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
                ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
                lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp_p),
                ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp_p),
                p=p
            )
        else:    
            res = self._opt_t_solver(
                x0=self._xut0,
                lam_x0 = self._lam_x0,
                lam_g0 = self._lam_g0,
                lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
                ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
                lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp_p),
                ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp_p),
                p=p
            )
        self._xut0 = res['x'].full().flatten()
        self._lam_x0 = res["lam_x"]
        self._lam_g0 = res["lam_g"]
        return res

import csv
def save_traj(res, opt: WayPointOpt, csv_f):
    with open(csv_f, 'w') as f:
        traj_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        labels = ['t',
                  "p_x", "p_y", "p_z",
                  "v_x", "v_y", "v_z",
                  "q_w", "q_x", "q_y", "q_z",
                  "w_x", "w_y", "w_z",
                  "u_1", "u_2", "u_3", "u_4"]
        traj_writer.writerow(labels)
        x = res['x'].full().flatten()
        
        t = 0
        s = x[(opt._Herizon-1)*opt._X_dim: opt._Herizon*opt._X_dim]
        u = x[opt._Herizon*opt._X_dim: opt._Herizon*opt._X_dim+opt._U_dim]
        traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])
        for i in range(opt._wp_num):
            dt = x[-opt._wp_num+i]
            for j in range(opt._Ns[i]):
                idx = opt._N_wp_base[i]+j
                t += dt
                s = x[idx*opt._X_dim: (idx+1)*opt._X_dim]
                if idx != opt._Herizon-1:
                    u = x[opt._Herizon*opt._X_dim+(idx+1)*opt._U_dim: opt._Herizon*opt._X_dim+(idx+2)*opt._U_dim]
                else:
                    u = [0,0,0,0]
                traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])

from gates.gates import Gates
def cal_Ns(gates:Gates, l_per_n:float, loop:bool, init_pos=[0,0,0]):
    if loop:
        init_pos = gates._pos[-1]
    
    Ns = []
    l = np.linalg.norm(np.array(gates._pos[0])-np.array(init_pos))
    Ns.append(int(l/l_per_n))

    for i in range(gates._N-1):
        l = np.linalg.norm(np.array(gates._pos[i])-np.array(gates._pos[i+1]))
        Ns.append(int(l/l_per_n))
    
    return Ns

def optimation(nx, quad):
    gate = Gates(BASEPATH+"gates/gates_"+nx+".yaml")

    dts = np.array([0.3]*gate._N)
    
    Ns = cal_Ns(gate, 0.3, loop=True)
    wp_opt = WayPointOpt(quad, gate._N, Ns, loop=True)
    wp_opt.define_opt()
    wp_opt.define_opt_t()
    
    print("\n\nWarm-up start ......\n")
    res = wp_opt.solve_opt([], np.array(gate._pos).flatten(), dts)
    print("\n\nTime optimization start ......\n")
    res_t = wp_opt.solve_opt_t([], np.array(gate._pos).flatten())
    save_traj(res, wp_opt, BASEPATH+"results/res_"+nx+".csv")
    save_traj(res_t, wp_opt, BASEPATH+"results/res_t_"+nx+".csv")

if __name__ == "__main__":    
    quad = QuadrotorModel(BASEPATH+'quad/quad_real.yaml')
    
    optimation("n7", quad)
