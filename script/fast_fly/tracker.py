import casadi as ca
import numpy as np

import os, sys
BASEPATH = os.path.abspath(__file__).split('fast_fly/', 1)[0]+'fast_fly/'
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel, rotate_quat


def linear(n, l, p:ca.SX, ls):
    y = 0
    for i in range(n-1):
        y += ca.logic_and(ls[i]<=l, l<ls[i+1])*( p[:,i]+(l-ls[i])/(ls[i+1]-ls[i])*(p[:,i+1]-p[:,i]) )
    # y += (ls[n-1]<=l)*p[:,n-1]
    y += (ls[n-1]<=l)*( p[:,n-2]+(l-ls[n-2])/(ls[n-1]-ls[n-2])*(p[:,n-1]-p[:,n-2]) )
    return y

def linear2(n, l, p:ca.SX, ls):
    y = 0
    v = 0
    for i in range(n-1):
        y += ca.logic_and(ls[i]<=l, l<ls[i+1])*( p[:,i]+(l-ls[i])/(ls[i+1]-ls[i])*(p[:,i+1]-p[:,i]) )
        v += ca.logic_and(ls[i]<=l, l<ls[i+1])*( (p[:,i+1]-p[:,i])/(ls[i+1]-ls[i]) )
    # y += (ls[n-1]<=l)*p[:,n-1]
    y += (ls[n-1]<=l)*( p[:,n-2]+(l-ls[n-2])/(ls[n-1]-ls[n-2])*(p[:,n-1]-p[:,n-2]) )
    v += (ls[n-1]<=l)*( (p[:,n-1]-p[:,n-2])/(ls[n-1]-ls[n-2]) )
    return y, v

def p_cost(v, Th):
    c = v.T@v
    return c

def traj_ploy(n, ploy_coef, ts, t):
    p=0
    v=0
    for i in range(n-1):
        a0=ploy_coef[:3,i]
        a1=ploy_coef[3:6,i]
        a2=ploy_coef[6:9,i]
        a3=ploy_coef[9:12,i]
        dt=t-ts[i]
        dtdt = dt*dt
        dtdtdt = dtdt*dt
        p += ca.logic_and(ts[i]<=t, t<ts[i+1])*( a0+a1*dt+a2*dtdt+a3*dtdtdt )
        v += ca.logic_and(ts[i]<=t, t<ts[i+1])*( a1+2*a2*dt+3*a3*dtdt )
    
    dt=t-ts[n-1]
    dtdt = dt*dt
    dtdtdt = dtdt*dt
    p += (ts[n-1]<=t)*( a0+a1*dt+a2*dtdt+a3*dtdtdt )
    v += (ts[n-1]<=t)*( a1+2*a2*dt+3*a3*dtdt )
    return p, v

class TrackerMPC():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 5
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.1+n*0.0)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._Herizon)

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 25,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_Trj_v = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjv = 0
        self._nlp_obj_u = 0
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-self._Trj_p[:,0], 0.5)

        # self._nlp_obj_u += self._Us[:,0].T@self._Us[:,0]
        self._nlp_obj_u += self._Xs[10:13,0].T@self._Xs[10:13,0]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-self._Trj_p[:,i], 0.5)

            # self._nlp_obj_u += self._Us[:,i].T@self._Us[:,i]
            self._nlp_obj_u += self._Xs[10:13,i].T@self._Xs[10:13,i]

        self._nlp_p_xinit += [self._X_init]

        for i in range(self._Herizon):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
    
    def reset_xut(self):
        self._xu0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        for i in range(self._Herizon):
            self._xu0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xut()
    
    def define_opt(self):
        nlp_dect = {
            'f': 1*self._nlp_obj_trjp+0.002*self._nlp_obj_u,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xut()
        
    
    def solve(self, xinit, Trjp):
        p = np.zeros(self._X_dim+3*self._Herizon)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._Herizon] = Trjp

        res = self._opt_solver(
            x0=self._xu0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            lbg=(self._nlp_lbg_dyn),
            ubg=(self._nlp_ubg_dyn),
            p=p
        )
        
        self._xu0 = res['x'].full().flatten()
        
        return res

class TrackerPos():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 5
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.1+n*0.0)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._t0 = ca.SX.sym('t0', 1)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 15
        self._Trj_ploy = ca.SX.sym("Trj_ploy", 12, self._trj_N)
        self._Trj_dt = ca.SX.sym("Trj_dt", 1, self._trj_N)
        self._Trj_t_ts =  [0]
        for i in range(self._trj_N-1):
            self._Trj_t_ts.append(self._Trj_t_ts[i] + self._Trj_dt[i])
        self._time_factor = ca.SX.sym("time_factor", 1)

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 20,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        
        self._nlp_x_t0 = []
        self._nlp_lbx_t0 = []
        self._nlp_ubx_t0 = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_ploy = []
        self._nlp_p_Trj_dt = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_u = 0
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        trjp, trjv = traj_ploy(self._trj_N, self._Trj_ploy, self._Trj_t_ts, self._t0)
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        # self._nlp_obj_trjv += (self._Xs[3:6,0]-trjv).T@(self._Xs[3:6,0]-trjv)
        # vv = (self._Xs[3:6,0].T@trjv/(trjv.T@trjv))*trjv
        # self._nlp_obj_trjv += (vv-trjv).T@(vv-trjv)
        self._nlp_obj_u += self._Xs[10:13,0].T@self._Xs[10:13,0]
        
        self._nlp_x_t0 += [self._t0]
        self._nlp_lbx_t0 += [0]
        self._nlp_ubx_t0 += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            trjp, trjv = traj_ploy(self._trj_N, self._Trj_ploy, self._Trj_t_ts, self._t0+self._time_factor*0.1*i)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            self._nlp_obj_u += self._Xs[10:13,i].T@self._Xs[10:13,i]

        self._nlp_p_xinit += [self._X_init]
        
        for i in range(self._trj_N):
            self._nlp_p_Trj_ploy += [self._Trj_ploy[:,i]]
            self._nlp_p_Trj_dt += [self._Trj_dt[:,i]]
    
    def reset_xut(self):
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+1)
        for i in range(self._Herizon):
            self._xut0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xut()
    
    def define_opt(self):
        nlp_dect = {
            'f': 1*self._nlp_obj_trjp+0.002*self._nlp_obj_u,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t0)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_ploy+self._nlp_p_Trj_dt), self._time_factor),
            'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xut()
        
    
    def solve(self, xinit, Trj, dt, time_factor=1):
        p = np.zeros(self._X_dim+12*self._trj_N+self._trj_N+1)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+12*self._trj_N] = Trj
        p[self._X_dim+12*self._trj_N:self._X_dim+13*self._trj_N] = dt
        p[-1] = time_factor
        res = self._opt_solver(
            x0=self._xut0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t0),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t0),
            lbg=(self._nlp_lbg_dyn),
            ubg=(self._nlp_ubg_dyn),
            p=p
        )
        
        self._xut0 = res['x'].full().flatten()
        
        return res

if __name__ == "__main__":
    
    quad = QuadrotorModel(BASEPATH+"quad/quad_real.yaml")
    tracker = TrackerPos(quad)
    tracker.define_opt()
    tracker._opt_solver.generate_dependencies("tracker_pos.c")    
