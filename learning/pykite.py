from ctypes import *
import numpy as np

coefficients=np.array([
    [-0.15, 0.005],
    [-0.05, 0.005],
    [0.05, 0.001],
    [0.2, 0.005],
    [0.35, 0.01],
    [0.45, 0.02],
    [0.55, 0.03],
    [0.65, 0.05],
    [0.75, 0.07],
    [0.82, 0.09],
    [0.9, 0.1],
    [1.0, 0.13],
    [1.08, 0.18],
    [1.1, 0.18],
    [1.05, 0.21]
])
bank_angles=np.array([-3, -2, -1, 0, 1, 2, 3])
n_beta=10

class vect(Structure):
    _fields_ = [
        ('theta', c_double),
        ('phi', c_double),
        ('r', c_double),
    ]
    def __init__(self, t, p, r):
        self.theta=t
        self.phi=p
        self.r=r

class kite(Structure):
    _fields_ = [
        ('position', vect),
        ('velocity', vect),
        ('wind', c_void_p),
        ('C_l', c_double),
        ('C_d', c_double),
        ('psi', c_double),
        ('time', c_double)
    ]
    def __init__(self, initial_pos, initial_vel, wind_type, params):
        self.position=initial_pos
        self.velocity=initial_vel
        self.C_l=0.35
        self.C_d=0.01
        self.psi=0
        self.time=0
        if wind_type=='turbo':
            libkite.init_turbo_wind(pointer(self))
        if wind_type=='lin':
            libkite.init_lin_wind(pointer(self), params['v_ground'], params['v_ang_coef'])
        if wind_type=='const':
            libkite.init_const_wind(pointer(self), params['v_wind_x'])
    def reset(self, initial_pos, initial_vel, wind_type, params):
        self.position=initial_pos
        self.velocity=initial_vel
        self.time=0
        if wind_type=='turbo':
            libkite.reset_turbo_wind(pointer(self))
        if wind_type=='lin':
            libkite.init_lin_wind(pointer(self), params['v_ground'], params['v_ang_coef'])
    def __str__(self):
        return "Position: "+str(self.position.theta)+","+str(self.position.phi)+","+str(self.position.r)+", Velocity"+ str(self.velocity.theta)+","+str(self.velocity.phi)+","+str(self.velocity.r)
    def simulate(self, step):
        return libkite.simulation_step(pointer(self), step)
    def evolve_system(self, attack_angle, bank_angle, integration_steps, step):
        self.update_coefficients(attack_angle, bank_angle)
        #C_l, self.C_d = coefficients[attack_angle,0], coefficients[attack_angle,1]
        #self.psi = np.deg2rad(bank_angles[bank_angle])
        return libkite.simulate(pointer(self), integration_steps, step)
    def beta(self):
        b=np.digitize(libkite.getbeta(pointer(self)), np.linspace(-np.pi/2, np.pi/2, n_beta))
        #b=libkite.getbeta(pointer(self))
        return b
    def accelerations(self):
        a=libkite.getaccelerations(pointer(self))
        return a.theta, a.phi, a.r
    def reward(self, learning_step):
        if self.position.r*np.cos(self.position.theta)<10:
            return libkite.getreward(pointer(self))*learning_step/2
        return libkite.getreward(pointer(self))*learning_step
    def update_coefficients(self, attack_angle, bank_angle):
        self.C_l, self.C_d = coefficients[attack_angle,0], coefficients[attack_angle,1]
        self.psi = np.deg2rad(bank_angles[bank_angle])


def setup_lib(lib_path):
    lib = cdll.LoadLibrary(lib_path)
    lib.simulation_step.argtypes = [POINTER(kite), c_double]
    lib.simulation_step.restype = c_int
    lib.simulate.argtypes =[POINTER(kite), c_int, c_double]
    lib.simulate.restype=c_int
    lib.init_const_wind.argtypes=[POINTER(kite), c_double]
    lib.init_lin_wind.argtypes=[POINTER(kite), c_double, c_double]
    lib.getbeta.argtypes = [POINTER(kite)]
    lib.getbeta.restype=c_double
    lib.getaccelerations.argtypes = [POINTER(kite)]
    lib.getaccelerations.restype=vect
    lib.getreward.argtypes = [POINTER(kite)]
    lib.getreward.restype=c_double
    return lib

libkite=setup_lib("./libkite.so")
