from ctypes import *
import numpy as np

coefficients=np.array([
    [-0.15, 0.005],
    [-0.05, 0.005],
    [0.05, 0.001],
    [0.2, 0.005],
    [0.35, 0.01],
    [0.45, 0.02],
    [0.55, 0.65],
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
n_beta=1
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
    ]
    def __init__(self, initial_pos, initial_vel):
        self.position=initial_pos
        self.velocity=initial_vel
    def simulate(self, step, wind):
        return libkite.simulation_step(pointer(self), step, wind)
    def evolve_system(self, attack_angle, bank_angle, integration_steps, step, wind):
        C_l, C_d = coefficients[attack_angle,0], coefficients[attack_angle,1]
        psi = np.deg2rad(bank_angles[bank_angle])
        return libkite.simulate(pointer(self), C_l, C_d, psi, integration_steps, step, wind)
    def beta(self, wind):
        b=np.digitize(libkite.getbeta(pointer(self), wind), np.linspace(-np.pi/2, np.pi/2, n_beta))
        return 0
    def reward(self, attack_angle, bank_angle, wind):
        C_l, C_d = coefficients[attack_angle,0], coefficients[attack_angle,1]
        psi = np.deg2rad(bank_angles[bank_angle])
        return libkite.getreward(pointer(self), wind, C_l, C_d, psi)


def setup_lib(lib_path):
    lib = cdll.LoadLibrary(lib_path)
    lib.simulation_step.argtypes = [POINTER(kite), c_double, vect]
    lib.simulation_step.restype = c_bool
    lib.simulate.argtypes =[POINTER(kite), c_double, c_double, c_double, c_int, c_double, vect]
    lib.simulate.restype=c_bool
    lib.getbeta.argtypes = [POINTER(kite), vect]
    lib.getbeta.restype=c_double
    lib.getreward.argtypes = [POINTER(kite), vect, c_double, c_double, c_double]
    lib.getreward.restype=c_double
    return lib

libkite=setup_lib("libkite.so")
