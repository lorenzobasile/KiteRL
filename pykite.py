from ctypes import *
import numpy as np

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

def setup_lib(lib_path):
    lib = cdll.LoadLibrary(lib_path)
    lib.simulation_step.argtypes = [POINTER(kite), c_double, vect]
    lib.simulation_step.restype = c_bool
    return lib

libkite=setup_lib("libkite.so")
