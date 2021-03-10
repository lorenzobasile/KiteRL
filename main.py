from ctypes import *
import numpy as np

class vect(Structure):
    _fields_ = [
        ('theta', c_double),
        ('phi', c_double),
        ('r', c_double),
    ]

class kite(Structure):
    _fields_ = [
        ('position', vect),
        ('velocity', vect),
    ]
    def simulate(self, step, duration, wind):
        libkite.simulate(self, step, duration, wind)

def setup_lib(lib_path):
    lib = cdll.LoadLibrary(lib_path)

    lib.init_vect.argtypes = [c_double, c_double, c_double]
    lib.init_vect.restype = vect

    lib.init_kite.argtypes = [vect, vect]
    lib.init_kite.restype = kite

    lib.simulate.argtypes = [kite, c_double, c_int, vect]

    return lib

libkite=setup_lib("libkite.so")
newvect=libkite.init_vect
newkite=libkite.init_kite

initial_pos=newvect(np.pi/3, np.pi/24, 10)
initial_vel=newvect(0, 0, 0)
wind=newvect(10,0,0)
k=newkite(initial_pos, initial_vel)
k.simulate(0.001, 600000, wind)
