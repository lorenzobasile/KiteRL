import pykite as pk
import numpy as np

initial_pos=pk.vect(np.pi/3, np.pi/24, 10)
initial_vel=pk.vect(0, 0, 0)
wind=pk.vect(30,0,0)
k=pk.kite(initial_pos, initial_vel)

for i in range(600000):
    print(i, k.position.r)
    if not k.simulate(0.001, wind):
        break
