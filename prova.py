import pykite as pk
initial_position=pk.vect(3.1415/6,0,50)
initial_velocity=pk.vect(0,0,0)
k=pk.kite(initial_position, initial_velocity)
print(k.C_l)
for i in range(600000):
    k.simulate(0.0001)
    k.accelerations()
print(k.position.r)
