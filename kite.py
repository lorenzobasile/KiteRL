import numpy as np

m=1.0
dl=0
rhol=970
g=9.8

d=1.0
delta_l=0.2
A=0.5
rho=1.2
C_d=0.29
C_l=1.5

M=5.0
a=0.2



class kite():
    def __init__(self, pos0, vel0):
        self.position=pos0
        self.velocity=vel0
    def __str__(self):
        return "position: "+np.array_str(self.position)+", velocity: "+np.array_str(self.velocity)
    def transition_matrix(self):
        m=np.zeros((3,3))
        m[0,0]=np.cos(self.position[0])*np.cos(self.position[1])
        m[0,1]=-np.sin(self.position[1])
        m[0,2]=np.sin(self.position[0])*np.cos(self.position[1])
        m[1,0]=np.cos(self.position[0])*np.sin(self.position[1])
        m[1,1]=np.cos(self.position[1])
        m[1,2]=np.sin(self.position[0])*np.sin(self.position[1])
        m[2,0]=-np.sin(self.position[0])
        m[2,2]=np.cos(self.position[0])
        return m
    def update_state(self, step, wind):
        f=self.compute_force(wind);
        t=self.tension(f);
        f-=t;
        self.velocity[0]+=(f[0]/(m*self.position[2])*step)
        self.velocity[1]+=(f[1]/(m*self.position[2]*np.sin(self.position[0]))*step)
        self.velocity[2]+=(f[2]/m*step)
        self.position+=self.velocity*step
        print("Power: ", self.velocity[2]*t[2])
        if self.position[0]>=np.pi/2:
            return False
        return True
    def compute_force(self, wind):
        f_grav=np.zeros(3)
        f_app=np.zeros(3)
        f_grav[0]=(m+rhol*np.pi*self.position[2]*(dl**2)/4)*g*np.sin(self.position[0])
        f_grav[2]=-(m+rhol*np.pi*self.position[2]*(dl**2)/4)*g*np.cos(self.position[0])
        f_app[0]=m*((self.velocity[1]**2)*self.position[2]*np.sin(self.position[0])*np.cos(self.position[0])-2*self.velocity[2]*self.velocity[0])
        f_app[1]=m*(-2*self.velocity[2]*self.velocity[1]*np.sin(self.position[0])-2*self.velocity[1]*self.velocity[0]*self.position[2]*np.cos(self.position[0]))
        f_app[2]=m*(self.position[2]*(self.velocity[0]**2)+self.position[2]*(self.velocity[1]**2)*(np.sin(self.position[0])**2))
        f_aer=self.aerodynamic_force(wind);
        return f_grav+f_app+f_aer;
    def tension(self, forces):
        k=(2*m+M)/M;
        return np.array([0,0,forces[2]/k])
    def aerodynamic_force(self, wind_vect):
        matrix=self.transition_matrix()
        W_l=wind_vect.dot(matrix)
        W_a=np.array([self.velocity[0]*self.position[2], self.velocity[1]*self.position[2]*np.sin(self.position[0]), self.velocity[2]])
        W_e=W_l-W_a
        e_r=np.array([0,0,1])
        e_w=W_e-e_r*(np.dot(e_r, W_e))
        psi=np.arcsin(delta_l/d)
        e_w_norm=np.linalg.norm(e_w, 2)
        eta=np.arcsin(np.dot(W_e, e_r)*np.tan(psi)/e_w_norm)
        e_w=e_w/e_w_norm
        #if W_e==np.zeros(3) or abs(W_e.dot(e_r)/W_e.dot(e_w)*tan(psi))>1) throw ("Aborting simulation");
        W_e_norm=np.linalg.norm(W_e, 2)
        x_w=-W_e/W_e_norm
        y_w=e_w*(-np.cos(psi)*np.sin(eta))+(np.cross(e_r, e_w))*(np.cos(psi)*np.cos(eta))+e_r*np.sin(psi)
        z_w=np.cross(x_w, y_w)
        lift=-1/2*C_l*A*rho*(W_e_norm**2)*z_w
        drag=-1/2*C_d*A*rho*(W_e_norm**2)*x_w
        print("lift: ", matrix.dot(lift))
        print("drag: ", matrix.dot(drag))
        return drag+lift;
    def simulate(self, step, duration, wind):
        i=0
        continuation=True
        while continuation and i<duration:
            if i%1==0:
                print("Position at step ", i, ": ", self.position)
            continuation=self.update_state(step, wind);
            i+=1



initial_velocity=np.array([0, 0, 0], dtype=np.float64)
initial_position=np.array([np.pi/3, np.pi/24, 10], dtype=np.float64)
k=kite(initial_position, initial_velocity)
print(k)
k.simulate(0.001, 600000, np.array([10, 0, 0]))
