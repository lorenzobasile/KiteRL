#ifndef _kite
#define _kite

#include <math.h>
#include "vect.hpp"
#include "constants.hpp"

class kite{

  vect position;
  vect velocity;

  public:

  kite()=default;
  kite(vect initial_position, vect initial_velocity): position{initial_position}, velocity{initial_velocity} {}
  ~kite()=default;


  void update_state(const double step, const double wind){
    const vect f=compute_force(wind);
    velocity.theta+=f.theta/(m*position.r)*step;
    velocity.phi+=f.phi/(m*position.r*sin(position.theta))*step;
    velocity.r+=f.r/m*step;
    position.theta+=velocity.theta*step;
    position.phi+=velocity.phi*step;
    position.r+=velocity.r*step;
    position.theta=atan2(sin(position.theta),cos(position.theta)); //?
    position.phi=atan2(sin(position.phi),cos(position.phi));
  }

  vect compute_force(const double wind) const{
    vect f_grav;
    vect f_app;
    vect f_aer;
    vect f_trac;

    vect constant_horizontal_wind{wind,0,0};
    f_grav.theta=(m+rhol*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    f_grav.phi=0;
    f_grav.r=-(m+rhol*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    f_app.theta=m*(pow(velocity.phi, 2)*position.r*sin(position.theta)*cos(position.theta)-2*velocity.r*velocity.theta);
    f_app.phi=m*(-2*velocity.r*velocity.phi*sin(position.theta)-2*velocity.phi*velocity.theta*position.r*cos(position.theta));
    f_app.r=m*(position.r*pow(velocity.theta, 2)+position.r*pow(velocity.phi, 2)*pow(sin(position.theta), 2));
    f_aer=aerodynamic_force(constant_horizontal_wind);
    f_trac.theta=0;
    f_trac.phi=0;
    f_trac.r=f_grav.r+f_app.r+f_aer.r; //r has no acceleration

    return f_grav+f_app+f_aer-f_trac;
  }

  vect aerodynamic_force(const vect& wind_vect) const{
    vect W_l{//for wind theta=x, phi=y, r=z
      wind_vect.theta*cos(position.theta)*cos(position.phi)+wind_vect.phi*cos(position.theta)*sin(position.phi)-wind_vect.r*sin(position.theta),
      -wind_vect.theta*sin(position.phi)+wind_vect.phi*cos(position.phi),
      wind_vect.theta*sin(position.theta)*cos(position.phi)+wind_vect.phi*sin(position.theta)*sin(position.phi)+wind_vect.r*cos(position.theta)
    };
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    vect W_e=W_l-W_a;
    vect e_r{sin(position.theta)*cos(position.phi), sin(position.theta)*sin(position.phi), cos(position.theta)};
    vect e_w=W_e-e_r*(e_r.dot(W_e));
    double psi=asin(delta_l/d);
    double eta=asin(W_e.dot(e_r)*tan(psi)/e_w.norm());
    e_w=e_w/e_w.norm();
    vect x_w=-W_e/W_e.norm();
    vect y_w=e_w*(-cos(psi)*sin(eta))+(e_r.cross(e_w))*(cos(psi)*cos(eta))+e_r*sin(psi);
    vect z_w=x_w.cross(y_w);
    return -1.0/2*C_d*A*rho*pow(W_e.norm(), 2)*x_w-1.0/2*C_l*A*rho*pow(W_e.norm(), 2)*z_w;
  }

  void simulate(const double step, const int duration, const double wind){
    for(int i=0; i<duration; i++){
      update_state(step, wind);
      if(i%100==0)std::cout<<"Position at step "<<i<<": "<<position<<std::endl;
    }
  }

};
#endif
