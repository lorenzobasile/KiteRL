#ifndef _kite
#define _kite

#include <math.h>
#include "vect.hpp"
#include "constants.hpp"
#include <utility>

class kite{
  vect position;
  vect velocity;

  public:
  kite()=default;
  kite(vect initial_position, vect initial_velocity): position{initial_position}, velocity{initial_velocity} {}
  ~kite()=default;



  bool update_state(const double step, const vect& wind, const double C_l=1.5, const double C_d=0.29, const double psi=0.05){
    std::pair<bool, vect> f=compute_force(wind, C_l, C_d, psi);
    if(!f.first){
      std::cout<<"Aborting simulation\n";
      return false;
    }
    vect force=f.second;
    vect t=tension(force);
    force-=t;
    velocity.theta+=(force.theta/(m*position.r)*step);
    velocity.phi+=(force.phi/(m*position.r*sin(position.theta))*step);
    velocity.r+=(force.r/m*step);
    if(velocity.r<0) velocity.r=0;
    position.theta+=(velocity.theta*step);
    position.phi+=(velocity.phi*step);
    position.r+=(velocity.r*step);
    if(position.theta>=pi/2) return false;
    return true;
  }

  double compute_power(const vect& wind, const double C_l, const double C_d, const double psi){
    vect f=compute_force(wind, C_l, C_d, psi).second;
    vect t=tension(f);
    return velocity.r*t.r;
  }

  double getbeta(const vect& wind){
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    W_a=W_a.tocartesian(position);
    vect W_e=wind-W_a;
    double beta=atan(W_e.z()/(sqrt(pow(W_e.x(), 2)+pow(W_e.y(), 2))));
    return beta;
  }

  std::pair<bool, vect> compute_force(const vect& wind, const double C_l, const double C_d, const double psi) const{
    vect f_grav;
    vect f_app;
    std::pair<bool, vect> aer;
    f_grav.theta=(m+rhol*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    f_grav.phi=0;
    f_grav.r=-(m+rhol*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    f_app.theta=m*(pow(velocity.phi, 2)*position.r*sin(position.theta)*cos(position.theta)-2*velocity.r*velocity.theta);
    f_app.phi=m*(-2*velocity.r*velocity.phi*sin(position.theta)-2*velocity.phi*velocity.theta*position.r*cos(position.theta));
    f_app.r=m*(position.r*pow(velocity.theta, 2)+position.r*pow(velocity.phi, 2)*pow(sin(position.theta), 2));
    aer=aerodynamic_force(wind, C_l, C_d, psi);
    if(aer.first) return std::pair<bool, vect> (true, f_grav+f_app+aer.second);
    else return std::pair<bool, vect> (false, vect{});
  }

  vect tension(const vect& forces) const{
    double k=(2*m+M)/M;
    return vect{0,0,forces.r/k};
  }

  std::pair<bool, vect> aerodynamic_force(const vect& wind_vect, const double C_l, const double C_d, const double psi) const{
    vect W_l{
      wind_vect.x()*cos(position.theta)*cos(position.phi)+wind_vect.y()*cos(position.theta)*sin(position.phi)-wind_vect.z()*sin(position.theta),
      -wind_vect.x()*sin(position.phi)+wind_vect.y()*cos(position.phi),
      wind_vect.x()*sin(position.theta)*cos(position.phi)+wind_vect.y()*sin(position.theta)*sin(position.phi)+wind_vect.z()*cos(position.theta)
    };
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    vect W_e=W_l-W_a;
    vect e_r{0,0,1};
    vect e_w=W_e-e_r*(e_r.dot(W_e));
    auto asin_arg=W_e.dot(e_r)*tan(psi)/e_w.norm();
    double eta=asin(asin_arg);
    e_w=e_w/e_w.norm();
    bool sign=W_e.x()*(abs(position.phi)<pi/2)>=0;
    vect x_w=-W_e/W_e.norm();
    vect y_w=e_w*(-cos(psi)*sin(eta))+(e_r.cross(e_w))*(cos(psi)*cos(eta))+e_r*sin(psi);
    vect z_w=x_w.cross(y_w);
    vect lift=-1.0/2*C_l*A*rho*pow(W_e.norm(), 2)*z_w;
    vect drag=-1.0/2*C_d*A*rho*pow(W_e.norm(), 2)*x_w;
    if(W_e==vect{0,0,0} || abs(W_e.dot(e_r)/W_e.dot(e_w)*tan(psi))>1) return std::pair<bool, vect> (false, vect{});
    return std::pair<bool, vect> (true, drag+lift);
  }

  void simulate(const double step, const int duration, const vect& wind){
    int i=0;
    bool continuation=true;
    while(continuation && i<duration){
      if(i%1==0)std::cout<<"Position at step "<<i<<": "<<position<<std::endl;
      continuation=update_state(step, wind);
      i++;
    }
  }

};
#endif
