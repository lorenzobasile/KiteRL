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


  bool update_state(const double step, const vect& wind){
    const vect f=compute_force(wind);

    velocity.theta+=(f.theta/(m*position.r)*step);
    velocity.phi+=(f.phi/(m*position.r*sin(position.theta))*step);
    velocity.r+=(f.r/m*step);
    position.theta+=(velocity.theta*step);
    position.phi+=(velocity.phi*step);
    position.r+=(velocity.r*step);
    if(position.theta>=pi/2) return false;
    return true;
  }

  vect compute_force(const vect& wind) const{
    vect f_grav;
    vect f_app;
    vect f_aer;
    vect f_trac;

    f_grav.theta=(m+rhol*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    f_grav.phi=0;
    f_grav.r=-(m+rhol*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    f_app.theta=m*(pow(velocity.phi, 2)*position.r*sin(position.theta)*cos(position.theta)-2*velocity.r*velocity.theta);
    f_app.phi=m*(-2*velocity.r*velocity.phi*sin(position.theta)-2*velocity.phi*velocity.theta*position.r*cos(position.theta));
    f_app.r=m*(position.r*pow(velocity.theta, 2)+position.r*pow(velocity.phi, 2)*pow(sin(position.theta), 2));
    f_aer=aerodynamic_force(wind);
    f_trac.theta=0;
    f_trac.phi=0;
    f_trac.r=f_grav.r+f_app.r+f_aer.r; //r has no acceleration
    /*std::cout<<"f_app: "<<f_app.tocartesian(position)<<std::endl;
    std::cout<<"f_grav: "<<f_grav.tocartesian(position)<<std::endl;
    std::cout<<"f_trac: "<<f_trac.tocartesian(position)<<std::endl;*/
    return f_grav+f_app+f_aer-f_trac;
  }

  vect aerodynamic_force(const vect& wind_vect) const{
    vect W_l{//for wind theta=x, phi=y, r=z
      wind_vect.x()*cos(position.theta)*cos(position.phi)+wind_vect.y()*cos(position.theta)*sin(position.phi)-wind_vect.z()*sin(position.theta),
      -wind_vect.x()*sin(position.phi)+wind_vect.y()*cos(position.phi),
      wind_vect.x()*sin(position.theta)*cos(position.phi)+wind_vect.y()*sin(position.theta)*sin(position.phi)+wind_vect.z()*cos(position.theta)
    };
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    vect W_e=W_l-W_a;
    vect e_r{0,0,1};
    //vect e_r{sin(position.theta)*cos(position.phi), sin(position.theta)*sin(position.phi), cos(position.theta)};
    vect e_w=W_e-e_r*(e_r.dot(W_e));
    double psi=asin(delta_l/d);
    //std::cout<<"arcsin arg: "<<W_e.dot(e_r)*tan(psi)/e_w.norm()<<std::endl;
    double eta=asin(W_e.dot(e_r)*tan(psi)/e_w.norm());
    //std::cout<<"e_w: "<<e_w<<" "<<e_w.norm()<<std::endl;
    e_w=e_w/e_w.norm();
    //std::cout<<"W_e: "<<W_e<<" "<<W_e.norm()<<std::endl;
    vect x_w=-W_e/W_e.norm();
    vect y_w=e_w*(-cos(psi)*sin(eta))+(e_r.cross(e_w))*(cos(psi)*cos(eta))+e_r*sin(psi);
    vect z_w=x_w.cross(y_w);
    vect lift=-1.0/2*C_l*A*rho*pow(W_e.norm(), 2)*z_w;
    vect drag=-1.0/2*C_d*A*rho*pow(W_e.norm(), 2)*x_w;
    std::cout<<"lift: "<<lift.tocartesian(position)<<std::endl;
    std::cout<<"drag: "<<drag.tocartesian(position)<<std::endl;
    //std::cout<<"x_w: "<<x_w<<std::endl<<"y_w: "<<y_w<<std::endl<<"z_w: "<<z_w<<std::endl;
    return drag+lift;
  }


  /* EQUIVALENT FORMULATION
  vect aerodynamic_force(const vect& wind_vect) const{
    vect w_l{//for wind theta=x, phi=y, r=z
      wind_vect.x()*cos(position.theta)*cos(position.phi)+wind_vect.y()*cos(position.theta)*sin(position.phi)-wind_vect.z()*sin(position.theta),
      -wind_vect.x()*sin(position.phi)+wind_vect.y()*cos(position.phi),
      wind_vect.x()*sin(position.theta)*cos(position.phi)+wind_vect.y()*sin(position.theta)*sin(position.phi)+wind_vect.z()*cos(position.theta)
    };
    vect w_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    vect w_e=w_l-w_a;
    vect e_r{sin(position.theta)*cos(position.phi), sin(position.theta)*sin(position.phi), cos(position.theta)};
    vect e_l=w_e/w_e.norm();
    vect w_ep=w_e-e_r*(e_r.dot(w_e));
    vect e_w=w_ep/w_ep.norm();
    vect e_0=e_r.cross(e_w);
    double psi=asin(delta_l/d);
    double eta=asin(w_e.dot(e_r)*tan(psi)/w_ep.norm());
    vect e_t=e_w*(-cos(psi)*sin(eta))+e_0*(cos(psi)*cos(eta))+e_r*sin(psi);
    double F_l=1.0/2*rho*pow(w_e.norm(), 2)*A*C_l;
    double F_d=1.0/2*rho*pow(w_e.norm(), 2)*A*C_d;
    return F_l*(e_l.cross(e_t))+F_d*e_l;
  }*/

  void simulate(const double step, const int duration, const vect& wind){
    int i=0;
    bool continuation=true;
    while(continuation && i<duration){
      if(i%1==0)std::cout<<"Position at step "<<i<<": "<<position<<std::endl;
      continuation=update_state(step, wind);
      i++;
      //if(!continuation) std::cout<<"Final position: "<<position.tocartesian()<<std::endl;
    }
  }

};
#endif
