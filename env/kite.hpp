#ifndef _kite
#define _kite

#include <math.h>
#include "vect.hpp"
#include "constants.hpp"
#include <utility>

class kite{
  vect position;
  vect velocity;
  double init_wind;
  vect wind;

  public:
  double C_l;
  double C_d;
  double psi;

  kite()=default;
  kite(vect initial_position, vect initial_velocity, double initial_wind): position{initial_position}, velocity{initial_velocity}, init_wind{initial_wind}, wind{initial_wind,0,0}, C_l{0.2}, C_d{0.05}, psi{-0.05} {}
  ~kite()=default;

  int update_state(const double step){
    auto accel=get_accelerations();
    if(!accel.first){
      return 2;
    }
    auto accelerations=accel.second;
    //std::cout<<"force "<<force.norm()<<std::endl;
    velocity.theta+=(accelerations.theta*step);
    velocity.phi+=(accelerations.phi*step);
    velocity.r+=(accelerations.r*step);
    if(velocity.r<0) velocity.r=0;
    position.theta+=(velocity.theta*step);
    position.phi+=(velocity.phi*step);
    position.r+=(velocity.r*step);
    if(position.theta>=pi/2) return 1;
    update_wind();
    return 0;
  }
    
  void update_wind(){
    double z= position.r*cos(position.theta);
    wind={init_wind+slope*(z-43),0,0};
  }

  double compute_power(){
    vect f=compute_force().second;
    vect t=tension(f);
    return velocity.r*t.r;
  }

  double getbeta(){
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    W_a=W_a.tocartesian(position);
    vect W_e=wind-W_a;
    double beta=atan(W_e.z()/(sqrt(pow(W_e.x(), 2)+pow(W_e.y(), 2))));
    return beta;
  }

  std::pair<bool, vect> get_accelerations(){
    std::pair<bool, vect> f=compute_force();
    if(!f.first){
      //std::cout<<"Aborting simulation\n";
      return {false, vect{}};
    }
    vect force=f.second;
    vect t=tension(force);
    //std::cout<<"tension "<<t.norm()<<std::endl;
    force-=t;
    return {true, vect{force.theta/(m*position.r), force.phi/(m*position.r*sin(position.theta)), force.r/m}};
  }

  std::pair<bool, vect> compute_force() const{
    vect f_grav;
    vect f_app;
    std::pair<bool, vect> aer;
    f_grav.theta=(m+rhol*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    f_grav.phi=0;
    f_grav.r=-(m+rhol*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    f_app.theta=m*(pow(velocity.phi, 2)*position.r*sin(position.theta)*cos(position.theta)-2*velocity.r*velocity.theta);
    f_app.phi=m*(-2*velocity.r*velocity.phi*sin(position.theta)-2*velocity.phi*velocity.theta*position.r*cos(position.theta));
    f_app.r=m*(position.r*pow(velocity.theta, 2)+position.r*pow(velocity.phi, 2)*pow(sin(position.theta), 2));
    aer=aerodynamic_force();
    if(aer.first) return std::pair<bool, vect> (true, f_grav+f_app+aer.second);
    else return std::pair<bool, vect> (false, vect{});
  }

  /*vect tension(const vect& forces) const{
    double k=(2*m+M)/M;
    double at=2*1/(M*k*a);
    return vect{0,0, forces.r/k+at*velocity.r/a};
  }*/
    vect tension(const vect& forces) const{
        auto num=M*a*forces.r+2*m*10*velocity.r/a;
        auto denom=2*m+M*a;
        return vect{0,0,num/denom};
    }

  std::pair<bool, vect> aerodynamic_force() const{
    vect W_l{
      wind.x()*cos(position.theta)*cos(position.phi)+wind.y()*cos(position.theta)*sin(position.phi)-wind.z()*sin(position.theta),
      -wind.x()*sin(position.phi)+wind.y()*cos(position.phi),
      wind.x()*sin(position.theta)*cos(position.phi)+wind.y()*sin(position.theta)*sin(position.phi)+wind.z()*cos(position.theta)
    };
    vect W_a{velocity.theta*position.r, velocity.phi*position.r*sin(position.theta), velocity.r};
    //std::cout<<"kite"<<W_a.norm()<<std::endl;
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
    //std::cout<<"y_w "<<y_w.phi<<std::endl;
    //std::cout<<"z_w "<<z_w.phi<<std::endl;
    //std::cout<<W_e.norm()<<std::endl;
    vect lift=-1.0/2*C_l*A*rho*pow(W_e.norm(), 2)*z_w;
    vect drag=-1.0/2*C_d*A*rho*pow(W_e.norm(), 2)*x_w;
    //std::cout<<"speed: "<<W_e.norm()<<std::endl;
    if(W_e==vect{0,0,0} || abs(W_e.dot(e_r)/W_e.dot(e_w)*tan(psi))>1) return std::pair<bool, vect> (false, vect{});
    //std::cout<<"drag: "<<drag.tocartesian(position)<<std::endl;
    //std::cout<<"lift: "<<lift.tocartesian(position)<<std::endl;
    return std::pair<bool, vect> (true, drag+lift);
  }

  void simulate(const double step, const int duration){
    int i=0;
    bool continuation=true;
    while(continuation && i<duration){
      if(i%1==0)std::cout<<"Position at step "<<i<<": "<<position<<std::endl;
      continuation=(update_state(step)==0);
      i++;
    }
  }

};
#endif
