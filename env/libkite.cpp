#include "kite.hpp"

extern "C" {

  int simulation_step(kite* k, const double step){
    auto status=k->update_state(step);
    return status;
  }
  void init_const_wind(kite* k, double vel){
    Wind3d_const* wind = new Wind3d_const{vel};
    wind->init(k->position.r*sin(k->position.theta)*cos(k->position.phi), k->position.r*sin(k->position.theta)*sin(k->position.phi), k->position.r*cos(k->position.theta));
    k->wind=wind;
  }
  void init_lin_wind(kite* k, double vel_ground, double ang_coef){
    Wind3d_lin* wind = new Wind3d_lin{vel_ground, ang_coef};
    wind->init(k->position.r*sin(k->position.theta)*cos(k->position.phi), k->position.r*sin(k->position.theta)*sin(k->position.phi), k->position.r*cos(k->position.theta));
    k->wind=wind;
  }
  void init_turboframe_wind(kite* k){
    Wind3d_turboframe* wind = new Wind3d_turboframe;
    wind->init(k->position.r*sin(k->position.theta)*cos(k->position.phi), k->position.r*sin(k->position.theta)*sin(k->position.phi), k->position.r*cos(k->position.theta));
    k->wind=wind;
  }

  void init_turbo_wind(kite* k){
      std::cout<<"Constructing turbo wind"<<std::endl;
      Wind3d_turbo* wind = new Wind3d_turbo;
      wind->init(k->position.r*sin(k->position.theta)*cos(k->position.phi), k->position.r*sin(k->position.theta)*sin(k->position.phi), k->position.r*cos(k->position.theta));
      k->wind=wind;
    }
   void reset_turbo_wind(kite* k){
   	k->wind->init(k->position.r*sin(k->position.theta)*cos(k->position.phi), k->position.r*sin(k->position.theta)*sin(k->position.phi), k->position.r*cos(k->position.theta));
   }

  int simulate(kite* k, const int integration_steps, const double step){
    bool continuation=true;
    int status;
    int i=0;
    while(continuation && i<integration_steps) {
        status=k->update_state(step);
        continuation=(status==0);
        i++;
      }
    return status;
  }

  double getbeta(kite* k){
    return k->getbeta();
  }

  vect getvrel(kite* k){
    return k->getvrel();
  }

  vect getaccelerations(kite* k){
    return k->get_accelerations().second;
  }


  double getreward(kite* k){
    return (k->compute_power())/3600000;
  }
  
  double get_effective_wind_speed(kite* k){
    return k->effective_wind_speed();
    
  }

}
