#include "kite.hpp"

extern "C" {
  vect init_vect(double theta, double phi, double r){
    return vect{theta, phi, r};
  }
  kite init_kite(vect p, vect v){
    return kite{p, v};
  }
  bool simulation_step(kite* k, const double step, const vect wind){
    bool continuation=true;
    try{
      continuation=k->update_state(step, wind);
    } catch(const char* exc) {
      continuation=false;
      std::cout<<exc<<std::endl;
    }
    return continuation;
  }

  bool simulate(kite* k, const double C_l, const double C_d, const double psi, const int integration_steps, const double step, const vect wind){
    bool continuation=true;
    int i=0;
    try{
      while(continuation && i<integration_steps) {
        continuation=k->update_state(step, wind, C_l, C_d, psi);
        std::cout<<i<<std::endl;
        i++;
      }
    } catch(const char* exc) {
      continuation=false;
      std::cout<<exc<<std::endl;
    }
    return continuation;
  }

  double getbeta(kite* k, const vect wind){
    return k->getbeta(wind);
  }

  double getreward(kite* k, const vect wind, const double C_l, const double C_d, const double psi){
    return (k->compute_power(wind, C_l, C_d, psi));
  }

}
