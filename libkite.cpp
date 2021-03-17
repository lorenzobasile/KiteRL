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

}
