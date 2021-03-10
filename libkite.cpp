#include "kite.hpp"

extern "C" {
  vect init_vect(double theta, double phi, double r){
    return vect{theta, phi, r};
  }
  kite init_kite(vect p, vect v){
    return kite{p, v};
  }
  void simulate(kite k, const double step, const int duration, const vect wind){
    try{
      k.simulate(step, duration, wind);//simulating for 10 minutes
    } catch(const char* exc) {
      std::cout<<exc<<std::endl;
    }
  }

}
