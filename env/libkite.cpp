#include "kite.hpp"

extern "C" {
  vect init_vect(double theta, double phi, double r){
    return vect{theta, phi, r};
  }
  kite init_kite(vect p, vect v){
    return kite{p, v};
  }
  int simulation_step(kite* k, const double step, const vect wind){
    auto status=k->update_state(step, wind);
    return status;
  }

  int simulate(kite* k, const int integration_steps, const double step, const vect wind){
    bool continuation=true;
    int status;
    int i=0;
    while(continuation && i<integration_steps) {
        status=k->update_state(step, wind);
        continuation=(status==0);
        i++;
      }
    return status;
  }

  double getbeta(kite* k, const vect wind){
    return k->getbeta(wind);
  }

  vect getaccelerations(kite* k, const vect wind){
    return k->get_accelerations(wind).second;
  }


  double getreward(kite* k, const vect wind){
    return (k->compute_power(wind));
  }

}
