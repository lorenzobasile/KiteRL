#include "kite.hpp"

extern "C" {
  vect init_vect(double theta, double phi, double r){
    return vect{theta, phi, r};
  }
  kite init_kite(vect p, vect v, double w){
    return kite{p, v, w};
  }
  int simulation_step(kite* k, const double step){
    auto status=k->update_state(step);
    return status;
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

  vect getaccelerations(kite* k){
    return k->get_accelerations().second;
  }


  double getreward(kite* k){
    return (k->compute_power());
  }

}
