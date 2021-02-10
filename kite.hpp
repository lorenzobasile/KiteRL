#ifndef _kite
#define _kite

#include <math.h>
#include "vect.hpp"
#include "constants.hpp"

class kite{
  /*vect position;
  vect velocity; //derivative of position*/

  public:

  vect position;
  vect velocity; //derivative of position

  kite()=default;
  kite(vect v1, vect v2): position{v1}, velocity{v2} {}
  ~kite()=default;


  void update_state(double step){
    const vect f=compute_force();
    position.theta+=velocity.theta*step;
    position.phi+=velocity.phi*step;
    position.r+=velocity.r*step;
    velocity.theta+=f.theta/(m*position.r);
    velocity.phi+=f.phi/(m*position.r*sin(position.theta));
    velocity.r+=f.r/m;
  }

  const vect compute_force() const{
    double f_grav_theta=(m+rho*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    double f_grav_phi=0;
    double f_grav_r=-(m+rho*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    const vect f(f_grav_theta, f_grav_phi, f_grav_r);
    return f;
  }
};
#endif
