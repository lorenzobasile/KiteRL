#ifndef _kite
#define _kite

#include <math.h>
#include "vect.hpp"
#include "constants.hpp"

class kite{

  public:

  vect position;
  vect velocity;

  kite() : position{}, velocity{} {};
  kite(vect initial_position, vect initial_velocity): position{initial_position}, velocity{initial_velocity} {}
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
    vect f_grav;
    vect f_app;
    vect f_aer;
    vect f_trac;

    f_grav.theta=(m+rho*pi*position.r*pow(dl, 2)/4)*g*sin(position.theta);
    f_grav.phi=0;
    f_grav.r=-(m+rho*pi*position.r*pow(dl, 2)/4)*g*cos(position.theta);
    f_app.theta=m*(pow(velocity.phi, 2)*position.r*sin(position.theta)*cos(position.theta)-2*velocity.r*velocity.theta);
    f_app.phi=m*(-2*velocity.r*velocity.phi*sin(position.theta)-2*velocity.phi*velocity.theta*position.r*cos(position.theta));
    f_app.r=m*(position.r*pow(velocity.theta, 2)+position.r*pow(velocity.phi, 2)*pow(sin(position.theta), 2));
    /*f_aer.theta=
    f_aer.phi=
    f_aer.r=
    f_trac.theta=0;
    f_trac.phi=0;
    f_trac.r=*/

    return f_grav+f_app+f_aer+f_trac;
  }

  void simulate(const double step, const int duration){
    for(int i=0; i<duration; i++){
      update_state(step);
    }
  }

};
#endif
