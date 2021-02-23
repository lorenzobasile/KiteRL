#include "kite.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  double wind_speed_x=atol(argv[1]);
  vect initial_position{pi/4, pi/12, 10};
  vect initial_velocity{0, 0, 0};
  kite k{initial_position, initial_velocity};
  k.simulate(0.001,600000,wind_speed_x);//simulating for 10 minutes
}
