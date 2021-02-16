#include "kite.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  double wind_speed_x=atol(argv[1]);
  vect initial_position{pi/4, pi/4, 10};
  vect initial_velocity{0, 0, 0};
  kite k{initial_position, initial_velocity};
  k.simulate(0.01,60000,wind_speed_x);//simulating for 10 minutes
}
