#include "kite.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  //KITE SIMULATION
  double wind_speed_x=atol(argv[1]);
  vect wind{wind_speed_x, 0, 0};
  vect initial_position{pi/3, pi/24, 10};
  vect initial_velocity{0, 0, 0};
  kite k{initial_position, initial_velocity};
  k.simulate(0.001,600000, wind);//simulating for 10 minutes
}
