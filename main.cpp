#include "kite.hpp"
#include <iostream>

int main(){
  vect initial_position{pi/4, pi/4, 10};
  std::cout<<initial_position.r<<"\n";
  vect initial_velocity{0, 0, 0};
  kite k{initial_position, initial_velocity};
  k.simulate(0.01,20);
}
