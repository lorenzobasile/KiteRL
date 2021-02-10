#include "kite.hpp"
#include <iostream>

int main(){
  vect initial_position;
  std::cout<<initial_position.r<<"\n";
  vect initial_velocity;
  kite k{initial_position, initial_velocity};
  k.update_state(1);
  k.update_state(1);
  std::cout<<k.position.theta<<"\n";

}
