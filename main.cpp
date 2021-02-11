#include "kite.hpp"
#include <iostream>

int main(){
  vect initial_position;
  std::cout<<initial_position.r<<"\n";
  vect initial_velocity;
  kite k{initial_position, initial_velocity};
  std::cout<<k.position.theta<<"\n";
  k.update_state(1);
  k.update_state(1);
  std::cout<<k.position.theta<<"\n";

  vect force{3,2,1};
  vect force2{2,1,4};
  force+=force2;
  std::cout<<force<<"\n";

}
