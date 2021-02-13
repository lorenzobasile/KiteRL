#include "kite.hpp"
#include <iostream>

int main(){
  vect initial_position;
  std::cout<<initial_position.r<<"\n";
  vect initial_velocity;
  kite k{initial_position, initial_velocity};
  std::cout<<k.position.theta<<"\n";
  k.simulate(1,2);
  std::cout<<k.position.theta<<"\n";
  vect force{3,2,1};
  vect force2{2,1,4};
  force+=force2;
  vect cross_product;
  cross_product=force.cross(force2);
  std::cout<<cross_product<<"\n";

}
