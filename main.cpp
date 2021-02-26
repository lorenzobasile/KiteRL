#include "kite.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  double wind_speed_x=atol(argv[1]);
  vect initial_position{pi/4, pi/12, 10};
  vect initial_velocity{0, 0, 0};
  kite k{initial_position, initial_velocity};
  //k.simulate(0.001,600000,wind_speed_x);//simulating for 10 minutes
  vect test1{3, 1, 2};
  std::cout<<"test1: "<<test1<<std::endl;
  vect test2{-3, 0, 4};
  std::cout<<"test2: "<<test2<<std::endl;
  std::cout<<"2*test1: "<<2*test1<<std::endl;
  std::cout<<"test1*2: "<<test1*2<<std::endl;
  std::cout<<"-test1: "<<-test1<<std::endl;
  std::cout<<"test1-test2: "<<test1-test2<<std::endl;
  std::cout<<"norm of test1: "<<test1.norm()<<std::endl;
  std::cout<<"test1.dot(test2): "<<test1.dot(test2)<<std::endl;
  std::cout<<"test1.cross(test2): "<<test1.cross(test2)<<std::endl;
}
