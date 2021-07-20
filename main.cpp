#include "kite.hpp"
#include "wind.hpp"
#include <iostream>
/*void read_grid_file(std::string path, double** grid_data){
    std::ifstream file;
    file.open(path);
    std::string line;
    vecd l = vecd(3);
    int count = 0;
    while ( getline (file, line) ){
        l = str2vecd(line, " ", false);
        grid_data[count][0] = l[0];
        grid_data[count][1] = l[2];
        grid_data[count][2] = l[1];
        count++;
    }
    file.close();
}*/

void read_grid_file2(std::string path, double** grid_data1, double** grid_data2){
    std::ifstream file;
    file.open(path);
    std::string line;
    vecd l = vecd(3);
    int count = 0;
    while ( getline (file, line) ){
        l = str2vecd(line, " ", false);
        if (count<185193){
          grid_data1[count][0] = l[0];
          grid_data1[count][1] = l[2];
          grid_data1[count][2] = l[1];
        }

        else{
          //std::cout<<count-185193;

          grid_data2[count-185193][0] = l[0];
          grid_data2[count-185193][1] = l[2];
          grid_data2[count-185193][2] = l[1];
        }
        count++;
    }
    file.close();
}

int main(int argc, char* argv[]){

  vect initial_position{pi/6, 0.0, 50.0};
  vect initial_velocity{0.0, 0.0, 0.0};
  //Wind3d_turboframe wind;
  /*const static int n_grid_points = 185193;
  double** q_grid=new double*[n_grid_points];
  for(int i=0; i<n_grid_points; i++){
    q_grid[i]=new double[3];
  }
  double** v_grid=new double*[n_grid_points];
  for(int i=0; i<n_grid_points; i++){
    v_grid[i]=new double[3];
  }
  read_grid_file2("q.txt", q_grid, v_grid);
  std::cout<<"ciaoo";*/
  //Wind3d_turboframe wind(q_grid, v_grid);
  Wind3d_turbo wind;
  //read_grid_file("v.txt", v_grid);
  //read_grid_file("q.txt", q_grid);
  kite k{initial_position, initial_velocity, &wind};
  k.simulate(0.0001,600000);//simulating for 10 minutes
}
