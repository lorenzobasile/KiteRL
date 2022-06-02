#ifndef _wind
#define _wind

#include "vect.hpp"
#include <fstream>
#include "utils/utils.h"
#ifdef PARALLEL
#include <omp.h>
#endif
// 3D WINDS

/* Abstract class */
class Wind3d {
    protected:
        double m_vel[3];
    public:
        /* Velocity variable that is referenced to by velocity method*/
        virtual double* init(double x0, double y0, double z0) { return velocity(x0,y0,z0,0.0); };
        virtual double* velocity(double x, double y, double z, double t) = 0;
        vect to_vect() const{
          return vect{m_vel[0], m_vel[1], m_vel[2]};
        }
        virtual ~Wind3d()=default;
};

// Constant wind
class Wind3d_const : public Wind3d {
    public:
        Wind3d_const(double vel) { m_vel[0] = vel; m_vel[1] = 0; m_vel[2] = 0;  };
        virtual double* velocity(double x, double y, double z, double t) {return m_vel; }
        const std::string descr() const { return "3d constant wind."; }
};

/* Linear wind */
class Wind3d_lin : public Wind3d {

    private:
        /* Wind x speed on the gorund */
        double vel_ground;
        /* Angular coefficient of the wind profile */
        double ang_coef;

    public:
        Wind3d_lin(double vel_ground, double ang_coef) : vel_ground{vel_ground}, ang_coef{ang_coef}
        { m_vel[1] = 0; m_vel[2]=0; };

        virtual double* velocity(double x, double y, double z, double t) {
            m_vel[0] = ang_coef * z + vel_ground;
            return m_vel;
        }
        virtual ~Wind3d_lin()=default;

};


// Wind of a static frame of a turbolent flow
class Wind3d_turboframe : public Wind3d {

protected:
        const static int n_grid_points = 185193;
        //const static int n_grid_points = 499059;
        const static int n_x_axis_points = 57;
        const static int n_y_axis_points = 57;
        const static int n_z_axis_points = 57;
        //const static int n_x_axis_points = 71;
        //const static int n_y_axis_points = 71;
        //const static int n_z_axis_points = 99;
        constexpr static double x_size = 100.531*1;
        constexpr static double y_size = 100.531*1;
        constexpr static double z_half_size = 50*1;

        const static int n_frames = 500;
        const double delta_time = 0.2;

        double** q_grid;
        double** v_grid;
        double wind_amplif=1;
        int n_x, n_y, n_z;


    public:

        void read_grid_file(std::string path, double** grid_data){
            std::ifstream file;
            file.open(path);
            int factor=1;

            std::string line;
            vecd l = vecd(3);
            int count = 0;
            while ( getline (file, line) ){
                l = str2vecd(line, " ", false);
                grid_data[count][0] = l[0]*factor;
                grid_data[count][1] = l[2]*factor;
                grid_data[count][2] = l[1]*factor;
                count++;
            }
            file.close();
        }
        Wind3d_turboframe() {
          q_grid=new double*[n_grid_points];
          v_grid=new double*[n_grid_points];
          for(int i=0; i<n_grid_points; i++){
            q_grid[i]=new double[3];
            v_grid[i]=new double[3];
          }
          read_grid_file("../env/q.txt", q_grid);
          read_grid_file("../env/v.txt", v_grid);
        };
        ~Wind3d_turboframe(){
          for(int i=0; i<n_grid_points; i++){
            delete[] q_grid[i];
            delete[] v_grid[i];
          }
          delete[] q_grid;
          delete[] v_grid;
        }




        virtual double* init(double x, double y, double z) {
            //std::cout << x << " " << y << " "<< z << "\n";
            // Imposing the periodic boundary condition on the x
            int mx = x/x_size;
            x -= mx*x_size;

            // Translating the y such that 0 is in the middle of the canal and imposing boundary conditions
            y += y_size / 2.0;
            float my = floor(y / y_size);
            y -= my*y_size;

            // Translating the z such that 0 is on the ground (we assume that z doesn't goes out of bounds)
            // Below the ground the velocity is the one on the ground
            z -= z_half_size;

            //std::cout << x << " " << y << " "<< z << "\n";
            for (size_t i = 0; i < n_y_axis_points-1; i++) {
                if (y>=q_grid[i][1] && y<q_grid[i+1][1]){
                  n_y=i;
                  break;
                }
            }
            //std::cout << n_y << "\n";
            for (size_t i = 0; i < n_x_axis_points-1; i++) {
                //std::cout << i << " " << q_grid[i*n_xy_axis_points][0] << " " << q_grid[(i+1)*n_xy_axis_points][0] << "\n";
                if (x>=q_grid[i*n_y_axis_points][0] && x<q_grid[(i+1)*n_y_axis_points][0]){
                  n_x=i;
                  break;
                }
            }
            //std::cout << n_x << "\n";
            if (z < -z_half_size) n_z = 0;
            else {
                for (size_t i = 0; i < n_z_axis_points-1; i++) {
                    //std::cout << i << " " << q_grid[i*n_xy_axis_points*n_xy_axis_points][1] << " " << q_grid[i+(n_xy_axis_points*n_xy_axis_points)][1] << "\n";
                    if (z>=q_grid[i*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(i+1)*n_x_axis_points*n_y_axis_points][2]){
                      n_z=i;
                      //if ((i+1)*n_xy_axis_points*n_xy_axis_points > n_grid_points-1) {
                          //std::cout << "index out of bounds" << '\n';}
                      break;
                    }
                }
            }
            //std::cout << n_x << " " << n_y << " " << n_z << "\n";
            return compute_velocity(x, y, z, 0);
        }

        virtual double* velocity(double x, double y, double z, double t){

            int frame = t / delta_time;
            frame = frame % n_frames;

            // Imposing the periodic boundary condition on the x
            int mx = x/x_size;
            x -= mx*x_size;

            // Translating the y such that 0 is in the middle of the canal and imposing boundary conditions
            y += y_size / 2.0;
            float my = floor(y / y_size);
            y -= my*y_size;

            // Translating the z such that 0 is on the ground (we assume that z doesn't goes out of bounds)
            // Below the ground the velocity is the one on the ground
            z -= z_half_size;

            if (!(y>=q_grid[n_y][1] && y<q_grid[n_y+1][1]))
            {
                if (n_y != 0 && y>=q_grid[n_y-1][1] && y<q_grid[n_y][1]) n_y -= 1;
                else if (n_y == 0 && y>=q_grid[n_y_axis_points-2][1] && y<q_grid[n_y_axis_points-1][1]) n_y = n_y_axis_points-2;
                else if (n_y != n_y_axis_points-2 && y>=q_grid[n_y+1][1] && y<q_grid[n_y+2][1]) n_y += 1;
                else if (n_y == n_y_axis_points-2 && y>=q_grid[0][1] && y<q_grid[1][1]) n_y = 0;
                else {
                    for (size_t i = 0; i < n_y_axis_points-1; i++) {
                        if (y>=q_grid[i][1] && y<q_grid[i+1][1]){
                          n_y=i;
                          break;
                        }
                    }
                }
            }

            if (!(x>=q_grid[n_x*n_y_axis_points][0] && x<q_grid[(n_x+1)*n_y_axis_points][0])){
                if (n_x != 0 && x>=q_grid[(n_x-1)*n_y_axis_points][0] && x<q_grid[n_x*n_y_axis_points][0]) n_x -= 1;
                else if (n_x == 0 && x>=q_grid[(n_x_axis_points-2)*n_y_axis_points][0] && x<q_grid[(n_x_axis_points-1)*n_y_axis_points][0]) n_x = n_x_axis_points-2;
                else if (n_x != n_x_axis_points-2 && x>=q_grid[(n_x+1)*n_y_axis_points][0] && x<q_grid[(n_x+2)*n_y_axis_points][0]) n_x += 1;
                else if (n_x == n_x_axis_points-2 && x>=q_grid[0][0] && x<q_grid[n_y_axis_points][0]) n_x = 0;
                else {
                    for (size_t i = 0; i < n_x_axis_points-1; i++) {
                        if (x>=q_grid[i*n_y_axis_points][0] && x<q_grid[(i+1)*n_y_axis_points][0]){
                          n_x=i;
                          break;
                        }
                    }
                }
            }

            if (!(z>=q_grid[n_z*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(n_z+1)*n_x_axis_points*n_y_axis_points][2])) {
                if (z < -z_half_size) n_z = 0;
                else {
                    if (n_z != 0 && z>=q_grid[(n_z-1)*n_x_axis_points*n_y_axis_points][2] && z<q_grid[n_z*n_x_axis_points*n_y_axis_points][2]) n_z -= 1;
                    else if (n_z != n_z_axis_points-2 && z>=q_grid[(n_z+1)*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(n_z+2)*n_x_axis_points*n_y_axis_points][2]) n_z += 1;
                    else {
                        for (size_t i = 0; i < n_z_axis_points-1; i++) {
                            if (z>=q_grid[i*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(i+1)*n_x_axis_points*n_y_axis_points][2]){
                              n_z=i;
                              break;
                            }
                        }
                    }
                }
            }
            //std::cout << n_x << " " << n_y << " " << n_z << "\n";

            return compute_velocity(x, y, z, frame);
        }


        virtual double* compute_velocity(double x, double y, double z, int frame) {

            int ind=n_z*n_x_axis_points*n_y_axis_points+n_x*n_y_axis_points+n_y;
            //std::cout << x << " " << y << " "<< z << " " << n_x << " " << n_y << " "<< n_z << "\n";

            double q_d[3];
            q_d[0]=(x-q_grid[ind][0])/(q_grid[ind+n_y_axis_points][0]-q_grid[ind][0]);
            q_d[1]=(y-q_grid[ind][1])/(q_grid[ind+1][1]-q_grid[ind][1]);
            q_d[2]=(z-q_grid[ind][2])/(q_grid[ind+n_x_axis_points*n_y_axis_points][2]-q_grid[ind][2]);
            //std::cout << q_d[0] << " " << q_d[1] << " "<< q_d[2] << "\n";

            double vel_corner[8];
            for (size_t i=0; i<3; i++){
                vel_corner[0] = v_grid[ind][i];
                vel_corner[1] = v_grid[ind+1][i];
                vel_corner[2] = v_grid[ind+n_x_axis_points*n_y_axis_points+1][i];
                vel_corner[3] = v_grid[ind+n_x_axis_points*n_y_axis_points][i];
                vel_corner[4] = v_grid[ind+n_y_axis_points][i];
                vel_corner[5] = v_grid[ind+n_y_axis_points+1][i];
                vel_corner[6] = v_grid[ind+n_x_axis_points*n_y_axis_points+n_y_axis_points+1][i];
                vel_corner[7] = v_grid[ind+n_x_axis_points*n_y_axis_points+n_y_axis_points][i];
                //std::cout << vel_corner[0] << " " << vel_corner[1] << " "<< vel_corner[2] << " "<< vel_corner[3] << " ";
                //std::cout << vel_corner[4] << " " << vel_corner[5] << " "<< vel_corner[6] << " "<< vel_corner[7] << "\n";
                m_vel[i] = interpolation(q_d, vel_corner)*wind_amplif;
                //std::cout << "vel: "<< m_vel[i] << "\n ";
            }
            //std::cout << '\n';

            return m_vel;
        }


        double interpolation(double q_d[], double vel[]){
           double vel_x[4];
           double vel_xy[2];
           vel_x[0]=vel[0]*(1-q_d[0])+vel[4]*q_d[0];
           vel_x[1]=vel[1]*(1-q_d[0])+vel[5]*q_d[0];
           vel_x[2]=vel[2]*(1-q_d[0])+vel[6]*q_d[0];
           vel_x[3]=vel[3]*(1-q_d[0])+vel[7]*q_d[0];

           vel_xy[0]=vel_x[0]*(1-q_d[1])+vel_x[1]*q_d[1];
           vel_xy[1]=vel_x[3]*(1-q_d[1])+vel_x[2]*q_d[1];

           return vel_xy[0]*(1-q_d[2])+vel_xy[1]*q_d[2];
        }


};


// Wind of a sequence of frames of a turbolent flow
class Wind3d_turbo : public Wind3d_turboframe {
    private:
        float vt_grid[n_frames][n_grid_points][3];
        //float*** vt_grid;


    public:
        //virtual double* init(double x, double y, double z);
        //virtual double* velocity(double x, double y, double z, double t);

        Wind3d_turbo() {

            /*vt_grid=new float** [n_frames];
            for(int i=0; i<n_frames; i++){
              vt_grid[i]=new float* [n_grid_points];
            }

            for(int i=0; i<n_frames; i++){
              for(int j=0; j<n_grid_points; j++){
                vt_grid[i][j]=new float [3];
              }
            }*/


            std::string v_dir, v_name, q_path;
            int start_frame;
            try {
                v_dir = "./env/v1/";
                v_name = "velocities";
                start_frame = 1000;
                q_path = "./env/q.txt";
                wind_amplif = 1;
            } catch (std::exception)
            { throw std::runtime_error( "Invalid parameters of turbulent wind" ); }

            read_grid_file(q_path, q_grid);
            read_grid_files(v_dir, v_name, start_frame);
        }
        ~Wind3d_turbo(){

         /* for(int i=0; i<n_frames; i++){
            for(int j=0; j<n_grid_points; j++){
              delete[] vt_grid[i][j];
            }
          }
          for(int i=0; i<n_frames; i++){
            delete[] vt_grid[i];
          }
          delete[] vt_grid;*/
        }


        void read_grid_files(std::string dir, std::string name, int start_frame){

            Perc perc(10, n_frames);
            std::cout << "Reading the velocities..";
            #ifdef PARALLEL
            #pragma omp parallel for
            #endif
            for (int t=0; t<n_frames; t++) {
              #ifdef PARALLEL
            	std::cout<<"Process "<<omp_get_thread_num()<<" working on "<<t<<std::endl;
              #endif
            	//std::cout<<"Working on "<<t<<std::endl;
                //perc.step(t);
                std::string path = dir + name + std::to_string(t+start_frame) + ".txt";
                std::ifstream file (path);
                if (!file.is_open())
                    throw std::runtime_error("Error in opening the wind file at "+path);

                std::string line;
                vecd l = vecd(3);
                int count = 0;
                while ( getline (file, line) ){
                   l = str2vecd(line, " ", false);
                   vt_grid[t][count][0] = 1*l[0];
                   vt_grid[t][count][1] = 1*l[2];
                   vt_grid[t][count][2] = 1*l[1];
                   count++;
                }
                file.close();
            }
            std::cout << "\n";
        }


        virtual double* compute_velocity(double x, double y, double z, int frame) {

            int ind=n_z*n_x_axis_points*n_y_axis_points+n_x*n_y_axis_points+n_y;
            //std::cout << x << " " << y << " "<< z << " " << n_x << " " << n_y << " "<< n_z << "\n";

            double q_d[3];
            q_d[0]=(x-q_grid[ind][0])/(q_grid[ind+n_y_axis_points][0]-q_grid[ind][0]);
            q_d[1]=(y-q_grid[ind][1])/(q_grid[ind+1][1]-q_grid[ind][1]);
            q_d[2]=(z-q_grid[ind][2])/(q_grid[ind+n_x_axis_points*n_y_axis_points][2]-q_grid[ind][2]);
            //std::cout << q_d[0] << " " << q_d[1] << " "<< q_d[2] << "\n";

            double vel_corner[8];
            for (size_t i=0; i<3; i++){
                vel_corner[0] = vt_grid[frame][ind][i];
                vel_corner[1] = vt_grid[frame][ind+1][i];
                vel_corner[2] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+1][i];
                vel_corner[3] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points][i];
                vel_corner[4] = vt_grid[frame][ind+n_y_axis_points][i];
                vel_corner[5] = vt_grid[frame][ind+n_y_axis_points+1][i];
                vel_corner[6] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+n_y_axis_points+1][i];
                vel_corner[7] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+n_y_axis_points][i];
                //std::cout << vel_corner[0] << " " << vel_corner[1] << " "<< vel_corner[2] << " "<< vel_corner[3] << " ";
                //std::cout << vel_corner[4] << " " << vel_corner[5] << " "<< vel_corner[6] << " "<< vel_corner[7] << "\n";
                m_vel[i] = interpolation(q_d, vel_corner)*wind_amplif;
                //std::cout << m_vel[i] << " ";
            }
            //std::cout << '\n';

            return m_vel;
        }



};


#endif
