#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
//#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "math.h"


using vecs = std::vector<std::string>;
using vecf = std::vector<float>;
using veci = std::vector<int>;
using vec2i = std::vector<veci>;
using vecd = std::vector<double>;
using vec2d = std::vector<vecd>;
using vec3d = std::vector<vec2d>;
using dictd = std::map<std::string, double>;
using dictvecd = std::map<std::string, vecd>;
using dicts = std::map<std::string, std::string>;
using d_i_fnc = std::function<double(int)>;


/* Parameters. They can be doubles, vector of doubles or strings */
struct param {
	dictd d;
	dictvecd vecd;
	dicts s;
};


// BASIC GLOBAL CONSTANTS
const double PI = 3.1415926535897932384626433;
/* Air density, kg/m^3 */
//const double rho = 1.225;
/* Gravity acceleration, m/s^2 */
//const double g = 9.806;


void par2pol_boltzmann(const vecd& params, vecd& policy);

void pol2par_boltzmann(const vecd& policy, vecd& params);

double plaw_decay(double t, double t_burn, double expn, double a0, double ac);

double pmix_decay(double t, double t_burn0, double t_burn1, double expn, double a0, double a1, double ac);

double p2steps_decay(double t, double t_burn, double a0, double a1);

double plaw_statedecay(int n, double n0, double expn, double a0);

vecd str2vecd(std::string line, std::string separator, bool sep_at_end);

param parse_param_file(std::string file_path);

vecd read_value(std::string file_path);

vec2d read_quality(std::string file_path);

vec2i read_occ(std::string file_path);

vec2d read_policy(std::string file_path);

vecd parse_str_of_doubles(std::string str);


/* Class for measuring the time between the reset and en enlapsed call */
class Timer {
	private:
		using clock_t = std::chrono::high_resolution_clock;
		using second_t = std::chrono::duration<double, std::ratio<1> >;
		std::chrono::time_point<clock_t> m_beg;
	public:
		Timer() : m_beg(clock_t::now()) { };
		/* Set the onset for the time measure */
		void reset() { m_beg = clock_t::now(); }
		/* Get the time in seconds enlapsed from reset */
		double elapsed() const { return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count(); }
};


class Perc {
	private:
		int m_perc_step;
		int m_max_steps;
		double m_last_perc;
	public:
		Perc(int perc_step, int max_steps) : m_max_steps{max_steps}, m_last_perc{0.0} {
			if (perc_step<1 || perc_step>100) {
				std::cout << "Invalid percentage step. Set by default to 10%\n";
				m_perc_step = 10;
			}
			else m_perc_step = perc_step;
		};
		bool step(int curr_step) {
			double perc = (double)curr_step/(double)m_max_steps*100;
			if (perc >= m_last_perc){
				std::cout << "\n" << round(perc) << "%";
				m_last_perc = round(perc) + m_perc_step;
				return true;
			}
			return false;
		}
};

#endif
