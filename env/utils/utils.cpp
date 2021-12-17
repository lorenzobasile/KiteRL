#include "utils.h"
#include <cctype>


void par2pol_boltzmann(const vecd& params, vecd& policy){
    double max = *std::max_element(params.begin(), params.end());
    double norm = 0;
    for (int i=0; i<params.size(); i++){
        double val = exp(params[i]-max);
        policy[i] = val;
        norm += val;
    }
    for (int i=0; i<policy.size(); i++) policy[i] /= norm;
}


void pol2par_boltzmann(const vecd& policy, vecd& params){
    for (int i=0; i<params.size(); i++){
        double val = 0;
        if (policy[i] > 10E-20)
            params[i] = log(policy[i]);
        else
            params[i] = -20;
    }
}


double plaw_decay(double t, double t_burn, double expn, double a0, double ac){
    if (t < t_burn)
        return a0;
    else {
        return a0 * ac / (ac + pow(t-t_burn, expn));
    }
}

double pmix_decay(double t, double t_burn0, double t_burn1, double expn, double a0, double a1, double ac){
    if (t < t_burn0)
        return a0;
    else {
      if ((t > t_burn0) && (t < t_burn1))
          return a0 - ((a0 - a1)/(t_burn1 - t_burn0))*(t-t_burn0);
      else
        return a1 * ac / (ac + pow(t-t_burn1, expn));
    }
}

double p2steps_decay(double t, double t_burn, double a0, double a1){
    if (t < t_burn)
        return a0;
    else
        return a1;
}

double plaw_statedecay(int n, double n0, double expn, double a0){
    return a0 / (1 + pow(n/n0, expn));
}


vecd str2vecd(std::string line, std::string separator, bool sep_at_end) {
    std::size_t sep_pos = line.find(separator);
    if (sep_pos == std::string::npos) {
        if (sep_at_end)
            throw std::runtime_error(separator + " separator not found in " + line);
        else {
            vecd v = {std::stod(line.substr(0, sep_pos))};
            return v;
        }
    }

    std::string elem = line.substr(0, sep_pos);
    vecd v = vecd(0);
    try {
        v.push_back(std::stod(elem));
    }
    catch (std::exception& e){
        v.push_back(0);
    }

    while (true){
        std::size_t next_sep_pos = line.find(separator, sep_pos+1);
        if (sep_at_end && next_sep_pos == std::string::npos) break;
        std::string elem = line.substr(sep_pos+1, next_sep_pos-sep_pos);
        try{
            v.push_back(std::stod(elem));
        }
        catch (std::exception& e){
            v.push_back(0);
        }
        if (!sep_at_end && next_sep_pos == std::string::npos) break;
        sep_pos = next_sep_pos;
    }

    return v;
}


param parse_param_file(std::string file_path){

    dictd paramd;
    dictvecd paramvecd;
    dicts params;

    std::ifstream param_file (file_path);
    if (!param_file.is_open())
        throw std::runtime_error("Error in opening the parameter file at "+file_path);

    std::string line;
    while ( getline (param_file, line) ) {
        std::size_t tab_pos = line.find("\t");
        std::string key = line.substr(0,tab_pos);
        std::string value = line.substr(tab_pos+1, std::string::npos);

        std::size_t comma_pos = value.find(",");
        if (value.find(",") != std::string::npos){
            paramvecd[key] = str2vecd(value, ",", true); // Parse a vector
        }
        else{
            try {
                double vald = std::stod(value); // Parse a double
                paramd[key] = vald;
            } catch (std::invalid_argument){
                if (isspace(value[value.size()-1]))
                    value = value.substr(0,value.size()-1);
                params[key] = value; // Parse a string if stod gives exception
            }
        }
    }
    param_file.close();

    if (paramd.size() == 0 && paramvecd.size() == 0 && params.size() == 0)
        throw std::runtime_error("Empty parameter file");

    return param{paramd, paramvecd, params};
}


vec2d read_policy(std::string file_path) {

    vec2d policy(0);
    std::ifstream pol_file (file_path);
    if (!pol_file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    std::string line;
    while ( getline (pol_file, line) )
    {
        vecd pol_at_state = str2vecd(line, " ", false);
        double cum_p=0;
        for (const double p : pol_at_state) cum_p+=p;
        if (cum_p > 1+10E-6)
            throw std::runtime_error("Not normalized policy at "+file_path+" "+std::to_string(cum_p));
        pol_at_state.push_back(1-cum_p);
        policy.push_back(pol_at_state);
    }
    pol_file.close();

    return policy;
}


vec2d read_quality(std::string file_path) {

    vec2d quality(0);
    std::ifstream pol_file (file_path);
    if (!pol_file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    std::string line;
    while ( getline (pol_file, line) )
        quality.push_back(str2vecd(line, ",", false));
    pol_file.close();

    return quality;
}

/*vec2i read_occ(std::string file_path) {

    vec2i occ(0);
    std::ifstream occ_file (file_path);
    if (!occ_file.is_open())
        throw std::runtime_error("Error in opening the occ_matrix file at "+file_path);

    std::string line;
    while ( getline (occ_file, line) )
        occ.push_back(int(str2vecd(line, ",", false)));
    occ_file.close();

    return occ;
}*/


vecd read_value(std::string file_path) {

    vecd val(0);
    std::ifstream val_file (file_path);
    if (!val_file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    std::string line;
    while ( getline (val_file, line) )
        val.push_back(std::stod(line));
    val_file.close();

    return val;
}
