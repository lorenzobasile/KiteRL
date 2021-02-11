#ifndef _vect
#define _vect
#include <iostream>

struct vect{
  double theta;
  double phi;
  double r;

  vect()=default;

  vect(double c1, double c2, double c3): theta{c1}, phi{c2}, r{c3} {}

  vect operator +(const vect& v) const {
    return vect{theta+v.theta, phi+v.phi, r+v.r};
  }

  vect& operator +=(const vect& v){
    theta+=v.theta;
    phi+=v.phi;
    r+=v.r;
    return *this;
  }

  bool operator==(const vect& v){
    return (theta==v.theta && phi==v.phi && r==v.r);
  }

  bool operator!=(const vect& v){
    return !(*this==v);
  }

  friend std::ostream& operator<<(std::ostream& out, const vect& v){
    return out<<v.theta<<", "<<v.phi<<", "<<v.r;
  }

};
#endif
