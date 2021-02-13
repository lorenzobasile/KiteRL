#ifndef _vect
#define _vect
#include <iostream>

struct vect{
  double theta;
  double phi;
  double r;

  vect()=default;

  vect(double c1, double c2, double c3): theta{c1}, phi{c2}, r{c3} {}

  vect(const vect& v)=default;

  vect& operator =(const vect& v){
    if(v!=*this){
      theta=v.theta;
      phi=v.phi;
      r=v.r;
    }
    return *this;
  }

  vect operator +(const vect& v) const{
    return vect{theta+v.theta, phi+v.phi, r+v.r};
  }

  vect& operator +=(const vect& v){
    theta+=v.theta;
    phi+=v.phi;
    r+=v.r;
    return *this;
  }

  vect cross(const vect& v) const{
    return vect{phi*v.r-r*v.phi, r*v.theta-theta*v.r, theta*v.phi-phi*v.theta};
  }

  double dot(const vect& v) const{
    return theta*v.theta+phi*v.phi+r*v.r;
  }

  bool operator==(const vect& v) const{
    return (theta==v.theta && phi==v.phi && r==v.r);
  }

  bool operator!=(const vect& v) const{
    return !(*this==v);
  }

};

std::ostream& operator<<(std::ostream& out, const vect& v){
  return out<<v.theta<<", "<<v.phi<<", "<<v.r;
}

#endif
