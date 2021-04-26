#ifndef _vect
#define _vect
#include <iostream>
#include <math.h>

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

  double x() const{
    return theta;
  }
  double y() const{
    return phi;
  }
  double z() const{
    return r;
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

  vect operator -() const{
    return vect{-theta, -phi, -r};
  }

  vect operator -(const vect& v) const{
    return *this+(-v);
  }

  vect& operator -=(const vect& v){
    return (*this+=(-v));
  }

  vect cross(const vect& v) const{
    return vect{phi*v.r-r*v.phi, r*v.theta-theta*v.r, theta*v.phi-phi*v.theta};
  }

  double dot(const vect& v) const{
    return theta*v.theta+phi*v.phi+r*v.r;
  }

  double norm() const{
    return sqrt(pow(theta, 2)+pow(phi, 2)+pow(r, 2));
  }

  bool operator ==(const vect& v) const{
    return (theta==v.theta && phi==v.phi && r==v.r);
  }

  bool operator !=(const vect& v) const{
    return !(*this==v);
  }

  vect operator *(const double s) const{
    return vect{theta*s, phi*s, r*s};
  }

  vect operator /(const double s) const{
    return (*this)*(1/s);
  }

  vect& operator *=(const double s){
    theta*=s;
    phi*=s;
    r*=s;
    return *this;
  }

  vect& operator /=(const double s){
    return (*this)*=(1/s);
  }
  vect tocartesian(const vect& pos) const{
    return vect{
      cos(pos.theta)*cos(pos.phi)*theta-sin(pos.phi)*phi+sin(pos.theta)*cos(pos.phi)*r,
      cos(pos.theta)*sin(pos.phi)*theta+cos(pos.phi)*phi+sin(pos.theta)*sin(pos.phi)*r,
      -sin(pos.theta)*theta+cos(pos.theta)*r
    };
  }
};


vect operator *(const double s, const vect& v){
  return v*s;
}

std::ostream& operator<<(std::ostream& out, const vect& v){
  return out<<v.theta<<" "<<v.phi<<" "<<v.r;
}

#endif
