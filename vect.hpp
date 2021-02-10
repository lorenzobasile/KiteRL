#ifndef _vect
#define _vect

struct vect{
  double theta;
  double phi;
  double r;

  vect()=default;

  vect(double c1, double c2, double c3): theta{c1}, phi{c2}, r{c3} {}
  
};
#endif
