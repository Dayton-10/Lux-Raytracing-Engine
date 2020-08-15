// Jordan Jones
// Computer Graphics
// Fall 2019

#include "stdio.h"
#include "math.h"

typedef struct Point {
	float x, y, z;
} Point;

typedef struct Vector {
	float x, y, z;
} Vector;

typedef struct Ray {
  Point point;
  Vector vector;
  int numReflections;
} Ray;

typedef struct Material {
  unsigned char r; // Red value
  unsigned char g; // Blue value
  unsigned char b; // Green value
  int reflective;  // Boolean for reflectivity
} Material;

typedef struct Sphere {
  Point center;       // Center of sphere
  float radius;       // Radius of sphere
  Material material;  // Color & reflectivity of sphere
} Sphere;

typedef struct Triangle {
  Point a;
  Point b;
  Point c;
  Material material;
} Triangle;

typedef struct RayHit {
  float t;           // Distance to hit
  Material material; // Material of object hit
  Point p;           // Point at hit
  Vector v;          // Vector normal to hit location
} RayHit;

// Returns norm of input vector
float norm(Vector v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// Returns distance between two points
float distance(Point p, Point q) {
  return sqrtf(p.x*q.x + p.y*q.y + p.z*q.z);
}

// Returns normalized input vector from vector v
Vector normalize(Vector v) {
  float n = norm(v);
  Vector result;
  result.x = v.x / n;
  result.y = v.y / n;
  result.z = v.z / n;
  return result;
}

// Returns scalar dot Product of vectors v and w
float dotProduct(Vector v, Vector w) {
  return v.x*w.x + v.y*w.y + v.z*w.z;
}

// Returns a vector of the corss product of vectors v and w
Vector crossProduct(Vector v, Vector w) {
  Vector result;
  result.x = v.y*w.z - v.z*w.y;
  result.y = v.z*w.x - v.x*w.z;
  result.z = v.x*w.y - v.y*w.x;
  return result;
}

// Returns a vector from point q to point p
Vector pointDifference(Point p, Point q) {
  Vector result;
  result.x = p.x - q.x;
  result.y = p.y - q.y;
  result.z = p.z - q.z;
  return result;
}

// Returns a vector normal to a sphere, relative to a point of intersection
Vector sphereNormal(Sphere s, Point p) {
  Vector v;
  v = pointDifference(p, s.center);
  return normalize(v);
}

// Returns a vector normal to a triangle's surface
Vector triangleNormal(Triangle t) {
  Vector a = pointDifference(t.a, t.b);
  Vector b = pointDifference(t.a, t.c);
  Vector v;
  v = crossProduct(a, b);
  return normalize(v);
}

// Returns a point equal to the point input plus the vector
Point pointPlusVector(Point p, Vector v) {
  Point result;
  result.x = p.x + v.x;
  result.y = p.y + v.y;
  result.z = p.z + v.z;
  return result;
}

// Returns a vector multiplied by scalar f
Vector scaleVector(Vector v, float f) {
  Vector result;
  result.x = v.x * f;
  result.y = v.y * f;
  result.z = v.z * f;
  return result;
}

// Returns the sum of two vectors
Vector vectorAdd(Vector v, Vector w) {
  Vector result;
  result.x = v.x + w.x;
  result.y = v.y + w.y;
  result.z = v.z + w.z;
  return result;
}

// Returns a RayHit struct describing the intersection of a Ray and Sphere
RayHit sphereIntersect(Ray ray, Sphere sphere) {
  // Calculate quadratic solutions for t
  float a = dotProduct(ray.vector, ray.vector);
  float b = dotProduct(scaleVector(ray.vector, 2), pointDifference(ray.point, sphere.center));
  float c = dotProduct(pointDifference(ray.point, sphere.center), pointDifference(ray.point, sphere.center)) - (sphere.radius * sphere.radius);
  float t1 = (-b + sqrtf(b*b - 4*a*c))/(2*a);
  float t2 = (-b - sqrtf(b*b - 4*a*c))/(2*a);
  
  RayHit hit;
  
  // Return smallest t value
  if (t1>0 && t1<t2) {
    hit.t = t1;
  } else {
    hit.t = t2;
  }
  
  hit.material = sphere.material;
  hit.p = pointPlusVector(ray.point, scaleVector(ray.vector, hit.t));
  hit.v = sphereNormal(sphere, hit.p);
  
  return hit;
}

// Returns a RayHit struct describing the intersection of a Ray and Triangle
RayHit triangleIntersect(Ray ray, Triangle triangle) {
  RayHit hit;
  
  // Helper variables
  float a = triangle.a.x - triangle.b.x;
  float b = triangle.a.y - triangle.b.y;
  float c = triangle.a.z - triangle.b.z;
  float d = triangle.a.x - triangle.c.x;
  float e = triangle.a.y - triangle.c.y;
  float f = triangle.a.z - triangle.c.z;
  float g = ray.vector.x;
  float h = ray.vector.y;
  float i = ray.vector.z;
  float j = triangle.a.x - ray.point.x;
  float k = triangle.a.y - ray.point.y;
  float l = triangle.a.z - ray.point.z;
  float m = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);
  
  // Result variables
  float beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/m;
  float gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/m;
  float t = -(f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c)) / m;
  
  // Check for hit within triangle
  if (t<0) {
    hit.t = INFINITY;
    return hit;
  } else if (gamma < 0 || gamma > 1) {
    hit.t = INFINITY;
    return hit;
  } else if (beta < 0 || beta > 1-gamma) {
    hit.t = INFINITY;
    return hit;
  }
  
  hit.t = t;
  hit.material = triangle.material;
  hit.p = pointPlusVector(ray.point, scaleVector(ray.vector, hit.t));
  hit.v = triangleNormal(triangle);
  
  return hit;
}

// Returns a new ray reflected from the hit of the input ray
Ray reflect(Ray ray, RayHit hit) {
  Ray result;
  
  result.point = hit.p;
  result.vector = normalize(vectorAdd(ray.vector, scaleVector(hit.v, -2*dotProduct(ray.vector, hit.v))));
  result.numReflections = ray.numReflections + 1;
  
  result.point = pointPlusVector(result.point, scaleVector(result.vector, 0.001));
  
  return result;
}