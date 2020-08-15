// Jordan Jones
// Computer Graphics
// Fall 2019

#include "stdio.h"
#include "math.h"

// Constant memory declarations
__constant__ float Camera[3];
__constant__ float Light[3];
__constant__ float ImagePlaneBL[3];
__constant__ float ImagePlaneTR[3];
__constant__ int XRes;
__constant__ int YRes;

// Device memory declarations (geometry attributes)
float *d_radii;
float *d_a; // Point coordinates stored in row-major order (x, y, z, x, y, z, etc.)
float *d_b;
float *d_c;
int *d_reflective;
unsigned char *d_color;

enum reflectivity { NO_REFL, REFL };

typedef struct {
  int n;                  // Number of objects
  float *radii;           // Radii of objects (-1 if triangles)
  float *a;               // Point A (center if sphere)
  float *b;               // Point B
  float *c;               // Point C
  int *reflective;        // Material reflectivity (boolean)
  unsigned char *color;   // Material color
} GeometryList;

// Initializes GeometryList struct and returns pointer
GeometryList* initGeometryList( ) {
  GeometryList *list = (GeometryList *) malloc(sizeof(GeometryList));
  list -> n = 0;
  list -> radii = (float *) malloc(sizeof(float) * 0);
  list -> a = (float *) malloc(sizeof(float) * 0);
  list -> b = (float *) malloc(sizeof(float) * 0);
  list -> c = (float *) malloc(sizeof(float) * 0);
  list -> reflective = (int *) malloc(sizeof(int) * 0);
  list -> color = (unsigned char *) malloc(sizeof(unsigned char) * 0);
  return list;
}

// Adds a sphere with given traits to the GeometryList specified
__host__ void addSphere(GeometryList *list, float radius, float center[3], int refl, unsigned char col[3]) {
  list -> n += 1;
  
  list -> radii = (float *) realloc(list->radii, (list->n) * sizeof(float));
  list -> radii[list->n-1] = radius;
  
  list -> a = (float *) realloc(list->a, (list->n) * sizeof(float) * 3);
  list -> a[3 * (list->n-1) + 0] = center[0];
  list -> a[3 * (list->n-1) + 1] = center[1];
  list -> a[3 * (list->n-1) + 2] = center[2];
  
  list -> b = (float *) realloc(list->b, (list->n) * sizeof(float) * 3);
  
  list -> c = (float *) realloc(list->c, (list->n) * sizeof(float) * 3);
  
  list -> reflective = (int *) realloc(list->reflective, (list->n) * sizeof(int));
  list -> reflective[list->n-1] = refl;
  
  list -> color = (unsigned char *) realloc(list->color, (list->n) * sizeof(unsigned char) * 3);
  list -> color[3 * (list->n-1) + 0] = col[0];
  list -> color[3 * (list->n-1) + 1] = col[1];
  list -> color[3 * (list->n-1) + 2] = col[2];
}

// Adds a triangle with given traits to the GeometryList specified
__host__ void addTriangle(GeometryList *list, float a[3], float b[3], float c[3], int refl, unsigned char col[3]) {
  list -> n += 1;
  
  list -> radii = (float *) realloc(list->radii, (list->n) * sizeof(float));
  list -> radii[list->n-1] = -1; // Radius of -1 indicates a triangle
  
  list -> a = (float *) realloc(list->a, (list->n) * sizeof(float) * 3);
  list -> a[3 * (list->n-1) + 0] = a[0];
  list -> a[3 * (list->n-1) + 1] = a[1];
  list -> a[3 * (list->n-1) + 2] = a[2];
  
  list -> b = (float *) realloc(list->b, (list->n) * sizeof(float) * 3);
  list -> b[3 * (list->n-1) + 0] = b[0];
  list -> b[3 * (list->n-1) + 1] = b[1];
  list -> b[3 * (list->n-1) + 2] = b[2];
  
  list -> c = (float *) realloc(list->c, (list->n) * sizeof(float) * 3);
  list -> c[3 * (list->n-1) + 0] = c[0];
  list -> c[3 * (list->n-1) + 1] = c[1];
  list -> c[3 * (list->n-1) + 2] = c[2];
  
  list -> reflective = (int *) realloc(list->reflective, (list->n) * sizeof(int));
  list -> reflective[list->n-1] = refl;
  
  list -> color = (unsigned char *) realloc(list->color, (list->n) * sizeof(unsigned char) * 3);
  list -> color[3 * (list->n-1) + 0] = col[0];
  list -> color[3 * (list->n-1) + 1] = col[1];
  list -> color[3 * (list->n-1) + 2] = col[2];
}

// Prints the information of the geometry at the given index
__host__ void printGeometry(GeometryList *list, int index) {
  if (index >= list -> n) {
    printf("No geometry at index %d\n", index);
    return;
  }
  
  printf("Geometry at index %d:\n", index);
  printf("\tRadius: %f\n", list->radii[index]);
  printf("\tPoint 1: %f, %f, %f\n", list->a[3*index+0], list->a[3*index+1], list->a[3*index+2]);
  printf("\tPoint 2: %f, %f, %f\n", list->b[3*index+0], list->b[3*index+1], list->b[3*index+2]);
  printf("\tPoint 3: %f, %f, %f\n", list->c[3*index+0], list->c[3*index+1], list->c[3*index+2]);
  printf("\tReflectivity: %d\n", list->reflective[index]);
  printf("\tColor: %d, %d, %d\n", list->color[3*index+0], list->color[3*index+1], list->color[3*index+2]);
  printf("\n");
}

__device__ float d_norm(float *v) {
  return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

__device__ float d_distance(float *p, float *q) {
  return sqrtf(p[0]*q[0] + p[1]*q[1] + p[2]*q[2]);
}

// Returns a normalized vector
__device__ void d_normalize(float *result, float *v) {
  float n = d_norm(v);
  result[0] = v[0] / n;
  result[1] = v[1] / n;
  result[2] = v[2] / n;
}

// Returns scalar result of dot product of input vectors
__device__ float d_dotProduct(float *v, float *w) {
  return v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
}

// Returns a vector of the cross product of input vectors
__device__ void d_crossProduct(float *result, float *v, float *w) {
  result[0] = v[1]*w[2] - v[2]*w[1];
  result[1] = v[2]*w[0] - v[0]*w[2];
  result[2] = v[0]*w[1] - v[1]*w[0];
}

// Returns a vector describing the difference between input points
__device__ void d_pointDifference(float *result, float *p, float *q) {
  result[0] = p[0] - q[0];
  result[1] = p[1] - q[1];
  result[2] = p[2] - q[2];
}

// Returns a vector orthogonal to a sphere, relative to a given point
__device__ void d_sphereNormal(float *result, float *center, float *p) {
  d_pointDifference(result, p, center);
  d_normalize(result, result);
}

// Returns a vector orthogonal to a triangle's surface
__device__ void d_triangleNormal(float *result, float *a, float *b, float *c) {
  float v[3];
  float w[3];
  d_pointDifference(v, a, b);
  d_pointDifference(w, a, c);
  d_crossProduct(result, v, w);
  d_normalize(result, result);
}

// Returns a point equal to the point given plus the given vector
__device__ void d_pointPlusVector(float *result, float *p, float *v) {
  result[0] = p[0] + v[0];
  result[1] = p[1] + v[1];
  result[2] = p[2] + v[2];
}

// Returns a vector multiplied by scalar f
__device__ void d_scaleVector(float *result, float *v, float f) {
  result[0] = v[0] * f;
  result[1] = v[1] * f;
  result[2] = v[2] * f;
}

// Returns the sum of two vectors
__device__ void d_vectorAdd(float *result, float *v, float *w) {
  result[0] = v[0] + w[0];
  result[1] = v[1] + w[1];
  result[2] = v[2] + w[2];
}

// Returns a float describing the distance to an intersection with the given geometry
//   The point of intersection can be calculated by adding the (rayDir * result) + rayPos
//   The normal vector of intersection can be calculated by using a d_*normal() function with the hit point given (if a sphere)
__device__ float d_intersect(float *rayPos, float *rayDir, int geomIndex, float *g_radii, float *g_a, float *g_b, float *g_c) {
  
  if (g_radii[geomIndex] == -1) { // If triangle
  
    float a = g_a[3*geomIndex + 0] - g_b[3*geomIndex + 0];
    float b = g_a[3*geomIndex + 1] - g_b[3*geomIndex + 1];
    float c = g_a[3*geomIndex + 2] - g_b[3*geomIndex + 2];
    
    float d = g_a[3*geomIndex + 0] - g_c[3*geomIndex + 0];
    float e = g_a[3*geomIndex + 1] - g_c[3*geomIndex + 1];
    float f = g_a[3*geomIndex + 2] - g_c[3*geomIndex + 2];
    
    float g = rayDir[0];
    float h = rayDir[1];
    float i = rayDir[2];
    
    float j = g_a[3*geomIndex + 0] - rayPos[0];
    float k = g_a[3*geomIndex + 1] - rayPos[1];
    float l = g_a[3*geomIndex + 2] - rayPos[2];
    
    float m = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);
    
    // Result variables
    float beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/m;
    float gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/m;
    float t = -(f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c)) / m;
    
    // Check for hit within triangle
    if (t < 0) {
      return INFINITY;
    } else if (gamma < 0 || gamma > 1) {
      return INFINITY;
    } else if (beta < 0 || beta > 1-gamma) {
      return INFINITY;
    } else {
      return t;
    }
    
    
  } else {                        // Else, if sphere
    
    float a = d_dotProduct(rayDir, rayDir);
    
    float v[3];
    float w[3];
    d_scaleVector(v, rayDir, 2);
    d_pointDifference(w, rayPos, &g_a[3*geomIndex]);
    float b = d_dotProduct(v, w);
    
    float c = d_dotProduct(w, w) - (g_radii[geomIndex] * g_radii[geomIndex]);
    
    float t1 = (-b + sqrtf(b*b - 4*a*c))/(2*a);
    float t2 = (-b - sqrtf(b*b - 4*a*c))/(2*a);
    
    if (t1 > 0 && t1 < t2) {
      return t1;
    } else if (t2 > 0 && t2 < t1) {
      return t2;
    } else {
      return INFINITY;
    }

  }
}

// Returns a new ray reflected from the hit of the input ray
//   Will set result Pos/Dir to reflected ray
//   Requires distance to hit and normal vector of hit object
// Always increment numReflections after calling this function!
__device__ void d_reflect(float *resultPos, float *resultDir, float *rayPos, float *rayDir, float *hit, float *vector) {
  float v[3];
  float w[3];
  d_scaleVector(v, vector, -2*d_dotProduct(rayDir, vector));
  d_vectorAdd(w, rayDir, v);
  d_normalize(resultDir, w);
  
  // Bump result ray to avoid reflection static
  d_scaleVector(v, resultDir, 0.001);
  d_pointPlusVector(resultPos, hit, v);
}