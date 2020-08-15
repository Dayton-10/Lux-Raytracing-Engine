// Jordan Jones
// Computer Graphics
// Fall 2019

#include <stdio.h>
#include <math.h>

enum Reflectivity {NO_REFL, REFL};
enum ShapeType {SPHERE, TRIANGLE};

typedef float Point[3];
typedef float Vector[3];
typedef unsigned char Color[3];
typedef float Ray[6]; // Point, Vector

// Constant memory declarations
__constant__ Point Camera;
__constant__ Point Light;
__constant__ Point ImagePlaneBL;
__constant__ Point ImagePlaneTR;
__constant__ int XRes;
__constant__ int YRes;

// Device memory declarations (geometry attributes)
// Point coordinates stored in row-major order (x, y, z, x, y, z, etc.)
int* d_shapeIndex;
ShapeType* d_shape;
Reflectivity* d_reflective;
unsigned char* d_color;
float* d_shapeInfo;

typedef struct {
  int nShapes; // Total number of shapes stored
  int* shapeIndex;
  ShapeType* shape;
  Reflectivity* reflective;
  unsigned char* color;
  
  int nInfo;   // Total length of shapeInfo array (units of floats)
  float* shapeInfo;
} GeometryList;

// Checks provided GeometryList for Null pointers
void checkGeometryList(GeometryList* list) {
  if (list == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList\n");
    exit(-1);
  } else if (list->shapeIndex == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList shapeIndex array\n");
    exit(-1);
  } else if (list->shape == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList shape array\n");
    exit(-1);
  } else if (list->shapeInfo == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList shapeInfo array\n");
    exit(-1);
  } else if (list->reflective == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList reflective array\n");
    exit(-1);
  } else if (list->color == NULL) {
    printf("ERROR: Unable to allocate memory for GeometryList color array\n");
    exit(-1);
  }
}

// Initializes GeometryList struct and returns pointer
GeometryList* initGeometryList( ) {
  GeometryList* list = (GeometryList*) malloc(sizeof(GeometryList));
  list -> nShapes = 0;
  list -> shapeIndex = (int*) malloc(sizeof(int) * 0);
  list -> shape = (ShapeType*) malloc(sizeof(ShapeType) * 0);
  list -> reflective = (Reflectivity*) malloc(sizeof(int) * 0);
  list -> color = (unsigned char*) malloc(sizeof(Color) * 0);
  
  list -> nInfo = 0;
  list -> shapeInfo = (float*) malloc(sizeof(float) * 0);
  
  checkGeometryList(list);
  return list;
}

// Adds a sphere with provided traits to the GeometryList specified
void addSphere(GeometryList* list, Point center, float radius, Color col, Reflectivity refl) {
  
  list -> nShapes += 1;
  list -> shapeIndex = (int*)           realloc(list->shapeIndex, sizeof(int)          * (list->nShapes));
  list -> shape      = (ShapeType*)     realloc(list->shape,      sizeof(ShapeType)    * (list->nShapes));
  list -> reflective = (Reflectivity*)  realloc(list->reflective, sizeof(Reflectivity) * (list->nShapes));
  list -> color      = (unsigned char*) realloc(list->color,      sizeof(Color)        * (list->nShapes));
  
  list -> shapeInfo = (float*) realloc(list->shapeInfo, sizeof(float) * (list->nInfo + 4)); // 4 floats for a Sphere (Point, radius)
  
  checkGeometryList(list);
  
  // Add new info
  list -> shapeIndex[list->nShapes-1] = list->nInfo;
  list -> shape     [list->nShapes-1] = SPHERE;
  list -> reflective[list->nShapes-1] = refl;
  
  list -> color     [3*(list->nShapes-1) + 0] = col[0];
  list -> color     [3*(list->nShapes-1) + 1] = col[1];
  list -> color     [3*(list->nShapes-1) + 2] = col[2];
  
  list -> shapeInfo[list->nInfo+0] = center[0];
  list -> shapeInfo[list->nInfo+1] = center[1];
  list -> shapeInfo[list->nInfo+2] = center[2];
  list -> shapeInfo[list->nInfo+3] = radius;
  
  list -> nInfo += 4;
}

// Adds a triangle with given traits to the GeometryList specified
void addTriangle(GeometryList* list, Point a, Point b, Point c, Color col, Reflectivity refl) {
  list -> nShapes += 1;
  list -> shapeIndex = (int*)           realloc(list->shapeIndex, sizeof(int)          * (list->nShapes));
  list -> shape      = (ShapeType*)     realloc(list->shape,      sizeof(ShapeType)    * (list->nShapes));
  list -> reflective = (Reflectivity*)  realloc(list->reflective, sizeof(Reflectivity) * (list->nShapes));
  list -> color      = (unsigned char*) realloc(list->color,      sizeof(Color)        * (list->nShapes));
  
  list -> shapeInfo = (float*) realloc(list->shapeInfo, sizeof(float) * (list->nInfo + 9)); // 9 floats for a triangle (3 Points)
  
  checkGeometryList(list);
  
  // Add new info
  list -> shapeIndex[list->nShapes-1] = list->nInfo;
  list -> shape     [list->nShapes-1] = TRIANGLE;
  list -> reflective[list->nShapes-1] = refl;
  
  list -> color[3*(list->nShapes-1) + 0] = col[0];
  list -> color[3*(list->nShapes-1) + 1] = col[1];
  list -> color[3*(list->nShapes-1) + 2] = col[2];
  
  list -> shapeInfo[list->nInfo+0] = a[0];
  list -> shapeInfo[list->nInfo+1] = a[1];
  list -> shapeInfo[list->nInfo+2] = a[2];
  
  list -> shapeInfo[list->nInfo+3] = b[0];
  list -> shapeInfo[list->nInfo+4] = b[1];
  list -> shapeInfo[list->nInfo+5] = b[2];
  
  list -> shapeInfo[list->nInfo+6] = c[0];
  list -> shapeInfo[list->nInfo+7] = c[1];
  list -> shapeInfo[list->nInfo+8] = c[2];
  
  list -> nInfo += 9;
}

// Removes the geometry at the given index from the GeometryList
void removeGeometry(GeometryList* list, int index) {
  if (index < 0 || index >= list->nShapes) {
    printf("WARNING: removeGeometry: invalid index %d\n", index);
    return;
  }
  
  // Copy data forwards in shapeInfo array
  switch(list->shape[index]) {
    
    case SPHERE: {
      for (int i=list->shapeIndex[index]; i<list->nInfo-4; i++) {
        list -> shapeInfo[i] = list -> shapeInfo[i+4];
      }
      for (int i=index; i<(list->nShapes); i++) {
        list -> shapeIndex[i] -= 4;
      }
      list->nInfo -= 4;
    }
    break;
    
    case TRIANGLE: {
      for (int i=list->shapeIndex[index]; i<list->nInfo-9; i++) {
        list -> shapeInfo[i] = list -> shapeInfo[i+9];
      }
      for (int i=index; i<(list->nShapes); i++) {
        list -> shapeIndex[i] -= 9;
      }
      list->nInfo -= 9;
    }
    break;
  }
  
  // Copy data forwards in other arrays
  for (int i=index; i<(list->nShapes)-1; i++) {
    list -> shapeIndex[i] = list -> shapeIndex[i+1];
    list -> shape[i]      = list -> shape[i+1];
    list -> reflective[i] = list -> reflective[i+1];
    
    list -> color[3 * i + 0] = list -> color[3 * (i+1) + 0];
    list -> color[3 * i + 1] = list -> color[3 * (i+1) + 1];
    list -> color[3 * i + 2] = list -> color[3 * (i+1) + 2];
  }
  list->nShapes -= 1;
  
  // Realloc arrays
  list -> shapeIndex = (int*)           realloc(list->shapeIndex, sizeof(int)          * (list->nShapes));
  list -> shape      = (ShapeType*)     realloc(list->shape,      sizeof(ShapeType)    * (list->nShapes));
  list -> reflective = (Reflectivity*)  realloc(list->reflective, sizeof(Reflectivity) * (list->nShapes));
  list -> color      = (unsigned char*) realloc(list->color,      sizeof(Color)        * (list->nShapes));
  
  list -> shapeInfo = (float*) realloc(list->shapeInfo, sizeof(float) * (list->nInfo));
}

// Prints the information of the requested geometry
void printGeometry(GeometryList* list, int index) {
  if (index < 0 || index > list->nShapes) {
    printf("Invalid index: %d\n", index);
    return;
  }
  
  float* data = &(list->shapeInfo[list->shapeIndex[index]]); // Get address of first float of shape
  
  switch(list->shape[index]) {
    case SPHERE: {
      printf("Geometry at index %d: Sphere\n", index);
      printf("Center at: %f, %f, %f\n", data[0], data[1], data[2]);
      printf("Radius of: %f\n", data[3]);
      printf("Color of: %d, %d, %d\n", list->color[3*index+0], list->color[3*index+1], list->color[3*index+2]);
      printf("Reflectivity of: %d\n", list->reflective[index]);
    }
    break;
    
    case TRIANGLE: {
      printf("Geometry at index %d: Triangle\n", index);
      printf("Point A at: %f, %f, %f\n", data[0], data[1], data[2]);
      printf("Point B at: %f, %f, %f\n", data[3], data[4], data[5]);
      printf("Point C at: %f, %f, %f\n", data[6], data[7], data[8]);
      printf("Color of: %d, %d, %d\n", list->color[3*index+0], list->color[3*index+1], list->color[3*index+2]);
      printf("Reflectivity of: %d\n", list->reflective[index]);
    }
  }
  printf("\n");
}

void printAllGeometry(GeometryList* list) {
  for (int i=0; i<list->nShapes; i++) {
    printGeometry(list, i);
  }
}

// Returns a null-terminated string concisely describing the selected Geometry
//   String must be freed after use
__host__ char* toStringGeometry(GeometryList* list, int index) {
  
  if (index < 0 || index > list->nShapes) {
    printf("Invalid index: %d\n", index);
    return NULL;
  }
  
  char* buffer = (char*) malloc(sizeof(char) * 100);
  float* data = &(list->shapeInfo[list->shapeIndex[index]]); // Get address of first float of shape
  
  switch(list->shape[index]) {
    
    case SPHERE:
      sprintf(buffer, "Sphere:  {%.2f, %.2f, %.2f} r=%f\0", data[0], data[1], data[2], data[3]);
    break;
    
    case TRIANGLE:
      sprintf(buffer, "Triangle: {%.2f, %.2f, %.2f} {%.2f, %.2f, %.2f} {%.2f, %.2f, %.2f}\0", data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    break;
  }
  
  return buffer;
}

__device__ float d_norm(Vector v) {
  return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

__device__ float d_distance(Point p, Point q) {
  return sqrtf(p[0]*q[0] + p[1]*q[1] + p[2]*q[2]);
}

// Returns a normalized vector
__device__ void d_normalize(Vector result, Vector v) {
  float n = d_norm(v);
  result[0] = v[0] / n;
  result[1] = v[1] / n;
  result[2] = v[2] / n;
}

// Returns scalar result of dot product of input vectors
__device__ float d_dotProduct(Vector v, Vector w) {
  return v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
}

// Returns a vector of the cross product of input vectors
__device__ void d_crossProduct(Vector result, Vector v, Vector w) {
  result[0] = v[1]*w[2] - v[2]*w[1];
  result[1] = v[2]*w[0] - v[0]*w[2];
  result[2] = v[0]*w[1] - v[1]*w[0];
}

// Returns a vector describing the difference between input points
__device__ void d_pointDifference(Vector result, Point p, Point q) {
  result[0] = p[0] - q[0];
  result[1] = p[1] - q[1];
  result[2] = p[2] - q[2];
}

// Returns a vector orthogonal to a sphere, relative to a given point
__device__ void d_sphereNormal(Vector result, Point center, Point p) {
  d_pointDifference(result, p, center);
  d_normalize(result, result);
}

// Returns a vector orthogonal to a triangle's surface
__device__ void d_triangleNormal(Vector result, Point a, Point b, Point c) {
  float v[3];
  float w[3];
  d_pointDifference(v, a, b);
  d_pointDifference(w, a, c);
  d_crossProduct(result, v, w);
  d_normalize(result, result);
}

// Returns a point equal to the point given plus the given vector
__device__ void d_pointPlusVector(Point result, Point p, Vector v) {
  result[0] = p[0] + v[0];
  result[1] = p[1] + v[1];
  result[2] = p[2] + v[2];
}

// Returns a vector multiplied by scalar f
__device__ void d_scaleVector(Vector result, Vector v, float f) {
  result[0] = v[0] * f;
  result[1] = v[1] * f;
  result[2] = v[2] * f;
}

// Returns the sum of two vectors
__device__ void d_vectorAdd(Vector result, Vector v, Vector w) {
  result[0] = v[0] + w[0];
  result[1] = v[1] + w[1];
  result[2] = v[2] + w[2];
}

// Returns a float describing the distance to an intersection with the given geometry
//   The point of intersection can be calculated by adding the (rayDir * result) + rayPos
//   The normal vector of intersection can be calculated by using a d_*normal() function with the hit point given (if a sphere)
__device__ float d_intersect(Ray ray, int index, int* shapeIndex, ShapeType* shape, float* shapeInfo) {
  
  if (shape[index] == TRIANGLE) { // If triangle
  
    float a = shapeInfo[shapeIndex[index] + 0] - shapeInfo[shapeIndex[index] + 3];
    float b = shapeInfo[shapeIndex[index] + 1] - shapeInfo[shapeIndex[index] + 4];
    float c = shapeInfo[shapeIndex[index] + 2] - shapeInfo[shapeIndex[index] + 5];
    
    float d = shapeInfo[shapeIndex[index] + 0] - shapeInfo[shapeIndex[index] + 6];
    float e = shapeInfo[shapeIndex[index] + 1] - shapeInfo[shapeIndex[index] + 7];
    float f = shapeInfo[shapeIndex[index] + 2] - shapeInfo[shapeIndex[index] + 8];
    
    float g = ray[3];
    float h = ray[4];
    float i = ray[5];
    
    float j = shapeInfo[shapeIndex[index] + 0] - ray[0];
    float k = shapeInfo[shapeIndex[index] + 1] - ray[1];
    float l = shapeInfo[shapeIndex[index] + 2] - ray[2];
    
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
    
    
  } else { // Else, if sphere
    
    float a = d_dotProduct(&ray[3], &ray[3]);
    
    Vector v;
    Vector w;
    d_scaleVector(v, &ray[3], 2);
    d_pointDifference(w, &ray[0], &shapeInfo[shapeIndex[index]]);
    float b = d_dotProduct(v, w);
    
    float c = d_dotProduct(w, w) - (shapeInfo[shapeIndex[index]+3] * shapeInfo[shapeIndex[index]+3]);
    
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
__device__ void d_reflect(Point resultPos, Vector resultDir, Point rayPos, Vector rayDir, Point hit, Vector vector) {
  Vector v;
  Vector w;
  d_scaleVector(v, vector, -2*d_dotProduct(rayDir, vector));
  d_vectorAdd(w, rayDir, v);
  d_normalize(resultDir, w);
  
  // Bump result ray to avoid reflection static
  d_scaleVector(v, resultDir, 0.001);
  d_pointPlusVector(resultPos, hit, v);
}