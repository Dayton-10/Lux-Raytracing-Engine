// Jordan Jones
// Computer Graphics
// Fall 2019

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "math.h"

#define X_RES 512
#define Y_RES 512

typedef struct Point {
	float x, y, z;
} Point;

typedef struct Vector {
	float x, y, z;
} Vector;

typedef struct Ray {
  Point point;
  Vector vector;
} Ray;

typedef struct Sphere {
  Point center;
  float radius;
} Sphere;

// Create a white test image with pixel 0 being black
void createWhiteImage(unsigned char *image) {
  for(int x=0; x<X_RES; x++) {
    for(int y=0; y<Y_RES; y++) {
      image[(3*x)+(X_RES*y*3)+0] = 255; // Set red channel to max
      image[(3*x)+(X_RES*y*3)+1] = 255; // Set green channel to max
      image[(3*x)+(X_RES*y*3)+2] = 255; // Set blue channel to max
    }
  }

  // Set pixel 0 to black
  image[0] = 0;
  image[1] = 0;
  image[2] = 0;
}

// Create a gradient test image
void createGradientImage(unsigned char *image) {
  for(int x=0; x<X_RES; x++) {
    for(int y=0; y<Y_RES; y++) {
      image[3*(x+X_RES*y)+0] = (float)x/X_RES*255;       // Red gradient (left to right)
      image[3*(x+X_RES*y)+1] = 255 - (float)y/Y_RES*255; // Green gradient (top to bottom)
      image[3*(x+X_RES*y)+2] = 128;                      // Set blue channel to 128
    }
  }
}

// Create a checkerboard test image
void createCheckerboardImage(unsigned char *image) {
  
  for(int y=0; y<Y_RES; y+=128)        // Split image into 128x128 chunks
    for(int x=0; x<X_RES; x+=128)
      for(int yIn=y; yIn<y+128 && yIn<Y_RES; yIn++) // In each 128x128 chunk, iterate through all pixels
        for(int xIn=x; xIn<x+128 && xIn<X_RES; xIn++) {
          
          int currentPixel = xIn + (X_RES * yIn);
          
          if (xIn-x<64 && yIn-y<64) {        // Set lower-left square blue
            image[3*currentPixel+0] = 0;
            image[3*currentPixel+1] = 0;
            image[3*currentPixel+2] = 255;
          } else if (xIn-x>63 && yIn-y>63) { // Set upper-right square blue
            image[3*currentPixel+0] = 0;
            image[3*currentPixel+1] = 0;
            image[3*currentPixel+2] = 255;
          } else {                           // Set rest red
            image[3*currentPixel+0] = 255;
            image[3*currentPixel+1] = 0;
            image[3*currentPixel+2] = 0;
          }
        }
}

// Returns norm of input vector
float norm(Vector v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// Prints vector as "<x, y, z>"
void printVector(Vector v) {
  printf("<%f, %f, %f>", v.x, v.y, v.z);
}

// Prints point as "(x, y, z)"
void printPoint(Point p) {
  printf("(%f, %f, %f)", p.x, p.y, p.z);
}

// Prints ray information in custom assignment format
void printRay(Ray r) {
  printf("RayPosition %f %f %f\n", r.point.x, r.point.y, r.point.z);
  printf("RayDirection %f %f %f\n", r.vector.x, r.vector.y, r.vector.z);
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

// Returns a vector describing the difference of points p and q
Vector pointDifference(Point p, Point q) {
  Vector result;
  result.x = p.x - q.x;
  result.y = p.y - q.y;
  result.z = p.z - q.z;
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

float sphereIntersect(Ray ray, Sphere sphere) {
  // Calculate quadratic solutions for t
  float a = dotProduct(ray.vector, ray.vector);
  float b = dotProduct(scaleVector(ray.vector, 2), pointDifference(ray.point, sphere.center));
  float c = dotProduct(pointDifference(ray.point, sphere.center), pointDifference(ray.point, sphere.center)) - (sphere.radius * sphere.radius);
  float t1 = (-b + sqrtf(b*b - 4*a*c))/(2*a);
  float t2 = (-b - sqrtf(b*b - 4*a*c))/(2*a);
  
  // Return smallest t value
  if (t1>0 && t1<t2) {
    return t1;
  } else {
    return t2;
  }
}

int main(void) {
	
  // In this program, the origin is assumed to be at the lower
  //   left-hand corner of an image
  stbi_flip_vertically_on_write(1);
  
  // Allocate memory for image array
  // The image is stored as described above, in row-major order
	unsigned char *image;
	image = malloc(X_RES*Y_RES*3*sizeof(unsigned char));
	
	// Check for null pointer
	if (image == NULL) {
		printf("Error allocating memory for image array. Exiting...\n");
		exit(0);
	}
  
  // Generate white test image with lower-left pixel black
  //createWhiteImage(image);
  //stbi_write_png("white.png", X_RES, Y_RES, 3, image, X_RES*3);
  
  // Generate gradient test image
  createGradientImage(image);
  stbi_write_png("gradient.png", X_RES, Y_RES, 3, image, X_RES*3);
  
  // Generate checkerboard test image
  createCheckerboardImage(image);
  stbi_write_png("checkerboard.png", X_RES, Y_RES, 3, image, X_RES*3);
  
  // Allocate memory for ray array
  Ray *rays;
  rays = malloc(X_RES*Y_RES*sizeof(Ray));
  
  // Check for null pointer
  if (rays == NULL) {
    printf("Error allocating memory for ray array. Exiting...\n");
    exit(0);
  }
  
  // Create camera location
  Point camera;
  camera.x = 0;
  camera.y = 0;
  camera.z = 0;
	
	// Create image plane corners
  // TL = top left, BR = bottom right
  Point imagePlaneBL;
  imagePlaneBL.x = -1;
  imagePlaneBL.y = -1;
  imagePlaneBL.z = -2;
  
  Point imagePlaneTR;
  imagePlaneTR.x = 1;
  imagePlaneTR.y = 1;
  imagePlaneTR.z = -2;
  
  // Set ray positions; all (0,0,0)  
  for(int x=0; x<X_RES; x++) {
    for(int y=0; y<Y_RES; y++) {
      rays[x+y*X_RES].point.x = camera.x;
      rays[x+y*X_RES].point.y = camera.y;
      rays[x+y*X_RES].point.z = camera.z;
    }
  }
  
  // Calculate ray directions
  float pixelWidth = (imagePlaneTR.x - imagePlaneBL.x) / X_RES;
  float pixelHeight = (imagePlaneTR.y - imagePlaneBL.y) / Y_RES;
  int currentRay;
	for(int y=0; y<Y_RES; y++) {
    for(int x=0; x<X_RES; x++) {
      currentRay = x+y*X_RES; // Calculate current ray index
      
      // Calculate current ray vector
      rays[currentRay].vector.x = imagePlaneBL.x + x*pixelWidth + 0.5*pixelWidth;
      rays[currentRay].vector.y = imagePlaneBL.y + y*pixelHeight + 0.5*pixelHeight;
      rays[currentRay].vector.z = -2;
      
      rays[currentRay].vector = normalize(rays[currentRay].vector); // Nomalize resultant vectors
    }
  }
  
  // Print bottom-left pixel ray info
  printf("Bottom left pixel\n");
  printRay(rays[0+0*X_RES]);
  printf("\n");
  
  // Print top-right pixel ray info
  printf("Top right pixel\n");
  printRay(rays[(X_RES-1)+(Y_RES-1)*X_RES]);
  printf("\n");
  
  // Print middle pixel ray info
  printf("Middle pixel\n");
  printRay(rays[(X_RES/2)+(Y_RES/2)*X_RES]);
  printf("\n");
  
  // Instantiate sphere
  Sphere sphere;
  sphere.center.x = 2;
  sphere.center.y = 2;
  sphere.center.z = -16;
  sphere.radius = 5.3547;
  
  // Calculate ray-sphere intersections
  float intersection;
  for(int y=0; y<Y_RES; y++)
    for(int x=0; x<X_RES; x++) {
      currentRay = x+y*X_RES;
      intersection = sphereIntersect(rays[currentRay], sphere);
      
      if (intersection > 0) {         // Intersection, draw white
        image[currentRay*3 +0] = 255;
        image[currentRay*3 +1] = 255;
        image[currentRay*3 +2] = 255;
      } else {                        // No intersection, draw red
        image[currentRay*3 +0] = 128;
        image[currentRay*3 +1] = 0;
        image[currentRay*3 +2] = 0;
      }
      
    }
  
  // Write ray-traced image
  stbi_write_png("sphere.png", X_RES, Y_RES, 3, image, X_RES*3);
  
	
	// Free malloc-ed arrays
	free(image);
  free(rays);
}