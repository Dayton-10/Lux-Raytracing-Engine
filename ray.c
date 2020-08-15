// Jordan Jones
// Computer Graphics
// Fall 2019

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "math.h"
#include "ray.h"

#define X_RES 512
#define Y_RES 512
#define MAX_REFLECTIONS 10

// Declare global variables
unsigned char *image;         // Image array, stored in row-major order
Ray *rays;                    // Ray array
Sphere *spheres;              // Sphere array
Triangle *triangles;          // Triangle array
int numSpheres, numTriangles; // Sphere/triangle array sizes

Point camera;       // Camera location
Point light;        // Light source location
Point imagePlaneBL; // Image plane bottom-left corner
Point imagePlaneTR; // Image plane top-right corner


int main(void) {
	
  // In this program, the origin is assumed to be at the lower left-hand corner of an image
  stbi_flip_vertically_on_write(1);
  
  // Initiating materials
  Material refl  = { .r=0,   .g=0,   .b=0,   .reflective=1 };
  Material blue  = { .r=0,   .g=0,   .b=255, .reflective=0 };
  Material red   = { .r=255, .g=0,   .b=0,   .reflective=0 };
  Material white = { .r=255, .g=255, .b=255, .reflective=0 };
  
  // Allocate memory for respective arrays
	image = malloc(X_RES*Y_RES*3*sizeof(unsigned char));
  rays  = malloc(X_RES*Y_RES*sizeof(Ray));
	
	// Check for null pointers
	if (image == NULL) {
		printf("Error allocating memory for image array. Exiting...\n");
		exit(0);
	} else if (rays == NULL) {
    printf("Error allocating memory for ray array. Exiting...\n");
    exit(0);
  }
  
  // Initialize respective locations:
  // Camera
  camera.x = 0;
  camera.y = 0;
  camera.z = 0;
  // Light source
  light.x = 3;
  light.y = 5;
  light.z = -15;
	// Image plane corners
  imagePlaneBL.x = -1;
  imagePlaneBL.y = -1;
  imagePlaneBL.z = -2;
  imagePlaneTR.x = 1;
  imagePlaneTR.y = 1;
  imagePlaneTR.z = -2;
  
  // Instantiate spheres array and fill w/ spheres
  numSpheres = 3;
  spheres   = malloc(numSpheres*sizeof(Sphere));
  if (spheres == NULL) {
		printf("Error allocating memory for spheres array. Exiting...\n");
		exit(0);
	}
  spheres[0] = (Sphere) { .center.x=0,  .center.y=0,  .center.z=-16, .radius=2, .material=refl };
  spheres[1] = (Sphere) { .center.x=3,  .center.y=-1, .center.z=-14, .radius=1, .material=refl };
  spheres[2] = (Sphere) { .center.x=-3, .center.y=-1, .center.z=-14, .radius=1, .material=red };
  
  // Instantiate triangles array and fill w/ triangles
  numTriangles = 5;
  triangles = malloc(numTriangles*sizeof(Triangle));
  if (triangles == NULL) {
		printf("Error allocating memory for triangles array. Exiting...\n");
		exit(0);
	}
  triangles[0] = (Triangle) { .a.x=-8, .a.y=-2, .a.z=-20, .b.x=8,  .b.y=-2, .b.z=-20, .c.x=8,  .c.y=10, .c.z=-20, .material=blue };
  triangles[1] = (Triangle) { .a.x=-8, .a.y=-2, .a.z=-20, .b.x=8,  .b.y=10, .b.z=-20, .c.x=-8, .c.y=10, .c.z=-20, .material=blue };
  triangles[2] = (Triangle) { .a.x=-8, .a.y=-2, .a.z=-20, .b.x=8,  .b.y=-2, .b.z=-10, .c.x=8,  .c.y=-2, .c.z=-20, .material=white };
  triangles[3] = (Triangle) { .a.x=-8, .a.y=-2, .a.z=-20, .b.x=-8, .b.y=-2, .b.z=-10, .c.x=8,  .c.y=-2, .c.z=-10, .material=white };
  triangles[4] = (Triangle) { .a.x=8,  .a.y=-2, .a.z=-20, .b.x=8,  .b.y=-2, .b.z=-10, .c.x=8,  .c.y=10, .c.z=-20, .material=red };
  
  
  // Set ray positions to camera position
  for(int i=0; i<X_RES*Y_RES; i++) {
    rays[i].point.x = camera.x;
    rays[i].point.y = camera.y;
    rays[i].point.z = camera.z;
    rays[i].numReflections = 0; // Ray hasn't reflected yet
  }
  
  // Calculate ray directions
  float deltaX = (imagePlaneTR.x - imagePlaneBL.x) / X_RES;
  float deltaY = (imagePlaneTR.y - imagePlaneBL.y) / Y_RES;
  int currentRay;
	for(int y=0; y<Y_RES; y++) {
    for(int x=0; x<X_RES; x++) {
      currentRay = x+y*X_RES; // Calculate current ray index
      
      // Calculate current ray vector
      rays[currentRay].vector.x = imagePlaneBL.x + x*deltaX + 0.5*deltaX;
      rays[currentRay].vector.y = imagePlaneBL.y + y*deltaY + 0.5*deltaY;
      rays[currentRay].vector.z = (imagePlaneBL.z + imagePlaneTR.z) / 2;
      
      rays[currentRay].vector = normalize(rays[currentRay].vector); // Nomalize resultant vectors
    }
  }
  
  
  
  // Calculate ray intersections
  RayHit hit, temp, shadowHit;
  Ray shadowRay;               // Ray used for testing for shadows
  Vector toLight;              // Vector pointing towards light source
  float diffuse;               // Coefficient used for diffuse shading
  
  for(int y=0; y<Y_RES; y++) {
    for(int x=0; x<X_RES; x++) {
      
      currentRay = x+y*X_RES; // Calculate current ray index
      
      do {
        
        hit.t = INFINITY; // Initialize to no hits
      
        // Find shortest hit in spheres
        for(int i=0; i<numSpheres; i++) {
          temp = sphereIntersect(rays[currentRay], spheres[i]);
          if (temp.t < hit.t && temp.t > 0) {
            hit = temp;
          }
        }
      
        // Find shortest hit in triangles
        for(int i=0; i<numTriangles; i++) {
          temp = triangleIntersect(rays[currentRay], triangles[i]);
          if (temp.t < hit.t && temp.t > 0) {
            hit = temp;
          }
        }
      
        if (hit.t < INFINITY && hit.material.reflective == 1) {
          rays[currentRay] = reflect(rays[currentRay], hit);
        } else {
          break;
        }
      
      } while (hit.material.reflective == 1 && rays[currentRay].numReflections < MAX_REFLECTIONS); // Continue while hitting reflective materials
      
      // Intersection, draw object appropriately
      if (hit.t > 0 && hit.t < INFINITY) {
        
        // Calculate shading
        toLight = normalize(pointDifference(light, hit.p));
        
        // Shadows
        shadowRay.vector = toLight;
        shadowRay.point  = hit.p;
        shadowRay.point  = pointPlusVector(shadowRay.point, scaleVector(shadowRay.vector, 0.001)); // Fix shadow acne
        
       
        // Determine any objects between hit and light source
        shadowHit.t = INFINITY;
        for(int i=0; i<numSpheres; i++) {
          temp = sphereIntersect(shadowRay, spheres[i]);
          if (temp.t < shadowHit.t && temp.t > 0) {
            shadowHit = temp;
          }
        }
        for(int i=0; i<numTriangles; i++) {
          temp = triangleIntersect(shadowRay, triangles[i]);
          if (temp.t < shadowHit.t && temp.t > 0) {
            shadowHit = temp;
          }
        }
        
        // Check if in shadow
        if (shadowHit.t < distance(shadowRay.point, light)) {
          diffuse = 0.2;
        } else { // Diffuse shading
          diffuse = dotProduct(hit.v, toLight);
          if (diffuse < 0.2) {
            diffuse = 0.2;
          }
        }
        
        // Draw pixel with appropriate color and shading
        image[currentRay*3+0] = hit.material.r * diffuse;
        image[currentRay*3+1] = hit.material.g * diffuse;
        image[currentRay*3+2] = hit.material.b * diffuse;
        
      } else {
        // No intersection, draw black
        image[currentRay*3 +0] = 0;
        image[currentRay*3 +1] = 0;
        image[currentRay*3 +2] = 0;
      }
    }
  }
  
  // Write ray-traced image
  stbi_write_png("reference.png", X_RES, Y_RES, 3, image, X_RES*3);
  
	// Free malloc-ed arrays
	free(image);
  free(rays);
  free(spheres);
  free(triangles);
}