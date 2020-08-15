// Jordan Jones
// Computer Graphics
// Fall 2019

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "math.h"
#include "ray.h"
#include "cuda.h"

#define MAX_REFLECTIONS 5

// Declare global variables
int xRes, yRes;               // Resolution to render scene in
unsigned char *image;         // Image array, stored in row-major order

GeometryList *list; // List of spheres/triangles

// Geometry constant memory declarations
__constant__ int N;

// Checks CUDA error status and prints any errors
void cudaPrintLastError() {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA Failure in %s:%d:'%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    exit(2);
  }
}

// Ray Tracing kernel
//   Propagates one ray per block; one thread per geometry
//   Requires instantiation with (number of geometry objects) threads per pixel, xRes x yRes blocks
__global__ void traceRay(unsigned char *image, float *radii, float *a, float *b, float *c, int *reflective, unsigned char *color) {
  
  int x = blockDim.x * blockIdx.x + threadIdx.x;  // X-position of pixel
  int y = blockDim.y * blockIdx.y + threadIdx.y;  // Y-position of pixel
  int z = threadIdx.z; // Geometry index at pixel (x, y) to calculate intersection with
  
  extern __shared__ float ds_distances[]; // Calculated distances array (size depends on number of geometries) (row-major ordering)
  
  float rayPos[3]; // Ray point for certain pixel
  float rayDir[3]; // Ray vector for certain pixel
  int numRefl;     // Number of reflections for ray at certain pixel
  

  // Set ray initial position
  rayPos[0] = Camera[0];
  rayPos[1] = Camera[1];
  rayPos[2] = Camera[2];
  
  // Set ray direction based on image plane
  float deltaX = (ImagePlaneTR[0] - ImagePlaneBL[0]) / (float) XRes;
  float deltaY = (ImagePlaneTR[1] - ImagePlaneBL[1]) / (float) YRes;
  rayDir[0] = ImagePlaneBL[0] + (float) x * deltaX + 0.5 * deltaX;
  rayDir[1] = ImagePlaneBL[1] + (float) y * deltaY + 0.5 * deltaY;
  rayDir[2] = (ImagePlaneBL[2] + ImagePlaneTR[2]) / 2;
  d_normalize(rayDir, rayDir);
  
  // Initialize number of reflections
  numRefl = 0;

  /*
  // Debug information
  if (x == 0 && y == 0 && z == 0) {
    printf("Image resolution: %dx%d\n", XRes, YRes);
    printf("deltaX/deltaY = %f, %f\n", deltaX, deltaY);
    printf("ImagePlaneBL: %f, %f, %f\n", ImagePlaneBL[0], ImagePlaneBL[1], ImagePlaneBL[2]);
    printf("ImagePlaneTR: %f, %f, %f\n", ImagePlaneTR[0], ImagePlaneTR[1], ImagePlaneTR[2]);
  }
  */
  
  
  int closestGeom;
  int currentIndex;
  float closestDist;
  float hit[3];
  float hitNorm[3];
  
  do {
  
    // Calculate intersection distances and put in shared array
    ds_distances[N * (threadIdx.y * blockDim.x + threadIdx.x) + z] = d_intersect(rayPos, rayDir, z, radii, a, b, c);
    __syncthreads(); // Synchronizes ONLY the threads in THIS block
    
    
  
    // Thread 0 selects lowest distance
    //if (z == 0) {
      closestGeom = -1;
      closestDist = INFINITY;

      for(int i=0; i<N; i++) {
        currentIndex = N * (threadIdx.y * blockDim.x + threadIdx.x) + i;
        if (ds_distances[currentIndex] < closestDist) {
          closestDist = ds_distances[currentIndex];
          closestGeom = i;
        }
      }
    
      // Calculate position of hit (if any)
      if (closestGeom != -1) {
        d_scaleVector(hit, rayDir, closestDist);
        d_pointPlusVector(hit, rayPos, hit);
      }
    
      // Calculate normal vector of hit object
      if (closestGeom != -1 && radii[closestGeom] == -1) { // Triangle
        d_triangleNormal(hitNorm, &a[3*closestGeom], &b[3*closestGeom], &c[3*closestGeom]);
      } else if (closestGeom != -1) { // Sphere
        d_sphereNormal(hitNorm, &a[3*closestGeom], hit);
      }
      
      if (closestGeom != -1 && reflective[closestGeom]) {
        d_reflect(rayPos, rayDir, rayPos, rayDir, hit, hitNorm);
        numRefl ++;
      } else {
        break;
      }
      
    //}
    
    __syncthreads();

  } while (numRefl <= MAX_REFLECTIONS);
  
  
  
  
  // Raytrace one more time for shadows
  
  int lightDist = d_distance(hit, Light);
  float shadowRay[3];
  int inShadow = 0; // Boolean value for shadow rendering
  
  d_pointDifference(shadowRay, Light, hit);
  d_normalize(shadowRay, shadowRay);
  float temp[3];
  d_scaleVector(temp, shadowRay, 0.001); // Fix shadow acne
  d_pointPlusVector(hit, hit, temp);
  
  ds_distances[N * (threadIdx.y * blockDim.x + threadIdx.x) + z] = d_intersect(hit, shadowRay, z, radii, a, b, c);
  __syncthreads();

  for(int i=0; i<N; i++) {
    currentIndex = N * (threadIdx.y * blockDim.x + threadIdx.x) + i;
    if (ds_distances[currentIndex] < lightDist) {
      inShadow = 1;
      break;
    }
  }
  
  
  

  // Final pixel output
  //   Dependent on hit location, hit normal, hit geometry, and light position
  if (x < XRes && y < YRes && z == 0) {
    
    if (closestGeom == -1) { // If no intersection, show black
      image[(y * XRes + x) * 3 + 0] = 0;
      image[(y * XRes + x) * 3 + 1] = 0;
      image[(y * XRes + x) * 3 + 2] = 0;
      
    } else { // Intersection
    
      // Calculate diffuse shading
      float toLight[3];
      float diffuse = 1;
     
      d_pointDifference(toLight, Light, hit); // Calculate vector to light
      d_normalize(toLight, toLight);
      
      // Calculate diffuse multiplier
      if (inShadow) {
        diffuse = 0.2;
      } else {
        diffuse = d_dotProduct(hitNorm, toLight);
        if (diffuse < 0.2) { // Clamp minimum diffuse value
          diffuse = 0.2;
        }
      }
      
    
      image[(y * XRes + x) * 3 + 0] = color[closestGeom * 3 + 0] * diffuse;
      image[(y * XRes + x) * 3 + 1] = color[closestGeom * 3 + 1] * diffuse;
      image[(y * XRes + x) * 3 + 2] = color[closestGeom * 3 + 2] * diffuse;
    }
  }
  
  /* DEBUG: Create gradient to test output
  if (z == 0 && x < XRes && y < YRes) {
    image[(y * XRes + x) * 3 + 0] = 255 - 255*rayPos[0];
    image[(y * XRes + x) * 3 + 1] = 255 - 255*rayPos[1];
    image[(y * XRes + x) * 3 + 2] = 255 - 255*rayPos[2];
    
    if (y == 0 && x == 0) {
      printf("rayPos: %f, %f, %f\n", rayPos[0], rayPos[1], rayPos[2]);
      printf("rayDir: %f, %f, %f\n", rayDir[0], rayDir[1], rayDir[2]);
    }
  }
  //*/
  
}


int main(int argc, char **argv) {
	
  // ----- PARSE ARGUMENTS -----
  if (argc == 1) { // No resolution specified at runtime
    xRes = 512; // Set default resolution
    yRes = 512;
  } else if (argc == 2) { // Only x resolution specified at runtime
    xRes = atoi(argv[1]);
    yRes = xRes; // Assume 1:1 aspect ratio
  } else if (argc == 3) { // Full resolution specified at runtime
    xRes = atoi(argv[1]);
    yRes = atoi(argv[2]);
  } else {
    printf("Too many arguments!\nUsage: ./ray xRes yRes\n");
    exit(-1);
  }
  printf("Running at resolution of %dx%d, raytracing %d pixels\n", xRes, yRes, xRes*yRes);
  
  // In this program, the origin is assumed to be at the lower left-hand corner of an image
  stbi_flip_vertically_on_write(1);
  
  // Allocate memory for image array
	image = (unsigned char *) malloc(xRes * yRes * 3 * sizeof(unsigned char));
	
	// Check for null pointers
	if (image == NULL) {
		printf("Error allocating memory for image array. Exiting...\n");
		exit(0);
	}
  
  // ----- Initialize GeometryList -----
  list = initGeometryList();
  
  // Geometry 0: Sphere
  float center0[3] = {0, 0, -16};
  unsigned char black[3] = {64, 64, 64};
  addSphere(list, 2, center0, REFL, black);
  
  // Geometry 1: Sphere
  float center1[3] = {3, -1, -14};
  addSphere(list, 1, center1, REFL, black);
  
  // Geometry 2: Sphere
  float center2[3] = {-3, -1, -14};
  unsigned char red[3] = {255, 0, 0};
  addSphere(list, 1, center2, NO_REFL, red);
  
  // Geometry 3: Triangle
  float a1[3] = {-8, -2, -20};
  float b1[3] = { 8, -2, -20};
  float c1[3] = { 8, 10, -20};
  unsigned char blue[3] = {0, 0, 255};
  addTriangle(list, a1, b1, c1, NO_REFL, blue);
  
  // Geometry 4: Triangle
  float c2[3] = {-8, 10, -20};
  addTriangle(list, a1, c1, c2, NO_REFL, blue);
  
  // Geometry 5: Triangle
  float b3[3] = {8, -2, -10};
  unsigned char white[3] = {255, 255, 255};
  addTriangle(list, a1, b3, b1, NO_REFL, white);
  
  // Geometry 6: Triangle
  float b4[3] = {-8, -2, -10};
  addTriangle(list, a1, b4, b3, NO_REFL, white);
  
  // Geometry 7: Triangle
  float c5[3] = {8, 10, -20};
  addTriangle(list, b1, b3, c5, NO_REFL, red);
  //printGeometry(list, 7);
  //printGeometry(list, 8);
  
  /*
  float a6[3] = {-1, 1, -3};
  float b6[3] = {1, 1, -3};
  float c6[3] = {1, -1, -3};
  unsigned char lightBlue[3] = {0, 255, 255};
  
  addTriangle(list, a6, b6, c6, NO_REFL, lightBlue);
  */
  
  // ----- Prepare device constants -----
  float Camera_h[3] = {0, 0, 0};
  float Light_h[3] = {3, 5, -15};
  float ImagePlaneBL_h[3] = {-1, -((float)yRes/(float)xRes), -2};
  float ImagePlaneTR_h[3] = {1, ((float)yRes/(float)xRes), -2};
  printf("ImagePlaneBL: %f, %f, %f\n", ImagePlaneBL_h[0], ImagePlaneBL_h[1], ImagePlaneBL_h[2]);
  printf("ImagePlaneTR: %f, %f, %f\n", ImagePlaneTR_h[0], ImagePlaneTR_h[1], ImagePlaneTR_h[2]);
  
  cudaMemcpyToSymbol(Camera, Camera_h, 3 * sizeof(float));
  cudaMemcpyToSymbol(Light, Light_h, 3 * sizeof(float));
  cudaMemcpyToSymbol(ImagePlaneBL, ImagePlaneBL_h, 3 * sizeof(float));
  cudaMemcpyToSymbol(ImagePlaneTR, ImagePlaneTR_h, 3 * sizeof(float));
  
  cudaMemcpyToSymbol(XRes, &xRes, sizeof(int));
  cudaMemcpyToSymbol(YRes, &yRes, sizeof(int));
  
  cudaMemcpyToSymbol(N, &(list->n), sizeof(int)); // Total number of objects to render
  

  // Prepare device memory
  cudaMalloc((void **) &d_radii, (list->n) * sizeof(int));
  cudaMalloc((void **) &d_a, (list->n) * sizeof(float) * 3);
  cudaMalloc((void **) &d_b, (list->n) * sizeof(float) * 3);
  cudaMalloc((void **) &d_c, (list->n) * sizeof(float) * 3);
  cudaMalloc((void **) &d_reflective, (list->n) * sizeof(int));
  cudaMalloc((void **) &d_color, (list->n) * sizeof(unsigned char) * 3);
  
  cudaMemcpy(d_radii, list->radii, (list->n) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, list->a, (list->n) * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, list->b, (list->n) * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, list->c, (list->n) * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_reflective, list->reflective, (list->n) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, list->color, (list->n) * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
  cudaPrintLastError();
  
  
  // Allocate image output space for device
  unsigned char *d_image;
  cudaMalloc((void **) &d_image, xRes * yRes * 3 * sizeof(unsigned char));
  
  // ----- Prepare kernel dimensions -----
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  //printf("Max threads per block is %d\n", deviceProperties.maxThreadsPerBlock);
  //printf("Max threads per dim: %d %d %d\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
  //printf("Max blocks per dim: %d %d %d\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
  
  int xThreads = sqrt(deviceProperties.maxThreadsPerBlock);
  int yThreads = sqrt(deviceProperties.maxThreadsPerBlock) / (list -> n);
  int zThreads = list -> n;
  
  int xBlocks = xRes / xThreads + 1;
  int yBlocks = yRes / yThreads + 1;
  
  dim3 threads = dim3(xThreads, yThreads, zThreads); // One thread group (x,y) for each pixel, one threads per thread group (z) for each of N geometry
  dim3 blocks = dim3(xBlocks, yBlocks, 1);           // Enough blocks to cover image
  
  // If too big for device, exit (assuming your host can even allocate enough memory for the render, anyways)
  if (xRes > deviceProperties.maxGridSize[0] || yRes > deviceProperties.maxGridSize[1]) {
    printf("Selected image size too large to render; not enough grids on device\n");
    exit(-1);
  }
  
  printf("Threads per block: %dx%dx%d\n", xThreads, yThreads, zThreads);
  printf("Blocks per grid: %dx%d\n", xBlocks, yBlocks);
  
  size_t reqMem = (list->n) * sizeof(float) * xThreads * yThreads; // Enough shared memory for each block to have N distances for each thread group


  
  // ----- RUN KERNEL -----
  printf("Starting ray tracing kernel...\n");
  traceRay<<<blocks, threads, reqMem>>>(d_image, d_radii, d_a, d_b, d_c, d_reflective, d_color);
  cudaPrintLastError();
  printf("\tKernel complete!\n");
  
  // Copy output image to host
  printf("Copying output image to host...\n");
  cudaMemcpy(image, d_image, xRes * yRes * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaPrintLastError();
  printf("\tCopy complete!\n");
  
  // Write ray-traced image
  printf("Writing image to file...\n");
  char filename[100];
  sprintf(filename, "raytraced_%dx%d.png", xRes, yRes);
  stbi_write_png(filename, xRes, yRes, 3, image, xRes*3);
  printf("\tWrite complete!\n");
  
	// Free malloc-ed arrays
	free(image);
}