#include <windows.h>
#include <commctrl.h>

#include "resource.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "math.h"
#include "ray.h"
#include "cuda.h"

#define WINDOW_WIDTH (1920/4)
#define WINDOW_HEIGHT (1080/4)

// Values suggested by Microsoft's UX Style Guide: https://docs.microsoft.com/en-us/windows/win32/uxguide/vis-layout
#define BUTTON_WIDTH 75
#define BUTTON_HEIGHT 23
#define MARGINS 11
#define PROGRESS_HEIGHT 15
#define EDIT_WIDTH 75
#define EDIT_HEIGHT 23
#define ALIKE_MARGINS 7
#define LABEL_MARGIN 5

#define MAX_REFLECTIONS 10

GeometryList* geomList = NULL;

const char szTitle[] = "Lux Raytracer";

HWND startButton = NULL;
HWND xResEdit = NULL;
HWND yResEdit = NULL;
HWND xResDesc = NULL;
HWND yResDesc = NULL;
HWND progress = NULL;

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// Geometry constant memory declarations
__constant__ int NShapes;
__constant__ int NInfo;

//GeometryList* geomList; // List of spheres/triangles

// Checks CUDA error status and prints any errors
void cudaPrintLastError(const char *file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA Failure in %s:%d:'%s'\n", file, line, cudaGetErrorString(e));
    exit(2);
  }
}
#define cudaPrintLastError() cudaPrintLastError( __FILE__, __LINE__)

// Ray Tracing kernel
//   Propagates one ray per block; one thread per geometry
//   Requires instantiation with (number of geometry objects) threads per pixel, xRes x yRes blocks
__global__ void traceRay(unsigned char* image, int* shapeIndex, ShapeType* shape, Reflectivity* reflective, unsigned char* color, float* shapeInfo) {
  
  int x = blockDim.x * blockIdx.x + threadIdx.x;  // X-position of pixel
  int y = blockDim.y * blockIdx.y + threadIdx.y;  // Y-position of pixel
  int z = threadIdx.z; // Geometry index at pixel (x, y) to calculate intersection with
  
  extern __shared__ float ds_distances[]; // Calculated distances array (size depends on number of geometries) (row-major ordering)
  
  Ray ray; // Ray for certain pixel
  
  int numRefl;     // Number of reflections for ray at certain pixel
  
  // Set ray initial position
  ray[0] = Camera[0];
  ray[1] = Camera[1];
  ray[2] = Camera[2];
  
  // Set ray direction based on image plane
  float deltaX = (ImagePlaneTR[0] - ImagePlaneBL[0]) / (float) XRes;
  float deltaY = (ImagePlaneTR[1] - ImagePlaneBL[1]) / (float) YRes;
  
  // Set ray direction based on image plane
  ray[3] = ImagePlaneBL[0] + (float) x * deltaX + 0.5 * deltaX;
  ray[4] = ImagePlaneBL[1] + (float) y * deltaY + 0.5 * deltaY;
  ray[5] = (ImagePlaneBL[2] + ImagePlaneTR[2]) / 2;
  d_normalize(&ray[3], &ray[3]);
  
  // Initialize number of reflections
  numRefl = 0;
  
  int closestGeom;
  int currentIndex;
  float closestDist;
  Point hit;
  Vector hitNorm;
  
  do {
  
    // Calculate intersection distances and put in shared array
    ds_distances[NShapes * (threadIdx.y * blockDim.x + threadIdx.x) + z] = d_intersect(ray, z, shapeIndex, shape, shapeInfo); //d_intersect(&ray[0], &ray[3], z, radii, a, b, c);
    
    __syncthreads(); // Synchronizes ONLY the threads in THIS block
    
    closestGeom = -1;
    closestDist = INFINITY;

    for(int i=0; i<NShapes; i++) {
      currentIndex = NShapes * (threadIdx.y * blockDim.x + threadIdx.x) + i;
      if (ds_distances[currentIndex] < closestDist) {
        closestDist = ds_distances[currentIndex];
        closestGeom = i;
      }
    }
    
    // Calculate position of hit (if any)
    if (closestGeom != -1) {
      d_scaleVector(hit, &ray[3], closestDist);
      d_pointPlusVector(hit, &ray[0], hit);
    }
    
    // Calculate normal vector of hit object
    if (closestGeom != -1 && shape[closestGeom] == TRIANGLE) { // Triangle
      d_triangleNormal(hitNorm, &shapeInfo[shapeIndex[closestGeom] + 0], &shapeInfo[shapeIndex[closestGeom] + 3], &shapeInfo[shapeIndex[closestGeom] + 6]);
    } else if (closestGeom != -1) { // Sphere
      d_sphereNormal(hitNorm, &shapeInfo[shapeIndex[closestGeom]], hit);
    }
      
    if (closestGeom != -1 && reflective[closestGeom]) {
      d_reflect(&ray[0], &ray[3], &ray[0], &ray[3], hit, hitNorm);
      numRefl ++;
    } else {
      break;
    }
    
    __syncthreads();

  } while (numRefl <= MAX_REFLECTIONS);
  
  // Raytrace one more time for shadows
  Ray shadowRay;
  int inShadow = 0; // Boolean value for shadow rendering
  
  d_pointDifference(&shadowRay[3], Light, hit);
  d_normalize(&shadowRay[3], &shadowRay[3]);
  Vector temp;
  d_scaleVector(temp, &shadowRay[3], 0.001); // Fix shadow acne
  d_pointPlusVector(hit, hit, temp);
  
  shadowRay[0] = hit[0];
  shadowRay[1] = hit[1];
  shadowRay[2] = hit[2];
  
  ds_distances[NShapes * (threadIdx.y * blockDim.x + threadIdx.x) + z] = d_intersect(shadowRay, z, shapeIndex, shape, shapeInfo); //d_intersect(hit, shadowRay, z, radii, a, b, c);
  __syncthreads();

  for(int i=0; i<NShapes; i++) {
    currentIndex = NShapes * (threadIdx.y * blockDim.x + threadIdx.x) + i;
    if (ds_distances[currentIndex] < d_distance(hit, Light)) {
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
      Vector toLight;
      float diffuse;
      
      Vector camDir;
      Vector halfVec;
      float specular;
     
      d_pointDifference(toLight, Light, hit); // Calculate vector to light
      d_normalize(toLight, toLight);
      
      d_pointDifference(camDir, Camera, hit); // Calculate vector to camera
      d_normalize(camDir, camDir);
      
      d_vectorAdd(halfVec, camDir, toLight); // Calculate half-vector
      d_scaleVector(halfVec, halfVec, 0.5);
      d_normalize(halfVec, halfVec);
      
      specular = d_dotProduct(hitNorm, halfVec);
      
      if (specular == 0) {
        specular = 0;
      } else {
        specular = pow(specular, 10);
      }
      
      // Calculate diffuse multiplier
      if (inShadow) {
        diffuse = 0.2;
        specular = 0;
      } else {
        diffuse = d_dotProduct(hitNorm, toLight);
        if (diffuse < 0.2) { // Clamp minimum diffuse value
          diffuse = 0.2;
        }
      }
      
    
      image[(y * XRes + x) * 3 + 0] = color[closestGeom * 3 + 0] / 2 * (diffuse + specular);
      image[(y * XRes + x) * 3 + 1] = color[closestGeom * 3 + 1] / 2 * (diffuse + specular);
      image[(y * XRes + x) * 3 + 2] = color[closestGeom * 3 + 2] / 2 * (diffuse + specular);
    }
    
    // Test gradient
    //image[(y * XRes + x) * 3 + 0] = 255 * ((float)x/XRes);
    //image[(y * XRes + x) * 3 + 1] = 255 * ((float)y/YRes);
    //image[(y * XRes + x) * 3 + 2] = 0;
  }
}

void initGeometry( ) {
  
  // ----- Initialize GeometryList -----
  geomList = initGeometryList();
  
  // Geometry 0: Sphere
  Point center0 = {0, 0, -16};
  Color black = {64, 64, 64};
  addSphere(geomList, center0, 2, black, REFL);
  //printGeometry(geomList, 0);
  //printf("Geometry 0: %s\n", toStringGeometry(geomList, 0)); // Not recommended; string is not freed
  
  // Geometry 1: Sphere
  Point center1 = {3, -1, -14};
  addSphere(geomList, center1, 1, black, REFL);
  
  // Geometry 2: Sphere
  Point center2 = {-3, -1, -14};
  Color red = {255, 0, 0};
  addSphere(geomList, center2, 1, red, NO_REFL);
  
  // Geometry 3: Triangle
  Point a1 = {-8, -2, -20};
  Point b1 = { 8, -2, -20};
  Point c1 = { 8, 10, -20};
  Color blue = {0, 0, 255};
  addTriangle(geomList, a1, b1, c1, blue, NO_REFL);
  
  // Geometry 4: Triangle
  Point c2 = {-8, 10, -20};
  addTriangle(geomList, a1, c1, c2, blue, NO_REFL);
  
  // Geometry 5: Triangle
  Point b3 = {8, -2, -10};
  Color white = {255, 255, 255};
  addTriangle(geomList, a1, b3, b1, white, NO_REFL);
  
  // Geometry 6: Triangle
  Point b4 = {-8, -2, -10};
  addTriangle(geomList, a1, b4, b3, white, NO_REFL);
  
  // Geometry 7: Triangle
  Point c5 = {8, 10, -20};
  addTriangle(geomList, b1, b3, c5, red, NO_REFL);
  
  printAllGeometry(geomList);
  removeGeometry(geomList, 1);
  printAllGeometry(geomList);
}

void rayTrace(int xRes, int yRes) {
  
  printf("Running at resolution of %dx%d, raytracing %d pixels\n", xRes, yRes, xRes*yRes);
  
  // In this program, the origin is assumed to be at the lower left-hand corner of an image
  stbi_flip_vertically_on_write(1);
  
  // Allocate memory for image array (row-major order)
	unsigned char* image = (unsigned char *) malloc(xRes * yRes * 3 * sizeof(unsigned char));
	
	// Check for null pointers
	if (image == NULL) {
		printf("Error allocating memory for image array. Exiting...\n");
		exit(0);
	}
  
  // ----- Prepare device constants -----
  Point Camera_h = {0, 0, 0};
  Point Light_h = {3, 5, -15};
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
  
  cudaMemcpyToSymbol(NShapes, &(geomList->nShapes), sizeof(int)); // Total number of objects to render
  cudaMemcpyToSymbol(NInfo,   &(geomList->nInfo),   sizeof(int)); // Total number of object data points
  

  // Prepare device memory
  cudaMalloc((void**) &d_shapeIndex, sizeof(int)           * (geomList->nShapes));
  cudaMalloc((void**) &d_shape,      sizeof(ShapeType)     * (geomList->nShapes));
  cudaMalloc((void**) &d_reflective, sizeof(Reflectivity)  * (geomList->nShapes));
  cudaMalloc((void**) &d_color,      sizeof(Color)         * (geomList->nShapes));
  cudaMalloc((void**) &d_shapeInfo,  sizeof(float)         * (geomList->nInfo));
  
  cudaMemcpy(d_shapeIndex, geomList->shapeIndex, sizeof(int)           * (geomList->nShapes), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shape,      geomList->shape,      sizeof(ShapeType)     * (geomList->nShapes), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reflective, geomList->reflective, sizeof(Reflectivity)  * (geomList->nShapes), cudaMemcpyHostToDevice);
  cudaMemcpy(d_color,      geomList->color,      sizeof(Color)         * (geomList->nShapes), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shapeInfo,  geomList->shapeInfo,  sizeof(float)         * (geomList->nInfo),   cudaMemcpyHostToDevice);
  
  cudaPrintLastError();
  
  
  // Allocate image output space for device
  unsigned char* d_image;
  cudaMalloc((void **) &d_image, xRes * yRes * 3 * sizeof(unsigned char));
  
  // ----- Prepare kernel dimensions -----
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  //printf("Max threads per block is %d\n", deviceProperties.maxThreadsPerBlock);
  //printf("Max threads per dim: %d %d %d\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
  //printf("Max blocks per dim: %d %d %d\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
  
  int xThreads = sqrt(deviceProperties.maxThreadsPerBlock);
  int yThreads = sqrt(deviceProperties.maxThreadsPerBlock) / (geomList -> nShapes);
  int zThreads = geomList -> nShapes;
  
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
  
  size_t reqMem = (geomList->nShapes) * sizeof(float) * xThreads * yThreads; // Enough shared memory for each block to have N distances for each thread group

  
  // ----- RUN KERNEL -----
  printf("Starting ray tracing kernel...\n");
  traceRay<<<blocks, threads, reqMem>>>(d_image, d_shapeIndex, d_shape, d_reflective, d_color, d_shapeInfo);
  cudaPrintLastError();
  printf("\tKernel complete!\n");
  
  SendMessage(progress, PBM_SETPOS, 50, 0);
  
  // Copy output image to host
  printf("Copying output image to host...\n");
  cudaMemcpy(image, d_image, xRes * yRes * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaPrintLastError();
  printf("\tCopy complete!\n");
  
  SendMessage(progress, PBM_SETPOS, 75, 0);
  
  // Write ray-traced image
  printf("Writing image to file...\n");
  char filename[100];
  sprintf(filename, "raytraced_%dx%d.png", xRes, yRes);
  stbi_write_png(filename, xRes, yRes, 3, image, xRes*3);
  printf("\tWrite complete!\n");
  
	// Free malloc-ed arrays
	free(image);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
  
  initGeometry( );
  
  // Create & populate WNDCLASSEX structure
  WNDCLASSEX wcex;  
  wcex.cbSize         = sizeof(WNDCLASSEX);
  wcex.style          = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc    = (WNDPROC) WndProc;
  wcex.cbClsExtra     = 0;
  wcex.cbWndExtra     = 0;
  wcex.hInstance      = hInstance;
  wcex.hIcon          = LoadIcon(NULL, IDI_APPLICATION); // Load program icon (32x32)
  wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
  wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
  wcex.lpszMenuName   = MAKEINTRESOURCE(IDR_MENU);
  wcex.lpszClassName  = szTitle;
  wcex.hIconSm        = LoadIcon(NULL, IDI_APPLICATION); // Load small version of program icon
  
  // Register WNDCLASSEX instance with Windows
  if (!RegisterClassEx(&wcex))  
  {
    MessageBox(NULL, "Window registration failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
    return -1;
  }
  
  // Create window
  /* The parameters to CreateWindow explained:  
  // szWindowClass: the name of the application  
  // szTitle: the text that appears in the title bar  
  // WS_OVERLAPPEDWINDOW: the type of window to create  
  // CW_USEDEFAULT, CW_USEDEFAULT: initial position (x, y)  
  // 500, 100: initial size (width, length)  
  // NULL: the parent of this window  
  // NULL: this application does not have a menu bar  
  // hInstance: the first parameter from WinMain  
  // NULL: not used in this application
  */
  HWND hWnd = CreateWindow(szTitle, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT, NULL, NULL, hInstance, NULL);

  if (!hWnd)  
  {  
    MessageBox(NULL, "Window creation failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);  
    return -1;
  }
  
  // Make window visible
  /* The parameters to ShowWindow explained:  
  // hWnd: the value returned from CreateWindow  
  // nCmdShow: the fourth parameter from WinMain
  */
  ShowWindow(hWnd, nCmdShow);
  UpdateWindow(hWnd);
  
  // Message Loop
  MSG Msg;
  while(GetMessage(&Msg, NULL, 0, 0) > 0) { // Grab next message in queue (blocking) (returns 0 upon WM_QUIT)
    TranslateMessage(&Msg);
    DispatchMessage(&Msg);
  }
  return Msg.wParam;
}

// Defines positions for each control relative to the window's size
void positionControls(HWND hwnd) {
  RECT windowSize;
  GetClientRect(hwnd, &windowSize);
  
  SetWindowPos(progress,    HWND_TOP,                 windowSize.left + MARGINS,      windowSize.bottom - PROGRESS_HEIGHT - MARGINS - 3,              windowSize.right - 3 * MARGINS - BUTTON_WIDTH, PROGRESS_HEIGHT, SWP_SHOWWINDOW);
  SetWindowPos(startButton, HWND_TOP, windowSize.right - BUTTON_WIDTH - MARGINS,            windowSize.bottom - MARGINS - BUTTON_HEIGHT,                                               BUTTON_WIDTH,   BUTTON_HEIGHT, SWP_SHOWWINDOW);
  SetWindowPos(xResDesc,    HWND_TOP,                 windowSize.left + MARGINS,                               windowSize.top + MARGINS, windowSize.right - MARGINS - EDIT_WIDTH - LABEL_MARGIN - 1,     EDIT_HEIGHT, SWP_SHOWWINDOW);
  SetWindowPos(xResEdit,    HWND_TOP,   windowSize.right - EDIT_WIDTH - MARGINS,                               windowSize.top + MARGINS,                                                 EDIT_WIDTH,     EDIT_HEIGHT, SWP_SHOWWINDOW);
  SetWindowPos(yResDesc,    HWND_TOP,                 windowSize.left + MARGINS, windowSize.top + MARGINS + EDIT_HEIGHT + ALIKE_MARGINS, windowSize.right - MARGINS - EDIT_WIDTH - LABEL_MARGIN - 1,     EDIT_HEIGHT, SWP_SHOWWINDOW);
  SetWindowPos(yResEdit,    HWND_TOP,   windowSize.right - EDIT_WIDTH - MARGINS, windowSize.top + MARGINS + EDIT_HEIGHT + ALIKE_MARGINS,                                                 EDIT_WIDTH,     EDIT_HEIGHT, SWP_SHOWWINDOW);
}

// Initialize controls with placeholder placing/sizes (replaced using positionControls function)
void initControls(HWND hwnd) {
  xResDesc    = CreateWindow("static",       "X Resolution:", WS_CHILD | WS_VISIBLE | SS_RIGHT,                          0, 0, 0, 0, hwnd, (HMENU) IDC_TEXT_XRES,       GetModuleHandle(NULL), NULL);
  xResEdit    = CreateWindow("edit",         "512",           WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER | ES_CENTER, 0, 0, 0, 0, hwnd, (HMENU) IDC_EDIT_XRES,       GetModuleHandle(NULL), NULL);
  yResDesc    = CreateWindow("static",       "Y Resolution:", WS_CHILD | WS_VISIBLE | SS_RIGHT,                          0, 0, 0, 0, hwnd, (HMENU) IDC_TEXT_YRES,       GetModuleHandle(NULL), NULL);
  yResEdit    = CreateWindow("edit",         "512",           WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER | ES_CENTER, 200, 150, 50, 20, hwnd, (HMENU) IDC_EDIT_YRES,       GetModuleHandle(NULL), NULL);
  progress    = CreateWindow(PROGRESS_CLASS, NULL,            WS_CHILD | WS_VISIBLE | PBS_SMOOTH,                        0, 0, 0, 0, hwnd, (HMENU) IDC_RENDER_PROGRESS, GetModuleHandle(NULL), NULL);
  startButton = CreateWindow("button",       "Start",         WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,                  0, 0, 0, 0, hwnd, (HMENU) ID_BUTTON_START,     GetModuleHandle(NULL), NULL);
}

HWND shapeSelectCombo = NULL;
HWND dialogAddText = NULL;
HWND dialogButtonAdd = NULL;
HWND dialogListView = NULL;

void initDialogControls(HWND hwnd) {
  shapeSelectCombo = CreateWindow("combobox", NULL,  CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_CHILD | WS_VISIBLE, 25, 10, 150, 80, hwnd, (HMENU) IDD_COMBO_TYPE, GetModuleHandle(NULL), NULL);
  SendDlgItemMessage(hwnd, IDD_COMBO_TYPE, CB_ADDSTRING, 0, (LPARAM) "Circle");
  SendDlgItemMessage(hwnd, IDD_COMBO_TYPE, CB_ADDSTRING, 0, (LPARAM) "Triangle");
  SendDlgItemMessage(hwnd, IDD_COMBO_TYPE, CB_SETCURSEL, 0, 0);

  dialogAddText    = CreateWindow("static",  "Add:", WS_CHILD | WS_VISIBLE, 7, 10, 14, 8, hwnd, (HMENU) IDD_TEXT_ADD, GetModuleHandle(NULL), NULL);
  dialogButtonAdd  = CreateWindow("button",  "&Add", WS_CHILD | WS_VISIBLE, 25, 25, 150, 20, hwnd, (HMENU) IDD_ADD_GEOM, GetModuleHandle(NULL), NULL);
  /*dialogListView   = CreateWindow(WC_LISTBOX, NULL,  WS_CHILD | WS_VISIBLE | LBS_HASSTRINGS | WS_VSCROLL | WS_HSCROLL | LBS_DISABLENOSCROLL, 0, 0, 0, 0, hwnd, (HMENU) IDD_LIST_BOX, GetModuleHandle(NULL), NULL);
  for(int i=0; i<geomList->nShapes; i++) {
    SendDlgItemMessage(hwnd, IDD_LIST_BOX, LB_ADDSTRING, 0, (LPARAM) toStringGeometry(geomList, i));
  }
  */
  dialogListView = CreateWindow(WC_LISTVIEW, NULL, WS_CHILD | WS_VISIBLE | LVS_EDITLABELS | LVS_REPORT | LVS_EX_FULLROWSELECT, 0, 0, 0, 0, hwnd, (HMENU) IDD_LIST_BOX, GetModuleHandle(NULL), NULL);
  
  LVCOLUMN lvc;
  lvc.mask = LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM;
  lvc.iSubItem = 0;
  lvc.pszText = (char*) "Column 1\0";
  lvc.cx = 100;
  //lvc.fmt = LVCFMT_LEFT;
  //ListView_InsertColumn(dialogListView, 0, &lvc);
  SendDlgItemMessage(hwnd, IDD_LIST_BOX, LVM_INSERTCOLUMN, 0, (LPARAM) &lvc);
  
  lvc.iSubItem = 1;
  lvc.pszText = (char*) "Column 2\0";
  SendDlgItemMessage(hwnd, IDD_LIST_BOX, LVM_INSERTCOLUMN, 1, (LPARAM) &lvc);
  
  LVITEM item;
  item.mask = LVIF_TEXT;
  
  item.iItem = 0;
  item.iSubItem = 0;
  item.pszText = (char*) "Test Item 1\0";
  
  SendDlgItemMessage(hwnd, IDD_LIST_BOX, LVM_INSERTITEM, 0, (LPARAM) &item);
  
  //item.iItem = 0;
  item.iSubItem = 1;
  item.pszText = (char*) "Test Item 2\0";
  SendDlgItemMessage(hwnd, IDD_LIST_BOX, LVM_SETITEM, 0, (LPARAM) &item);
}

void positionDialogControls(HWND hwnd) {
  RECT dialogSize;
  GetClientRect(hwnd, &dialogSize);
  
  SetWindowPos(dialogListView, HWND_TOP, dialogSize.left + MARGINS, dialogSize.top + 75, dialogSize.right - 2*MARGINS, dialogSize.bottom - MARGINS - 75, SWP_SHOWWINDOW);
}

// Handles Edit Scene dialog messages
LRESULT CALLBACK EditDlgProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  
  switch(uMsg) {
    case WM_INITDIALOG: {
      
      initDialogControls(hwnd);
      positionDialogControls(hwnd);
      
      // Load list data from GeometryList
    }
    return TRUE;
    
    case WM_COMMAND: {
      switch(LOWORD(wParam)) {
        case IDD_ADD_GEOM: {
          char buf[16];
          GetDlgItemText(hwnd, IDD_COMBO_TYPE, buf, GetWindowTextLength(GetDlgItem(hwnd, IDD_COMBO_TYPE)) + 1);
          MessageBox(hwnd, buf, "User Choice", MB_OK | MB_ICONINFORMATION);
        }
      }
    }
    break;
    
    case WM_NOTIFY: {
      switch(LOWORD(wParam)) {
        case IDD_LIST_BOX: {
          return TRUE;
        }
      }
    }
    break;
    
    case WM_CLOSE:
      // Save list data to GeometryList before closing
      EndDialog(hwnd, 0);
    break;
    
    default:
      return FALSE;
  }
  return TRUE;
}

// Handles window messages
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  
  switch(uMsg) {
    case WM_CREATE: {
      initControls(hwnd);
      positionControls(hwnd);
    }
    break;
    
    // Prevent window from getting smaller than original size
    case WM_GETMINMAXINFO:
    {
      LPMINMAXINFO lpMMI = (LPMINMAXINFO) lParam;
      lpMMI -> ptMinTrackSize.x = WINDOW_WIDTH;
      lpMMI -> ptMinTrackSize.y = WINDOW_HEIGHT;
    }
    
    // Reached when window is resized
    case WM_SIZE: {
      positionControls(hwnd);
    }
    break;
    
    case WM_COMMAND: {
      switch(LOWORD(wParam)) {
        
        case ID_BUTTON_START: {
          
          BOOL xSuccess, ySuccess;
          int xResolution = GetDlgItemInt(hwnd, IDC_EDIT_XRES, &xSuccess, FALSE);
          int yResolution = GetDlgItemInt(hwnd, IDC_EDIT_YRES, &ySuccess, FALSE);
          if (!xSuccess || !ySuccess || xResolution < 1 || yResolution < 1) {
            MessageBox(hwnd, "Invalid Resolution!", "Error", MB_OK | MB_ICONEXCLAMATION);
            break;
          }
          
          EnableWindow(startButton, FALSE);
          SendMessage(progress, PBM_SETPOS, 0, 0);
          rayTrace(xResolution, yResolution);
          EnableWindow(startButton, TRUE);
          SendMessage(progress, PBM_SETPOS, 100, 0);
        }
        break;
        
        case IDM_FILE_EXIT:
          PostMessage(hwnd, WM_CLOSE, 0, 0);
        break;
        
        case IDM_EDIT_SCENE: {
          DialogBox(GetModuleHandle(NULL), MAKEINTRESOURCE(IDD_EDIT_SCENE), hwnd, EditDlgProc);
        }
        break;
      }
    }
    break;
    
    case WM_CLOSE:
      DestroyWindow(hwnd);
    break;
    
    case WM_DESTROY:
      PostQuitMessage(0);
    break;
    
    default:
      return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
  
  return 0;
}