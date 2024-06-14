/*
  CSCI 420 Computer Graphics, Computer Science, USC
  Assignment 2: Roller Coaster
  C/C++ starter code

  Student username: shuklaak
*/

#include "openGLHeader.h"
#include "glutHeader.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "pipelineProgram.h"
#include "vbo.h"
#include "vao.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openGLHeader.h"
#include "imageIO.h"
#include <iostream>
#include <cstring>
// Additional include statements to allow for the use of glm library
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#if defined(WIN32) || defined(_WIN32)
#ifdef _DEBUG
#pragma comment(lib, "glew32d.lib")
#else
#pragma comment(lib, "glew32.lib")
#endif
#endif

#if defined(WIN32) || defined(_WIN32)
char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
char shaderBasePath[1024] = "../openGLHelper";
#endif

using namespace std;

int mousePos[2]; // x,y screen coordinates of the current mouse position

int leftMouseButton = 0;   // 1 if pressed, 0 if not
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0;  // 1 if pressed, 0 if not

// Represents one spline control point.
struct Point
{
  double x, y, z;
};

// Contains the control points of the spline.
struct Spline
{
  int numControlPoints;
  Point *points;
} spline;

typedef enum
{
  ROTATE,    // Triggered by dragging of mouse
  TRANSLATE, // Triggered by holding down 't' key and dragging of mouse
  SCALE      // Triggered by holding down 'Shift' key and dragging of mouse
} CONTROL_STATE;
CONTROL_STATE controlState = ROTATE; // Default control state is ROTATE

// Transformations of the terrain.
float terrainRotate[3] = {0.0f, 0.0f, 0.0f};
// terrainRotate[0] gives the rotation around x-axis (in degrees)
// terrainRotate[1] gives the rotation around y-axis (in degrees)
// terrainRotate[2] gives the rotation around z-axis (in degrees)
float terrainTranslate[3] = {0.0f, 0.0f, 0.0f};
float terrainScale[3] = {1.0f, 1.0f, 1.0f};

// Width and height of the OpenGL window, in pixels.
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 Homework 2";

// Stores the image loaded from disk.
ImageIO *heightmapImage;

// Number of vertices required for different objects
int numVerticesLines;
int numVerticesTrack;
int numVerticesTie;

// CSCI 420 helper classes.
OpenGLMatrix matrix;
PipelineProgram *pipelineProgram = nullptr;
// New pipeline program for the textured sky and ground
PipelineProgram *texPipelineProgram = nullptr;

// VBO and VAO global variables for sky box and ground rendering
VBO *vboVerticesGround = nullptr;
VBO *vboTextureGround = nullptr;
VAO *vaoGround = nullptr;

VBO *vboVerticesSkyTop = nullptr;
VBO *vboTextureSkyTop = nullptr;
VAO *vaoSkyTop = nullptr;

VBO *vboVerticesSkyFront = nullptr;
VBO *vboTextureSkyFront = nullptr;
VAO *vaoSkyFront = nullptr;

VBO *vboVerticesSkyBack = nullptr;
VBO *vboTextureSkyBack = nullptr;
VAO *vaoSkyBack = nullptr;

VBO *vboVerticesSkyLeft = nullptr;
VBO *vboTextureSkyLeft = nullptr;
VAO *vaoSkyLeft = nullptr;

VBO *vboVerticesSkyRight = nullptr;
VBO *vboTextureSkyRight = nullptr;
VAO *vaoSkyRight = nullptr;

// VBO and VAO global variables for track rendering
VBO *vboVerticesLeftTrack = nullptr;
VBO *vboNormalsLeftTrack = nullptr;
VAO *vaoLeftTrack = nullptr;

VBO *vboVerticesRightTrack = nullptr;
VBO *vboNormalsRightTrack = nullptr;
VAO *vaoRightTrack = nullptr;

VBO *vboVerticesTie = nullptr;
VBO *vboNormalsTie = nullptr;
VAO *vaoTie = nullptr;

// Vectors defined to use for camera movement (Level 2 functionality)
glm::vec3 cameraB; // Used to store the previous frame's binormal vector for normal calculation
float cameraU = 0; // Used to iterate through Frenet Frames for camera motion
std::vector<glm::vec3> splineTangents;
std::vector<glm::vec3> splineNormals;
std::vector<glm::vec3> splineBinormals;
std::vector<glm::vec3> splinePositions;

// Texture handles for two different possible textures
GLuint groundTexture;
GLuint skyTexture;

// Boolean in order to pause and unpause camera motion
bool pause = false;

// Write a screenshot to the specified filename.
void saveScreenshot(const char *filename)
{
  unsigned char *screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else
    cout << "Failed to save file " << filename << '.' << endl;

  delete[] screenshotData;
}

void idleFunc()
{
  // Static variable to count frame rate
  static int counter = 3;
  // Static variable to count number of screenshots
  static int screenshotCounter = 0;

  // Frequency of screenshots: Capture a screenshot every 10 frames
  const int frequency = 10;

  // Increment the counter
  counter++;

  // Check if it's time to take a screenshot and that the max number of screenshots has not been exceeded
  if ((counter % frequency == 0) && (screenshotCounter < 1001))
  {

    // Format the screenshot filename with the current screenshot number as specified in instructions
    char filename[128];
    sprintf(filename, "%03d.jpg", screenshotCounter);

    // Call saveScreenshot helper function with appropriate filename
    saveScreenshot(filename);

    // Increment number of screenshots saved
    screenshotCounter++;
  }

  // Notify GLUT that it should call displayFunc.
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  // When the window has been resized, we need to re-set our projection matrix.
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.LoadIdentity();
  // You need to be careful about setting the zNear and zFar.
  // Anything closer than zNear, or further than zFar, will be culled.
  const float zNear = 0.1f;
  const float zFar = 1000.0f;
  const float humanFieldOfView = 60.0f;
  matrix.Perspective(humanFieldOfView, 1.0f * w / h, zNear, zFar);
}

void mouseMotionDragFunc(int x, int y)
{
  // Mouse has moved, and one of the mouse buttons is pressed (dragging).

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = {x - mousePos[0], y - mousePos[1]};

  switch (controlState)
  {
  // translate the terrain
  case TRANSLATE:
    if (leftMouseButton)
    {
      // control x,y translation via the left mouse button
      terrainTranslate[0] += mousePosDelta[0] * 0.01f;
      terrainTranslate[1] -= mousePosDelta[1] * 0.01f;
    }
    if (middleMouseButton)
    {
      // control z translation via the middle mouse button
      terrainTranslate[2] += mousePosDelta[1] * 0.01f;
    }
    break;

  // rotate the terrain
  case ROTATE:
    if (leftMouseButton)
    {
      // control x,y rotation via the left mouse button
      terrainRotate[0] += mousePosDelta[1];
      terrainRotate[1] += mousePosDelta[0];
    }
    if (middleMouseButton)
    {
      // control z rotation via the middle mouse button
      terrainRotate[2] += mousePosDelta[1];
    }
    break;

  // scale the terrain
  case SCALE:
    if (leftMouseButton)
    {
      // control x,y scaling via the left mouse button
      terrainScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
      terrainScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
    }
    if (middleMouseButton)
    {
      // control z scaling via the middle mouse button
      terrainScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
    }
    break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // Mouse has moved.
  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void loadSpline(char *argv)
{
  FILE *fileSpline = fopen(argv, "r");
  if (fileSpline == NULL)
  {
    printf("Cannot open file %s.\n", argv);
    exit(1);
  }

  // Read the number of spline control points.
  fscanf(fileSpline, "%d\n", &spline.numControlPoints);
  printf("Detected %d control points.\n", spline.numControlPoints);

  // Allocate memory.
  spline.points = (Point *)malloc(spline.numControlPoints * sizeof(Point));
  // Load the control points.
  for (int i = 0; i < spline.numControlPoints; i++)
  {
    if (fscanf(fileSpline, "%lf %lf %lf",
               &spline.points[i].x,
               &spline.points[i].y,
               &spline.points[i].z) != 3)
    {
      printf("Error: incorrect number of control points in file %s.\n", argv);
      exit(1);
    }
  }
}

// Multiply C = A * B, where A is a m x p matrix, and B is a p x n matrix.
// All matrices A, B, C must be pre-allocated (say, using malloc or similar).
// The memory storage for C must *not* overlap in memory with either A or B.
// That is, you **cannot** do C = A * C, or C = C * B. However, A and B can overlap, and so C = A * A is fine, as long as the memory buffer for A is not overlaping in memory with that of C.
// Very important: All matrices are stored in **column-major** format.
// Example. Suppose
//      [ 1 8 2 ]
//  A = [ 3 5 7 ]
//      [ 0 2 4 ]
//  Then, the storage in memory is
//   1, 3, 0, 8, 5, 2, 2, 7, 4.
void MultiplyMatrices(int m, int p, int n, const double *A, const double *B, double *C)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double entry = 0.0;
      for (int k = 0; k < p; k++)
        entry += A[k * m + i] * B[j * p + k];
      C[m * j + i] = entry;
    }
  }
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // A mouse button has has been pressed or depressed.

  // Keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables.
  switch (button)
  {
  case GLUT_LEFT_BUTTON:
    leftMouseButton = (state == GLUT_DOWN);
    break;

  case GLUT_MIDDLE_BUTTON:
    middleMouseButton = (state == GLUT_DOWN);
    break;

  case GLUT_RIGHT_BUTTON:
    rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // Keep track of whether CTRL and SHIFT keys are pressed.
  switch (glutGetModifiers())
  {

  // Unused case due to M1 Macbook Pro, instead bound to the 't' key
  case GLUT_ACTIVE_CTRL:
    controlState = TRANSLATE;
    break;

  case GLUT_ACTIVE_SHIFT:
    controlState = SCALE;
    break;

  // If CTRL and SHIFT are not pressed, we are in rotate mode.
  default:
    controlState = ROTATE;
    break;
  }

  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
  case 27:   // ESC key
    exit(0); // exit the program
    break;

  case ' ':
    cout << "You pressed the spacebar." << endl;
    pause = !pause; // Boolean toggled to pause or unpause camera movement along track
    break;

  case 't':
    controlState = TRANSLATE; // Invoke translation here instead of 'ctrl' key
    break;

  case 'x':
    // Take a screenshot.
    saveScreenshot("screenshot.jpg");
    break;
  }
}

// Helper function defined in order to calculate the frenet frame of a position, given a NORMALIZED tangent vector
void calculateFrenetFrames(glm::vec3 tangentVec)
{
  // Set a constant small float epsilon to check edge cases of calculation
  const float EPSILON = 1e-6;
  // Create glm::vec3 to store the Frenet Frame
  glm::vec3 tangent = tangentVec;
  glm::vec3 normal;
  glm::vec3 binormal;

  // Set a boolean depending on whether the tangent vector is (0,0,-1), a specific edge case for splines goodRide.sp and rollercoaster.sp
  bool isTangentVertical = std::abs(tangent.x) < EPSILON &&
                           std::abs(tangent.y) < EPSILON &&
                           std::abs(tangent.z + 1) < EPSILON;

  // Considers whether tangent is vertical to calculate Frenet Frame
  if (isTangentVertical)
  {
    // Sets edge case normal and binormal that are orthogonal to (0,0,-1)
    normal = glm::vec3(1, 0, 0);
    binormal = glm::vec3(0, -1, 0);
  }
  else
  {
    // Considers if this is the first Frenet Frame
    if (cameraU == 0)
    {
      // Initializes a random vector to calculate normal
      glm::vec3 random(0, 1, 0);
      // Uses tangent and random vector to calculate first normal
      normal = glm::cross(tangent, random);
    }
    else
    {
      // Calculates normal based on previous binormal and current tangent
      normal = glm::cross(cameraB, tangent);
    }
    // Normalizes the normal to ensure it is a unit vector
    normal = glm::normalize(normal);

    // Calculates the binormal based on tangent and normal
    binormal = glm::cross(tangent, normal);
    // Normalizes the binormal to ensure it is a unit vector
    binormal = glm::normalize(binormal);
    // Sets global variable cameraB to the current binormal to be used for the next Frenet Frame
    cameraB = binormal;
  }

  // Adds Frenet Frame vectors to respective vectors to be accessed for track generation and camera movement
  splineTangents.push_back(tangent);
  splineNormals.push_back(normal);
  splineBinormals.push_back(binormal);
}

void displayFunc()
{
  // This function performs the actual rendering.
  // First, clear the screen.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set up the camera position, focus point, and the up vector.
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.LoadIdentity();

  // Obtain current spline position and Frenet Frame from global vectors, using a constantly incremented iterator cameraU
  glm::vec3 position = splinePositions[cameraU];
  glm::vec3 tangent = splineTangents[cameraU];
  glm::vec3 normal = splineNormals[cameraU];
  glm::vec3 binormal = splineBinormals[cameraU];

  // Create a look at matrix
  matrix.LookAt(position.x - binormal.x, position.y - binormal.y, position.z - binormal.z, // Position of camera is at the position of the spline, offset in the negative binormal direction (up vector)
                position.x + tangent.x, position.y + tangent.y, position.z + tangent.z,    // Look at of camera is at the position of the spline, offset along the tangent (along the track)
                -binormal.x, -binormal.y, -binormal.z);                                    // Up vector of camera is negative binormal (this aligns with my world-space orientation)

  // Considers whether the camera has seen the entirety of the track, and if the camera motion has been paused by user
  if (cameraU < numVerticesLines - 1 && !pause)
  {
    // Increments cameraU by 1, to move the camera 1 points along the track
    cameraU += 3;
  }

  // The following section of the code sets uniform variables for the Phong shaders

  // Bind the appropriate pipelineProgram for Phong shaders
  pipelineProgram->Bind();

  // Initialize the viewLightDirection vector
  float view[16];
  matrix.GetMatrix(view);

  glm::mat4 viewMatrix = glm::mat4(
      view[0], view[1], view[2], view[3],
      view[4], view[5], view[6], view[7],
      view[8], view[9], view[10], view[11],
      view[12], view[13], view[14], view[15]);

  glm::vec4 lightDirection(0.0f, 1.0f, 0.0f, 0.0f);
  glm::vec4 viewLightDirectionVec = viewMatrix * lightDirection;

  float *viewLightDirection = (float *)malloc(3 * sizeof(float));
  viewLightDirection[0] = viewLightDirectionVec.x;
  viewLightDirection[1] = viewLightDirectionVec.y;
  viewLightDirection[2] = viewLightDirectionVec.z;

  // Set viewLightDirection uniform variable
  pipelineProgram->SetUniformVariable3fv("viewLightDirection", viewLightDirection);

  // Initialize light ambience, based on piazza post @173
  float *La = (float *)malloc(4 * sizeof(float));
  La[0] = 1.0f;
  La[1] = 1.0f;
  La[2] = 1.0f;
  La[3] = 1.0f;

  // Set light ambience uniform variable
  pipelineProgram->SetUniformVariable4fv("La", La);

  // Initialize light diffuse, based on piazza post @173
  float *Ld = (float *)malloc(4 * sizeof(float));
  Ld[0] = 1.0f;
  Ld[1] = 1.0f;
  Ld[2] = 1.0f;
  Ld[3] = 1.0f;

  // Set light diffuse uniform variable
  pipelineProgram->SetUniformVariable4fv("Ld", Ld);

  // Initialize light specular, based on piazza post @173
  float *Ls = (float *)malloc(4 * sizeof(float));
  Ls[0] = 1.0f;
  Ls[1] = 1.0f;
  Ls[2] = 1.0f;
  Ls[3] = 1.0f;

  // Set light specular uniform variable
  pipelineProgram->SetUniformVariable4fv("Ls", Ls);

  // Initialize mesh ambient (experimentation and calculations to obtain a gold color shading)
  float *ka = (float *)malloc(4 * sizeof(float));
  ka[0] = 0.24725f;
  ka[1] = 0.1995f;
  ka[2] = 0.0745f;
  ka[3] = 1.0f;

  // Set mesh ambient uniform variable
  pipelineProgram->SetUniformVariable4fv("ka", ka);

  // Initialize mesh diffuse (experimentation and calculations to obtain a gold color shading)
  float *kd = (float *)malloc(4 * sizeof(float));
  kd[0] = 0.75164f;
  kd[1] = 0.60648f;
  kd[2] = 0.22648f;
  kd[3] = 1.0f;

  // Set mesh diffuse uniform variable
  pipelineProgram->SetUniformVariable4fv("kd", kd);

  // Initialize mesh specular (experimentation and calculations to obtain a gold color shading)
  float *ks = (float *)malloc(4 * sizeof(float));
  ks[0] = 0.628281f;
  ks[1] = 0.555802f;
  ks[2] = 0.366065f;
  ks[3] = 1.0f;

  // Set mesh specular uniform variable
  pipelineProgram->SetUniformVariable4fv("ks", ks);

  // Initialize alpha (experimentation and calculations to obtain a gold color shading)
  float alpha = 51.2f;

  // Set alpha uniform variable
  pipelineProgram->SetUniformVariablef("alpha", alpha);

  // Additional modeling performed on the object: translations, rotations and scales.
  matrix.Translate(terrainTranslate[0], terrainTranslate[1], terrainTranslate[2]);
  matrix.Rotate(terrainRotate[0], 1.0f, 0.0f, 0.0f); // Rotation on x-Axis
  matrix.Rotate(terrainRotate[1], 0.0f, 1.0f, 0.0f); // Rotation on y-Axis
  matrix.Rotate(terrainRotate[2], 0.0f, 0.0f, 1.0f); // Rotation on z-Axis
  matrix.Scale(terrainScale[0], terrainScale[1], terrainScale[2]);

  // Read the current modelview and projection matrices from our helper class.
  float modelViewMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetMatrix(modelViewMatrix);

  float projectionMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.GetMatrix(projectionMatrix);

  // Create normal matrix based on modelView matrix from our helper class
  float normalMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetNormalMatrix(normalMatrix);

  // Upload the modelview and projection matrices to the GPU. Note that these are "uniform" variables.
  pipelineProgram->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  pipelineProgram->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);
  pipelineProgram->SetUniformVariableMatrix4fv("normalMatrix", GL_FALSE, normalMatrix);

  // Execute the rendering.
  // Bind each object's VAO and draw, utilized different drawing modes based on experimentation
  vaoLeftTrack->Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, numVerticesTrack);
  vaoRightTrack->Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, numVerticesTrack);
  vaoTie->Bind();
  glDrawArrays(GL_TRIANGLES, 0, numVerticesTie);

  // The following section of code initializes uniform variables for the texture shaders

  // Bind the appropriate pipelineProgram for the sky box and ground
  texPipelineProgram->Bind();

  // Set modelView and projection uniform variables based on above calculations
  texPipelineProgram->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  texPipelineProgram->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);

  // Bind the desired texture to draw with (I have a commented out one to allow for me to use both in different cases)
  glBindTexture(GL_TEXTURE_2D, groundTexture);
  // glBindTexture(GL_TEXTURE_2D, skyTexture);

  // Bind each object's VAO and draw, utilized GL_TRIANGLES in order to render both the skybox and ground
  vaoGround->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  vaoSkyFront->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  vaoSkyBack->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  vaoSkyLeft->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  vaoSkyRight->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  vaoSkyTop->Bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // Swap the double-buffers.
  glutSwapBuffers();
}

int initTexture(const char *imageFilename, GLuint textureHandle)
{
  // Read the texture image.
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK)
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // Check that the number of bytes is a multiple of 4.
  if (img.getWidth() * img.getBytesPerPixel() % 4)
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // Allocate space for an array of pixels.
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char *pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // Fill the pixelsRGBA array with the image pixels.
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++)
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0;   // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0;   // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0;   // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // Bind the texture.
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // Initialize the texture.
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // Generate the mipmaps for this texture.
  glGenerateMipmap(GL_TEXTURE_2D);

  // Set the texture parameters.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Query support for anisotropic texture filtering.
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // Set anisotropic texture filtering.
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

  // Query for any errors.
  GLenum errCode = glGetError();
  if (errCode != 0)
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }

  // De-allocate the pixel array -- it is no longer needed.
  delete[] pixelsRGBA;

  return 0;
}

void initScene(int argc, char *argv[])
{
  // Set the background color.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black color.

  // Enable z-buffering (i.e., hidden surface removal using the z-buffer algorithm).
  glEnable(GL_DEPTH_TEST);

  // Create a pipeline program for the Phong shaders
  pipelineProgram = new PipelineProgram(); // Load and set up the pipeline program, including its shaders.
  // Load and set up the pipeline program, including its shaders.
  if (pipelineProgram->BuildShadersFromFiles(shaderBasePath, "vertexShader.glsl", "fragmentShader.glsl") != 0)
  {
    cout << "Failed to build the coaster pipeline program." << endl;
    throw 1;
  }
  cout << "Successfully built the coaster pipeline program." << endl;

  // Bind the pipeline program for the Phong shaders in order to set the in variables
  pipelineProgram->Bind();

  // Initialize constant for splineBasis calculation
  double s = 0.5000;

  // Generation of column-major splineBasis matrix
  double splineBasis[16] = {
      -s, 2 * s, -s, 0,
      2 - s, s - 3, 0, 1,
      s - 2, 3 - 2 * s, s, 0,
      s, -s, 0, 0};

  // Calculate the number of segments based on the amount of control points
  int numSegments = spline.numControlPoints - 3;
  // Calculate the number of vertices generated for the spline based on step size of u = 0.001
  numVerticesLines = 1000 * numSegments;

  // Create variables in order to accurately calculate spline positions, and the corresponding Frenet Frames
  int controlIndexIterator = 0;        // Used to iterate through the controlMatrix for each segment
  double specificPoint[3] = {0};       // Used to store the final calculated point
  double tangent[3] = {0};             // Used to store the final calculated tangent
  double uMatrix[4] = {0};             // Used to store the uMatrix for point calculation
  double uTangent[4] = {0};            // Used to store the uMatrix for tangent calculation
  double firstMatrixProduct[4] = {0};  // Used to store the first product of matrix multiplication for point calculation
  double firstTangentProduct[4] = {0}; // Used to store the first product of matrix multiplication for tangent calculation
  double controlMatrix[12] = {0};      // Used to store the controlMatrix for the current segment
  int indexPos = 0;
  int indexCol = 0;

  // Iterate through all segments in the spline
  for (int segment = 0; segment <= numSegments; segment++)
  {

    // For each segment, use controlPoints in order to create the corresponding controlMatrix in column-major form
    for (int i = 0; i < 4; i++)
    {
      int controlIndex = controlIndexIterator + i;
      controlMatrix[i] = spline.points[controlIndex].x;     // X coordinate
      controlMatrix[i + 4] = spline.points[controlIndex].y; // Y coordinate
      controlMatrix[i + 8] = spline.points[controlIndex].z; // Z coordinate
    }

    // Increment u from 0 to 1, with step size of 0.001 in order to calculate points
    for (double u = 0; u <= 1; u += 0.001)
    {

      // Create uMatrix using current u value
      uMatrix[0] = pow(u, 3);
      uMatrix[1] = pow(u, 2);
      uMatrix[2] = u;
      uMatrix[3] = 1;

      // Matrix multiplication of u matrix with spline basis, stored in firstMatrixProduct
      MultiplyMatrices(1, 4, 4, uMatrix, splineBasis, firstMatrixProduct);
      // Matrix multiplication of firstMatrixPrdouct with control matrix to calculate specific point
      MultiplyMatrices(1, 4, 3, firstMatrixProduct, controlMatrix, specificPoint);

      // Create a glm::vec3 with specific point's values and store it in splinePositions for camera movement and track generation
      splinePositions.push_back(glm::vec3(specificPoint[0], specificPoint[1], specificPoint[2]));

      // Create uTangent using current u value (a differentiated form of the uMatrix)
      uTangent[0] = (double)3 * pow(u, 2);
      uTangent[1] = (double)2 * u;
      uTangent[2] = (double)1;
      uTangent[3] = (double)0;

      // Matrix multiplication of differentiated u matrix with spline basis, stored in firstTangentProduct
      MultiplyMatrices(1, 4, 4, uTangent, splineBasis, firstTangentProduct);
      // Matrix multiplication of firstTangentProduct with control matrix to calculate tangent
      MultiplyMatrices(1, 4, 3, firstTangentProduct, controlMatrix, tangent);

      // Store the tangent's values in a glm::vec3 for ease in calculations
      glm::vec3 tangentVec((double)tangent[0], (double)tangent[1], (double)tangent[2]);
      // Normalize tangent to ensure it is a unit vector
      tangentVec = glm::normalize(tangentVec);
      // Call helper function in order to calculate corresponding normal and binormal, as well as to store the frenetFrame in global vectors
      calculateFrenetFrames(tangentVec);
    }
    // Iterate to next control matrix for the next segment
    controlIndexIterator++;
  }

  // To generate the cross section, each point in the spline will now be represented by a quad, made up of 2 triangles
  numVerticesTrack = numVerticesLines * 6;

  // The following section generates the left side of the two parallel tracks that follow the spline
  // Create float arrays to store the positions of the left side of the track, and the normals for Phong shading
  float *positionsLeftTrack = (float *)malloc(numVerticesTrack * 3 * sizeof(float));
  float *normalsLeftTrack = (float *)malloc(numVerticesTrack * 3 * sizeof(float));
  // Create a vector to store the appropriate trackNormals in order to ensure appropriate Phong shading for track ties
  std::vector<glm::vec3> trackNormals;

  int indexPosTrack = 0;
  int indexNormTrack = 0;

  // Experimented with values until tracks were not intersecting with one another and visually appealing
  float widthScale = 0.04f;
  float heightScale = 0.06f;
  float trackOffset = 0.1f;

  // Iterate through all the points in the spline
  for (int i = 0; i < splinePositions.size(); i++)
  {

    // Get current point from global splinePositions vector
    glm::vec3 currPosition(splinePositions[i].x, splinePositions[i].y, splinePositions[i].z);

    // Generate vertexes for cross section, using provided PDF with modifications to account for -binormal as my world's up vector
    // Accounted for a track offset in order to create two parallel tracks, this is the left track
    glm::vec3 vertex0 = currPosition - widthScale * splineNormals[i] - (trackOffset * splineNormals[i]) + heightScale * splineBinormals[i];
    glm::vec3 vertex1 = currPosition + widthScale * splineNormals[i] - (trackOffset * splineNormals[i]) + heightScale * splineBinormals[i];
    glm::vec3 vertex2 = currPosition + widthScale * splineNormals[i] - (trackOffset * splineNormals[i]) - heightScale * splineBinormals[i];
    glm::vec3 vertex3 = currPosition - widthScale * splineNormals[i] - (trackOffset * splineNormals[i]) - heightScale * splineBinormals[i];

    // Populate positionsLeftTrack with Triangle 1 of the quad
    positionsLeftTrack[indexPosTrack++] = vertex0.x;
    positionsLeftTrack[indexPosTrack++] = vertex0.y;
    positionsLeftTrack[indexPosTrack++] = vertex0.z;

    positionsLeftTrack[indexPosTrack++] = vertex1.x;
    positionsLeftTrack[indexPosTrack++] = vertex1.y;
    positionsLeftTrack[indexPosTrack++] = vertex1.z;

    positionsLeftTrack[indexPosTrack++] = vertex2.x;
    positionsLeftTrack[indexPosTrack++] = vertex2.y;
    positionsLeftTrack[indexPosTrack++] = vertex2.z;

    // Populate positionsLeftTrack with Triangle 2 of the quad
    positionsLeftTrack[indexPosTrack++] = vertex2.x;
    positionsLeftTrack[indexPosTrack++] = vertex2.y;
    positionsLeftTrack[indexPosTrack++] = vertex2.z;

    positionsLeftTrack[indexPosTrack++] = vertex3.x;
    positionsLeftTrack[indexPosTrack++] = vertex3.y;
    positionsLeftTrack[indexPosTrack++] = vertex3.z;

    positionsLeftTrack[indexPosTrack++] = vertex0.x;
    positionsLeftTrack[indexPosTrack++] = vertex0.y;
    positionsLeftTrack[indexPosTrack++] = vertex0.z;

    // Utilize glm::vec3 in order to determine two edges of the quad face
    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex2 - vertex1;

    // Determine the normal of the quad face in order to populate normalsLeftTrack for phong shading
    glm::vec3 normal = glm::cross(edge1, edge2);
    normal = glm::normalize(normal);
    trackNormals.push_back(normal);

    // Populate normalsLeftTrack with the calculated normal for each vertex in the quad face
    for (int j = 0; j < 6; j++)
    {
      normalsLeftTrack[indexNormTrack++] = normal.x;
      normalsLeftTrack[indexNormTrack++] = normal.y;
      normalsLeftTrack[indexNormTrack++] = normal.z;
    }
  }

  // Create VBOs to store the positions and normals of the left track
  vboVerticesLeftTrack = new VBO(numVerticesTrack, 3, positionsLeftTrack, GL_STATIC_DRAW); // 3 values per position
  vboNormalsLeftTrack = new VBO(numVerticesTrack, 3, normalsLeftTrack, GL_STATIC_DRAW);

  // Create the VAO
  vaoLeftTrack = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO.
  vaoLeftTrack->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboVerticesLeftTrack, "position");
  // Set up the relationship between the "normal" shader variable and the VAO.
  vaoLeftTrack->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboNormalsLeftTrack, "normal");

  // The following section generates the right side of the two parallel tracks that follow the spline
  // Create float arrays to store the positions of the right side of the track, and the normals for Phong shading
  float *positionsRightTrack = (float *)malloc(numVerticesTrack * 3 * sizeof(float));
  float *normalsRightTrack = (float *)malloc(numVerticesTrack * 3 * sizeof(float));

  indexPosTrack = 0;
  indexNormTrack = 0;

  // Iterate through all the points in the spline
  for (int i = 0; i < splinePositions.size(); i++)
  {

    // Get current point from global splinePositions vector
    glm::vec3 currPosition(splinePositions[i].x, splinePositions[i].y, splinePositions[i].z);

    // Generate vertexes for cross section, using provided PDF with modifications to account for -binormal as my world's up vector
    // Accounted for a track offset in order to create two parallel tracks, this is the left track
    glm::vec3 vertex0 = currPosition - widthScale * splineNormals[i] + (trackOffset * splineNormals[i]) + heightScale * splineBinormals[i];
    glm::vec3 vertex1 = currPosition + widthScale * splineNormals[i] + (trackOffset * splineNormals[i]) + heightScale * splineBinormals[i];
    glm::vec3 vertex2 = currPosition + widthScale * splineNormals[i] + (trackOffset * splineNormals[i]) - heightScale * splineBinormals[i];
    glm::vec3 vertex3 = currPosition - widthScale * splineNormals[i] + (trackOffset * splineNormals[i]) - heightScale * splineBinormals[i];

    // Populate positionsRightTrack with Triangle 1 of the quad
    positionsRightTrack[indexPosTrack++] = vertex0.x;
    positionsRightTrack[indexPosTrack++] = vertex0.y;
    positionsRightTrack[indexPosTrack++] = vertex0.z;

    positionsRightTrack[indexPosTrack++] = vertex1.x;
    positionsRightTrack[indexPosTrack++] = vertex1.y;
    positionsRightTrack[indexPosTrack++] = vertex1.z;

    positionsRightTrack[indexPosTrack++] = vertex2.x;
    positionsRightTrack[indexPosTrack++] = vertex2.y;
    positionsRightTrack[indexPosTrack++] = vertex2.z;

    // Populate positionsRightTrack with Triangle 2 of the quad
    positionsRightTrack[indexPosTrack++] = vertex2.x;
    positionsRightTrack[indexPosTrack++] = vertex2.y;
    positionsRightTrack[indexPosTrack++] = vertex2.z;

    positionsRightTrack[indexPosTrack++] = vertex3.x;
    positionsRightTrack[indexPosTrack++] = vertex3.y;
    positionsRightTrack[indexPosTrack++] = vertex3.z;

    positionsRightTrack[indexPosTrack++] = vertex0.x;
    positionsRightTrack[indexPosTrack++] = vertex0.y;
    positionsRightTrack[indexPosTrack++] = vertex0.z;

    // Utilize glm::vec3 in order to determine two edges of the quad face
    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex2 - vertex1;

    // Determine the normal of the quad face in order to populate normalsRight for phong shading
    glm::vec3 normal = glm::cross(edge1, edge2);
    normal = glm::normalize(normal);

    // Populate normalsLeftTrack with the previously calculated normal for each vertex in the quad face
    for (int j = 0; j < 6; j++)
    {
      normalsRightTrack[indexNormTrack++] = normal.x;
      normalsRightTrack[indexNormTrack++] = normal.y;
      normalsRightTrack[indexNormTrack++] = normal.z;
    }
  }

  // Create VBOs to store the positions and normals of the right track
  vboVerticesRightTrack = new VBO(numVerticesTrack, 3, positionsRightTrack, GL_STATIC_DRAW); // 3 values per position
  vboNormalsRightTrack = new VBO(numVerticesTrack, 3, normalsRightTrack, GL_STATIC_DRAW);

  // Create the VAO
  vaoRightTrack = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO.
  vaoRightTrack->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboVerticesRightTrack, "position");
  // Set up the relationship between the "normal" shader variable and the VAO.
  vaoRightTrack->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboNormalsRightTrack, "normal");

  // The following section generates the ties perpendicular to the tracks
  // Calculate the number of vertices to generate all the ties as quads (2 triangles) - 1000 to prevent extra ties, and divide by tie spacing
  float tieSpacing = 100;
  numVerticesTie = (splinePositions.size() - 1000) / tieSpacing * 6;

  // Create float arrays to store the positions of the track ties, and the normals for Phong shading
  float *positionsTie = (float *)malloc(numVerticesTie * 3 * sizeof(float));
  float *normalsTie = (float *)malloc(numVerticesTie * 3 * sizeof(float));

  int indexPosTie = 0;
  int indexNormTie = 0;

  // Parameters that determine the appearance of the ties
  float tieWidth = trackOffset * 2 + 0.2f; // This refers to the width of the tie in the perpendicicular direction to the tracks
  float tieExtension = 0.1f;               // This refers to the width of the tie in the direction of the tracks

  // Iterate through all the positions on the spline except the last 1000 (to prevent tie overflow past the end of the tracks), incrementing by 100
  for (int i = 0; i < splinePositions.size() - 1000; i += 100)
  {
    // Utilize global variables to obtain spline position, and Frenet Frame
    glm::vec3 currPosition = splinePositions[i];
    glm::vec3 tangent = splineTangents[i];
    glm::vec3 binormal = -splineBinormals[i];
    glm::vec3 normal = splineNormals[i];

    // Determine the center position of the tie, using experimentation for heightPosition
    float heightPosition = 0.06f;
    glm::vec3 tieCenter = currPosition + (binormal * heightPosition);

    // Determine the vertex position of the tie quad, using experimentation for widthFactor
    float widthFactor = 0.5f;
    glm::vec3 vertex0 = tieCenter - (normal * tieWidth * widthFactor);
    glm::vec3 vertex1 = tieCenter + (normal * tieWidth * widthFactor);
    glm::vec3 vertex2 = vertex0 + (tangent * tieExtension);
    glm::vec3 vertex3 = vertex1 + (tangent * tieExtension);

    // Populate positionsTie with Triangle 1 of the quad
    positionsTie[indexPosTie++] = vertex0.x;
    positionsTie[indexPosTie++] = vertex0.y;
    positionsTie[indexPosTie++] = vertex0.z;

    positionsTie[indexPosTie++] = vertex1.x;
    positionsTie[indexPosTie++] = vertex1.y;
    positionsTie[indexPosTie++] = vertex1.z;

    positionsTie[indexPosTie++] = vertex2.x;
    positionsTie[indexPosTie++] = vertex2.y;
    positionsTie[indexPosTie++] = vertex2.z;

    // Populate positionsTie with Triangle 2 of the quad
    positionsTie[indexPosTie++] = vertex1.x;
    positionsTie[indexPosTie++] = vertex1.y;
    positionsTie[indexPosTie++] = vertex1.z;

    positionsTie[indexPosTie++] = vertex3.x;
    positionsTie[indexPosTie++] = vertex3.y;
    positionsTie[indexPosTie++] = vertex3.z;

    positionsTie[indexPosTie++] = vertex2.x;
    positionsTie[indexPosTie++] = vertex2.y;
    positionsTie[indexPosTie++] = vertex2.z;

    // Utilize glm::vec3 in order to determine two edges of the quad face
    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex3 - vertex1;

    // Determine the normal of the quad face in order to populate normalsTie for phong shading
    glm::vec3 quadNormal = glm::cross(edge1, edge2);
    quadNormal = glm::normalize(quadNormal);

    // Since the geometry of the quad face uses -binormal as the up vector, calculate -binormal
    glm::vec3 quadBinormal = glm::cross(quadNormal, edge1);
    // Set up the relationship between the "normal" shader variable and the VAO.
    quadBinormal = -glm::normalize(quadBinormal);

    // Set -binormal as the normal for each of the vertices in the quad
    for (int j = 0; j < 6; j++)
    {
      normalsTie[indexNormTie++] = quadBinormal.x;
      normalsTie[indexNormTie++] = quadBinormal.y;
      normalsTie[indexNormTie++] = quadBinormal.z;
    }
  }

  // Create VBOs to store the positions and normals of the tie
  vboVerticesTie = new VBO(numVerticesTie, 3, positionsTie, GL_STATIC_DRAW);
  vboNormalsTie = new VBO(numVerticesTie, 3, normalsTie, GL_STATIC_DRAW);

  // Create the VAO
  vaoTie = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO.
  vaoTie->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboVerticesTie, "position");
  // Set up the relationship between the "normal" shader variable and the VAO.
  vaoTie->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboNormalsTie, "normal");

  // We don't need this data any more, as we have already uploaded it to the VBO. And so we can destroy it, to avoid a memory leak.
  free(positionsLeftTrack);
  free(normalsLeftTrack);
  free(positionsRightTrack);
  free(normalsRightTrack);
  free(positionsTie);
  free(normalsTie);

  // The following section generates the sky box
  // Create a pipeline program for the texture shaders
  texPipelineProgram = new PipelineProgram(); // Load and set up the pipeline program, including its shaders.
  // Load and set up the pipeline program, including its shaders.
  if (texPipelineProgram->BuildShadersFromFiles(shaderBasePath, "textureVertexShader.glsl", "textureFragmentShader.glsl") != 0)
  {
    cout << "Failed to build the texture pipeline program." << endl;
    throw 1;
  }
  cout << "Successfully built the texture pipeline program." << endl;

  // Create a float to store the size dimensions of the cube
  float size = 100.f;

  // Create a float array to store the positions of the ground face
  float positionsGround[] = {
      -size, -size, -size,
      -size, -size, size,
      size, -size, -size,
      -size, -size, size,
      size, -size, size,
      size, -size, -size};

  // Create a basic texture mapping for 6 vertices, these will be applied to all the faces
  float textureCoords[] = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 1.0f,
      1.0f, 0.0f};

  // Create VBOs to store the positions and texture coordinates for the ground face
  vboVerticesGround = new VBO(6, 3, positionsGround, GL_STATIC_DRAW);
  vboTextureGround = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO
  vaoGround = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoGround->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesGround, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoGround->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureGround, "texCoord");

  // Create a float array to store the positions of the top face
  float positionsSkyTop[] = {
      -size, size, -size,
      -size, size, size,
      size, size, -size,
      -size, size, size,
      size, size, size,
      size, size, -size};

  // Create VBOs to store the positions and texture coordinates for the top face
  vboVerticesSkyTop = new VBO(6, 3, positionsSkyTop, GL_STATIC_DRAW);
  vboTextureSkyTop = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO  
  vaoSkyTop = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoSkyTop->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesSkyTop, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoSkyTop->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureSkyTop, "texCoord");

  // Create a float array to store the positions of the front face
  float positionsSkyFront[] = {
      -size, -size, size,
      -size, size, size,
      size, -size, size,
      -size, size, size,
      size, size, size,
      size, -size, size};

  // Create VBOs to store the positions and texture coordinates for the top face
  vboVerticesSkyFront = new VBO(6, 3, positionsSkyFront, GL_STATIC_DRAW);
  vboTextureSkyFront = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO  
  vaoSkyFront = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoSkyFront->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesSkyFront, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoSkyFront->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureSkyFront, "texCoord");

  // Create a float array to store the positions of the back face
  float positionsSkyBack[] = {
      -size, -size, -size,
      -size, size, -size,
      size, -size, -size,
      -size, size, -size,
      size, size, -size,
      size, -size, -size};

  // Create VBOs to store the positions and texture coordinates for the back face
  vboVerticesSkyBack = new VBO(6, 3, positionsSkyBack, GL_STATIC_DRAW);
  vboTextureSkyBack = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO  
  vaoSkyBack = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoSkyBack->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesSkyBack, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoSkyBack->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureSkyBack, "texCoord");

  // Create a float array to store the positions of the left face
  float positionsSkyLeft[] = {
      -size, -size, -size,
      -size, -size, size,
      -size, size, -size,
      -size, -size, size,
      -size, size, size,
      -size, size, -size};

  // Create VBOs to store the positions and texture coordinates for the left face
  vboVerticesSkyLeft = new VBO(6, 3, positionsSkyLeft, GL_STATIC_DRAW);
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vboTextureSkyLeft = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO  
  vaoSkyLeft = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoSkyLeft->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesSkyLeft, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoSkyLeft->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureSkyLeft, "texCoord");

  // Create a float array to store the positions of the right face
  float positionsSkyRight[] = {
      size, -size, size,
      size, -size, -size,
      size, size, size,
      size, -size, -size,
      size, size, -size,
      size, size, size};

  // Create VBOs to store the positions and texture coordinates for the right face
  vboVerticesSkyRight = new VBO(6, 3, positionsSkyRight, GL_STATIC_DRAW);
  vboTextureSkyRight = new VBO(6, 2, textureCoords, GL_STATIC_DRAW);

  // Create the VAO  
  vaoSkyRight = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO
  vaoSkyRight->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboVerticesSkyRight, "position");
  // Set up the relationship between the "texCoord" shader variable and the VAO
  vaoSkyRight->ConnectPipelineProgramAndVBOAndShaderVariable(texPipelineProgram, vboTextureSkyRight, "texCoord");

  // Check for any OpenGL errors.
  std::cout << "GL error status is: " << glGetError() << std::endl;
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    printf("Usage: %s <spline file>\n", argv[0]);
    exit(0);
  }

  // Load spline from the provided filename.
  loadSpline(argv[1]);

  printf("Loaded spline with %d control point(s).\n", spline.numControlPoints);

  cout << "Initializing GLUT..." << endl;
  glutInit(&argc, argv);

  cout << "Initializing OpenGL..." << endl;

#ifdef __APPLE__
  glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
#else
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
#endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  // Initialize the groundTexture handle
  glGenTextures(1, &groundTexture);
  int codeGround = initTexture("water.jpg", groundTexture);
  if (codeGround != 0)
  {
    printf("Error loading the ground texture image.\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the skyTexture handle
  glGenTextures(1, &skyTexture);
  int codeSky = initTexture("clouds.jpg", skyTexture);
  if (codeSky != 0)
  {
    printf("Error loading the sky texture image.\n");
    exit(EXIT_FAILURE);
  }

#ifdef __APPLE__
  // This is needed on recent Mac OS X versions to correctly display the window.
  glutReshapeWindow(windowWidth - 1, windowHeight - 1);
#endif

  // Tells GLUT to use a particular display function to redraw.
  glutDisplayFunc(displayFunc);
  // Perform animation inside idleFunc.
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

// init glew
#ifdef __APPLE__
  // nothing is needed on Apple
#else
  // Windows, Linux
  GLint result = glewInit();
  if (result != GLEW_OK)
  {
    cout << "error: " << glewGetErrorString(result) << endl;
    exit(EXIT_FAILURE);
  }
#endif

  // Perform the initialization.
  initScene(argc, argv);

  // Sink forever into the GLUT loop.
  glutMainLoop();

  return 0;
}