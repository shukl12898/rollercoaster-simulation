Subject 	: CSCI420 - Computer Graphics 
Assignment 2: Simulating a Roller Coaster
Author		: Akanksha Shukla
USC ID 		: 3111450282

Description: In this assignment, we use Catmull-Rom splines along with OpenGL core profile shader-based texture mapping and Phong shading to create a roller coaster simulation.

Core Credit Features: (Answer these Questions with Y/N; you can also insert comments as appropriate)
======================

1. Uses OpenGL core profile, version 3.2 or higher - 

2. Completed all Levels:
  Level 1 : - Y
  level 2 : - Y
  Level 3 : - Y
  Level 4 : - Y
  Level 5 : - Y

3. Rendered the camera at a reasonable speed in a continuous path/orientation - Y

4. Run at interactive frame rate (>15fps at 1280 x 720) - Y

5. Understandably written, well commented code - Y

6. Attached an Animation folder containing not more than 1000 screenshots - Y

7. Attached this ReadMe File - Y

Extra Credit Features: (Answer these Questions with Y/N; you can also insert comments as appropriate)
======================

1. Render a T-shaped rail cross section - Y

2. Render a Double Rail - Y

3. Made the track circular and closed it with C1 continuity - N

4. Any Additional Scene Elements? (list them here) - N

5. Render a sky-box - Y

6. Create tracks that mimic real world roller coaster - N

7. Generate track from several sequences of splines - N

8. Draw splines using recursive subdivision - N

9. Render environment in a better manner - N

10. Improved coaster normals - N

11. Modify velocity with which the camera moves - N

12. Derive the steps that lead to the physically realistic equation of updating u - N

Additional Features: (Please document any additional features you may have implemented other than the ones described above)
1. The spacebar key functions as a 'pause' button. It allows the user to pause or unpause the camera's motion along the track.

2. I have two versions of my animation, one with the "sky" texture and one with the "water" texture. Please feel free to view either one / both.

Open-Ended Problems: (Please document approaches to any open-ended problems that you have tackled)
1. My world space's orientation somehow resulted in the up vector being the -binormal along any given position on the spline. As a result, I had to modify various aspects of my code (differing from provided helper equations and resources) in order to obtain the desired visuals.
    - I used a differing look at matrix than reccommended
    - I had to generate my tracks and tie vertices differently than in the suggested reference file

2. I struggled with the z-buffering of the scene, causing slight glitches on my screen. To combat this, I tried to differentiate z-levels of all rendered artifacts as much as possible. I also tried modified zNear and zFar, but when it made no difference, I reset them to their inital values. 

Keyboard/Mouse controls: (Please document Keyboard/Mouse controls if any)
1. Space bar is a pause button, which allows for the camera motion to stop moving along the track
2. To translate the scene, you have to use the 't' button binding, since I have a mac device
3. The rest are the same as in hw1

Names of the .cpp files you made changes to:
1. hw1.cpp

Names of the .glsl files you made changes to:
1. vertexShader.glsl
2. fragmentShader.glsl
3. textureVertexShader.glsl
4. textureFragmentShader.glsl


Comments : (If any)

