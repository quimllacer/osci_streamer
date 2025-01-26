# Purpose
The purpose of this repository is render images or animations as stereo sound. This stereo sound can be then visualized in an oscilloscope. Search for
> Oscilloscope music
on the internet for reference.

# Commands
The script initializes to the following state:
> f:mesh status:on name:octahedron\
> f:mic status:on\
> f:waves status:on n:0 wf:50 rate:0.001\
> f:rotx status:on n: 0, angle: 0, rate:0.0030\
> f:roty status:on n: 0, angle: 0, rate:0.0020\
> f:rotz status:on n: 0, angle: 0, rate:0.0010\

The user can change the state of the script by entering the following command in the terminal -while the script is running-:

> function parameter value

Here are a few command examples:
1. > mesh name cube
2. > waves status off
3. > rotx rate 0.01

Notes:
1. Microphone function only works when the script is executed from the terminal. This has to do with needing admin permissions.
2. Only one parameter can be changed at a time.
3. 3D objects to be rendered should have an .obj extension and be in the "files_to_render" folder. Their name is the one that corresponds to the "name" prameter for the "mesh" function. 

*****TO DO*****

+Display default meshes with rotation
+Change mesh every minute automatically
+RBP start code automatically upon reboot
+Power timer for safety
+Lower beam intensity automatically if audio stream stops
+Install front cables through the inside of the oscilloscope for a cleaner look
+Work on other animations