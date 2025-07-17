A set of tools to visualize and interact with sequences of 3D data with cross-platform support on Windows, Linux, and macOS.

## Installation
Pease run the following command in this folder.
```commandline
pip install -e .
```
## Features
* Native Python interface, easy to use and hack.
* Load [SMPL[-H/-X]](https://smpl.is.tue.mpg.de/) / [MANO](https://mano.is.tue.mpg.de/) / [FLAME](https://flame.is.tue.mpg.de/) / [STAR](https://github.com/ahmedosman/STAR) sequences and display them in an interactive viewer.
* Headless mode for server rendering of videos/images.
* Remote mode for non-blocking integration of visualization code.
* Render 3D data on top of images via weak-perspective or OpenCV camera models.
* Animatable camera paths.
* Edit SMPL sequences and poses manually.
* Prebuilt renderable primitives (cylinders, spheres, point clouds, etc).
* Built-in extensible GUI (based on Dear ImGui).
* Export screenshots, videos and turntable views (as mp4/gif)
* High-Performance ModernGL-based rendering pipeline (running at 100fps+ on most laptops).

![aitviewer SMPL Editing](https://user-images.githubusercontent.com/5639197/188625764-351100e9-992e-430c-b170-69d4f142f5dd.gif)
![instruction](../assets/ait.png)

* `SPACE` Start/stop playing animation.
* `.` Go to next frame.
* `,` Go to previous frame.
* `G` Open a window to change frame by typing the frame number.
* `X` Center view on the selected object.
* `O` Enable/disable orthographic camera.
* `T` Show the camera target in the scene.
* `C` Save the camera position and orientation to disk.
* `L` Load the camera position and orientation from disk.
* `K` Lock the selection to the currently selected object.
* `S` Show/hide shadows.
* `D` Enabled/disable dark mode.
* `P` Save a screenshot to the the export/screenshots directory.
* `I` Change the viewer mode to inspect.
* `V` Change the viewer mode to view.
* `E` If a mesh is selected, show the edges of the mesh.
* `F` If a mesh is selected, switch between flat and smooth shading.
* `Z` Show a debug visualization of the object IDs.
* `ESC` Exit the viewer.

## TODO
- [ ] Online testing.

## Acknowledgement
This visualizer is developed based on [aitviewer](https://eth-ait.github.io/aitviewer).
