This script is used to create user controlled video textures.

## Resources About Video Textures
1.  [GA Tech Video Textures Resource](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/index.htm)
2.  [Video Textures - Original Paper](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf)

## What Is This Project?
This project is an implementation of the pipeline described in [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) to create mouse-controlled, video-based video sprite animations.

See section `7.4 Video-based animation`, subsection `Mouse-controlled fish`.

## Inspiration
In Irfan Essa's Computational Photography class at GA Tech, one of the topics covered was video textures.  I became fascinated with the idea that video frames could be smoothly rearranged to create novel animations.  Something about that idea just seemed... magical: that an initial 30 second video of a candle flickering could be used to generate an infinite number of new, arbitrary length, non-repeating animations.  Later in the class, when I discovered some of the more advanced applications of video textures, I became hooked.  For the final project, I implemented a pipeline that enables the creation of mouse-controlled video sprites.

## Notes
1.  To better understand video textures and video sprites, I highly recommend checking out the two links under `Resources About Video Textures` above.  
2.  Use of this pipeline will require a thorough understanding of original paper.  (Specifically the purpose of each variable, and the math involved in the video texture generation process).
3.  The code in this pipeline could use improvement!  For the most part, I left this code in its original state (plus some additional comments).  There are therefore probably a multitude of possible improvements such as: better use of numpy for efficient computations, implementing multiprocessing for certain functions, implementing better background subtraction, etc.
4.  This pipeline was created to handle videos containing a single moving subject.  Multiple subjects and/or background movement will result in poor quality video textures.

## How To Use
To use this pipeline, do the following:
1.  Set all values as required in the `config.yaml` file.
2.  Make sure the input video file is in the same directory as `mouse_controlled_video_sprite_pipeline.py`.
3.  Run pipeline with `python mouse_controlled_video_sprite_pipeline.py`.
4.  Windows named 'Frame Sequence Visualization' displaying the present video volume will appear after the input video is read in, and after the background subtraction process, if the `show` parameter in `config.yaml` is set to `True`.  These windows can be closed by pressing `Esc`.
5.  A window named 'Markov Synthesis' will appear.  At this point, the user can control to which anchor point the video sprite's movement is biased by left clicking that area on the window.  This window can be closed by right clicking, or by pressing `Esc`.
6.  Another window named 'Frame Sequence Visualization' will appear showing an animation of the final video texture.  This window can be closed by pressing `Esc`.

Each time the pipeline is run using a new `config.yaml` file, all precomputations are cached.  This makes subsequent executions of the pipeline MUCH faster after the initial run.

## Outputs
After this pipeline completes, the following will have been created:
1.  A project directory in `/output`.
2.  A copy of the original input video file in `/output/<project_dir>`.
3.  A copy of the generated video texture in `/output/<project_dir>`.
4.  Image files for each frame used in the video texture in `/output/<project_dir>/frames`
5.  Visualizations at key computational steps (refer to [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf)) for each anchor point in `/output/<project_dir>/anchor_points/<anchor_point>`.
6.  A serialized file of all precomputations in `/cache/<cache_file>`.

## Synthesis Parameters
All parameters affecting the video texture synthesis process are specified in the `config.yaml` file.  A brief explanation of these parameters is as follows:
1.  `video_file` - the name of the input video file 
2.  `resize` - a float value for how the dimensions of the input video should be scaled (e.g. 0.25 means 25% length and height) 
3.  `frame_parse` - how many frames should be kept from the input video (e.g. 3 means every third frame is kept) 
4.  `mb_kern` - the side length in pixels of the median blur kernel that is used during the background subtraction process 
5.  `px_thresh` - the binary mask pixel threshold that is used during the background subtraction process (e.g. 127 means values above 127 are thresholded to 255, and below or equal are thresholded to 0) 
6.  `min_obj_size` - the minimum area in pixels of the video sprite subject to be used during the background subtraction process
7.  `velocity_window` - the gap in frames that should be used when calculating video sprite subject velocity (e.g. 3 means frame i and i+3 are used) 
8.  `w_1` - the w_1 parameter from equation 11 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
9.  `w_2` - the w_2 parameter from equation 11 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
10.  `w_3` - the w_3 parameter from equation 11 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
11.  `sigma` - the sigma parameter from equations 2, 5, and 10 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
12.  `p` - the p parameter from equations 4, 6, and 8 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
13.  `a` - the alpha parameter from equations 4, 6, and 8 from [Video Textures](https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/videotex.pdf) 
14.  `p_thresh` - the probability threshold used during the Markov chain pruning process (e.g 0.1 means all probabilities below 0.1 are pruned) 
15.  `start_frame` - which frame will be used as an initial frame in the synthesis process 
16.  `num_frames` - the total number of frames to be used from the input video file, prior to any parsing via `frame_parse`, etc. (e.g. 100 means use only the first 100 frames from the input video file) 
17.  `fps` - the frames per second that should be used when rendering the video file for the final video texture 
18.  `show` - whether to show the current video volume after reading in the video file and after the background subtraction process
19.  `show_wait_time_ms` - the amount of time in milliseconds that should be waited between frames when a video volume is being displayed to the user

## Final Thought
An input video, config file, and outputs have been included in this repository to assist as an example.  Feel free to tinker with them.  You should be able to feel that when you click near different anchor points, Kirby will soon begin to navigate in that direction.  An anchor point is any corner or the center of any side of the 'Markov Synthesis' window.

This example can be "glitchy" at times due to its short duration, which limits the number of possible good transitions.  Additionally, the fact that Kirby's velocity is constantly "bobbing" is more difficult to handle than the smooth motion of, for example, a swimming fish.

Every video file will require different settings in the `config.yaml` file, according to the nature of the video.












