# HV_plane_detection
Horizontal and Vertical Plane Detection using OpenCV (HoughTransform)


# Description

`HV_plane_detect.py` contains Source for identifying the Horizontal and Vertical Planes from the provided Image. It uses Hough Transformation Technique for identifying Horizontal and Vertical Lines.


# C++ varient Usage and Build support

All the C++ code is under cpp directory. (This is the same version of algorithm which is also implemented in Python)

    $ cd cpp
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make
    $ ./testHVPlaneDetect --input ../../data/IMG_6857.jpg


# Dependencies

In order to run the particular Algo, following packages are required.

    $ sudo apt-get install python-setuptools python-dev build-essential libglib2.0-dev python-tk
    $ sudo -H pip install -r requirements.txt

# Usage

    $ python HV_plane_detect.py -h
    usage: HV_plane_detect.py [-h] --input INPUT [--debug]

    optional arguments:
    -h, --help     show this help message and exit
    --input INPUT  input file path
    --debug        display intermediate outputs, requires matplotlib.pyplot


# Integrating to other System

As this is built based on Python Public Class, it is much easier for user to integrate the Algorithm to his System by simply creating Object of `class segmentation` and calling out it's public methods such as `floor_segmentation()` and `wall_segmentation()`

Follow below block of sample code in order to integrate this to Your Code.

```Python
segment = segmentation(filename=args.input)

floor_pts, floor_mask = segment.floor_segmentation()
wall_pts, wall_mask = segment.wall_segmentation()

segment.dump_results() # store the result JPG
```

In order to print debug logs, please set log level to `logging.DEBUG`.

```Python

log.setLevel(logging.DEBUG)
```