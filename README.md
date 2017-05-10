# HV_plane_detection
Horizontal and Vertical Plane Detection using OpenCV (HoughTransform)


# Description

`HV_plane_detect.py` contains Source for identifying the Horizontal and Vertical Planes from the provided Image. It uses Hough Transformation Technique for identifying Horizontal and Vertical Lines.


# Dependencies

In order to run the particular Algo, following packages are required.

    $ sudo apt-get install python-setuptools python-dev build-essential libglib2.0-dev python-tk
    $ sudo -H pip install -r requirements.txt

# Usage

    $ python HV_plane_detect.py input debug

    where, input - Input Image File name with Full path
           debug - If set to non-zero, internally utilises matplotlib to display intermediate output for the Algorithm. 
                   Useful for debugging purpose. Default is 0.


