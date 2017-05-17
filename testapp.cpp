#include <iostream>
#include "segmentation.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: testHVPlaneDetect <input>" << std::endl;
    }

    Segmentation seg(argv[1]);
    seg.floor();
}