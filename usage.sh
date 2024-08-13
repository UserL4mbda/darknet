#!/bin/sh
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output data/dog.jpg
