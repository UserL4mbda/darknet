#!/bin/sh

# Creation de l'image darknet (sh et powershe)
docker build -t darknet:cpu -f ./Dockerfile.cpu .

# Utilisation de la command darknet
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output data/dog.jpg
