#include <stdio.h>
#include "darknet.h"

int main(int argc, char **argv)
{
    // Charger le réseau YOLO
    char *cfg_file = "cfg/yolov4.cfg";
    char *weight_file = "yolov4.weights";
    char *data_file = "cfg/coco.data";
    printf("Avant le chargement de network\n");
    network *net = load_network(cfg_file, weight_file, 0);
    
    printf("Avant le chargement des metadata\n");
    // Charger les classes
    metadata meta = get_metadata(data_file);

    // Charger l'image
    char *input = "test.jpg";
    printf("Avant le chargement de l'image\n");
    image im = load_image_color(input, 0, 0);
    printf("Avant resize de l'image\n");
    image sized = letterbox_image(im, net->w, net->h);

    // Détecter les objets
    layer l = net->layers[net->n - 1];
    float *X = sized.data;

    printf("Avant network_predict\n");
    if (net == NULL) {
        fprintf(stderr, "Erreur : le réseau est NULL.\n");
        return 1;
    }

    if (X == NULL) {
        fprintf(stderr, "Erreur : les données d'entrée sont NULL.\n");
        return 1;
    }

    if (net->w <= 0 || net->h <= 0) {
        fprintf(stderr, "Erreur : les dimensions du réseau sont invalides.\n");
        return 1;
    }

    network_predict(*net, X);
    int nboxes = 0;
    printf("Avant get_network_boxes\n");
    detection *dets = get_network_boxes(net, im.w, im.h, 0.5, 0.5, 0, 1, &nboxes, 0);

    // Afficher les résultats
    for (int i = 0; i < nboxes; ++i) {
        if (dets[i].prob[0] > 0.5) { // Seuil de confiance
            //int class = max_index(dets[i].prob, l.classes);
            // You can use a loop to find the index of the maximum value in the 'dets[i].prob' array
            int class = 0;
            float max_prob = dets[i].prob[0];
            for (int j = 1; j < l.classes; ++j) {
                if (dets[i].prob[j] > max_prob) {
                    max_prob = dets[i].prob[j];
                    class = j;
                }
            }
            printf("%s: %.0f%%\n", meta.names[class], dets[i].prob[class] * 100);
        }
    }

    // Libérer la mémoire
    printf("Avant free_detections\n");
    free_detections(dets, nboxes);
    printf("Avant free_image im\n");
    free_image(im);
    printf("Avant free_image sized\n");
    free_image(sized);

    return 0;
}
