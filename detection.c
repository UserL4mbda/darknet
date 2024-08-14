#include "darknet.h"
#include "utils.h"
#include "image.h"
#include "parser.h"
#include "option_list.h"

int main(int argc, char **argv)
{
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    network *net = load_network("cfg/yolov4.cfg", "yolov4.weights", 0);
    set_batch_network(net, 1);
    //image im = load_image_color("data/dog.jpg", 0, 0);
    image im = load_image_color(argv[1], 0, 0);
    image sized = letterbox_image(im, net->w, net->h);
    layer l = net->layers[net->n - 1];
    network_predict(*net, sized.data);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, 0.5, 0.5, 0, 1, &nboxes, 0);
    do_nms_sort(dets, nboxes, l.classes, 0.4);

    list *options = read_data_cfg("cfg/coco.data");
    char *name_list = option_find_str(options, "names", "data/coco.names");
    char **names = get_labels(name_list);

    // Print the positions and dimensions of the bounding boxes
    printf("[\n");
    for (int i = 0; i < nboxes; ++i) {
        int class_id = -1;
        float prob = 0;
        box b = dets[i].bbox;
        for(int j = 0; j < l.classes; ++j) {
            if (dets[i].prob[j] > prob) {
                prob = dets[i].prob[j];
                class_id = j;
            }
        }
        if (class_id >= 0 && prob > 0.5) {
            //printf("%s: %.2f%%, x: %.2f, y: %.2f, width: %.2f, height: %.2f\n", names[class_id], prob*100, b.x, b.y, b.w, b.h);
            int left   = (b.x - b.w/2.) * im.w;
            int right  = (b.x + b.w/2.) * im.w;
            int top    = (b.y - b.h/2.) * im.h;
            int bottom = (b.y + b.h/2.) * im.h;

            //printf("%s: %.2f%%\n", names[class_id], prob*100);
            //printf("Bounding Box (pixels): Left: %d, Top: %d, Right: %d, Bottom: %d\n", left, top, right, bottom);
            //printf("Width: %d, Height: %d\n\n", right - left, bottom - top);
            printf("  {\n");
            printf("    \"class\": \"%s\",\n", names[class_id]);
            printf("    \"confidence\": %.2f%%,\n", prob*100);
            printf("    \"x\": %.2f,\n", b.x);
            printf("    \"y\": %.2f,\n", b.y);
            printf("    \"width\": %.2f,\n", b.w);
            printf("    \"height\": %.2f,\n", b.h);
            printf("    \"left\": %d,\n", left);
            printf("    \"right\": %d,\n", right);
            printf("    \"top\": %d,\n", top);
            printf("    \"bottom\": %d,\n", bottom);
            printf("    \"width_pixels\": %d,\n", right - left);
            printf("    \"height_pixels\": %d\n", bottom - top);
            printf("  }%s\n", (i == nboxes - 1) ? "" : ",");
        }
    }
    printf("]\n");

    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
    free_network(*net);
    free_ptrs((void **)names, l.classes);
    return 0;
}
