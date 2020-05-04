# COMP.5300 Deep Learning
# Terry Griffin
#
# Creates a collection of heatmap objects for each category in a
# COCO json input file. The collection is a dictionary keyed by category name.
# Given in input file of ABC.json the heatmap collection is written
# to ABC_heatmaps.pickle. This file can be used as input to the xray_plain_train_net.py
# program.
# An image of the heatmap for each category
# is written to a file named ABC_<category>.jpg. These images are generated
# only as a diagnostic and are not used for processing.

import json
import sys
import os
import pickle

from category_heatmap import Heatmap

def create_cat_heatmap(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # create a dictionary for the heatmaps keyed by category name
    cat_heatmaps = dict()
    for cat in data['categories']:
        cat_heatmaps[cat['id']] = Heatmap(cat['name'])

    # collect the image sizes, keyed by image_id
    image_sizes = dict()
    for image in data['images']:
        image_sizes[image['id']] = (image['width'], image['height'])

    # for each annotation, add the area to the heightmap for the
    # given category. If segmentation data is not available use the
    # bounding box data.
    for anno in data['annotations']:
        if anno['image_id'] not in image_sizes.keys():
            continue
        if 'segmentation' in anno.keys():
            add_segmentation(cat_heatmaps[anno['category_id']],
                             image_sizes[anno['image_id']],
                             anno['image_id'],
                             anno['segmentation'][0])
        elif 'bbox' in anno.keys():
            add_bbox(cat_heatmaps[anno['category_id']],
                     image_sizes[anno['image_id']],
                     anno['image_id'],
                     anno['bbox'])

    file_prefix = os.path.splitext(filename)[0]

    # write the image for each heatmap per category
    for heatmap in cat_heatmaps.values():
        img = heatmap.as_image()
        clean_name = (heatmap.name()).replace(' ', '_')
        output_filename = f'{file_prefix}_{clean_name}.jpg'
        print(f'Saving {output_filename}')
        img.save(output_filename)

    # write the collection
    output_filename = f'{file_prefix}_heatmaps.pickle'
    with open(output_filename,'wb') as file:
        pickle.dump(cat_heatmaps, file)


def add_bbox(heatmap, image_size, image_id, bbox):
    x1, y1 = bbox[:2]
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    seg_list = [x1, y1, x2, y1, x2, y2, x1, y2]
    add_segmentation(heatmap, image_size, image_id, seg_list)

def add_segmentation(heatmap, image_size, image_id, seg_list):
    points = []
    for i in range(0, len(seg_list), 2):
        points.append((round(seg_list[i]), round(seg_list[i+1])))
    points.append(points[0])
    heatmap.add_polygon(points, image_size, image_id)


if __name__ == '__main__':
    create_cat_heatmap(sys.argv[1])