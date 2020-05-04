# COMP.5300 Deep Learning
# Terry Griffin
#
# Heatmap class for collecting location data
# for each manifestation

import numpy as np
from PIL import Image, ImageDraw

class Heatmap:
    HEIGHT = 1024
    WIDTH = 1024

    def __init__(self, name):
        self._map = np.zeros((Heatmap.WIDTH, Heatmap.HEIGHT), dtype=float)
        self._image_ids = set()
        self._name = name
        self._normalized = False

    def name(self):
        return self._name

    def map_size(self):
        return self._map.shape

    def add_polygon(self, points, image_size, image_id):
        self._unnormalize()
        img = Image.new('L', image_size, 0)
        ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
        img = img.resize(self.map_size())
        mask = np.array(img)
        self._map += mask
        self._image_ids.add(image_id)

    def _normalize(self):
        if not self._normalized:
            self._normalizing_value = np.max(self._map)
            if self._normalizing_value > 0:
                self._map /= self._normalizing_value
            self._normalized = True

    def _unnormalize(self):
        if self._normalized:
            self._map = self._map * self._normalizing_value
            self._normalized = False

    def as_image(self):
        self._normalize()
        gray_scale = (self._map * 255).astype('uint8')
        img = Image.fromarray(gray_scale)
        return img

    def get_bbox_max(self, bbox, image_size):
        x1, y1 = bbox[:2]
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        return self.get_polygon_max(points, image_size)

    def get_polygon_max(self, points, image_size):
        self._normalize()
        img = Image.new('L', image_size, 0)
        ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
        img = img.resize(self.map_size())
        mask = np.array(img)
        max_value = np.max(self._map[mask == 1])
        return max_value
