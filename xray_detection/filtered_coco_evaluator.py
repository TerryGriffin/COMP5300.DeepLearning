# COMP.5300 Deep Learning
# Terry Griffin
#
# FilteredCOCOEvaluator subclasses COCOEvaluator
# to add filtering of false positives using a collection of heatmaps.
# An instance is filtered out if the area of the proposed bounding box
# in the heatmap does not contain any value above a given threshold.

import pickle

from detectron2.evaluation.coco_evaluation import (
    COCOEvaluator,
    instances_to_coco_json
)

from category_heatmap import Heatmap

class FilteredCOCOEvaluator(COCOEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir, cat_heatmap_file,
                 threshold = 0.2):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._threshold = threshold

        # create a reverse map to go from the contiguous set of category ids used during training
        # and testing to those used in the original json file
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }

        # load the heatmaps or default to an empty dict
        if cat_heatmap_file:
            with open(cat_heatmap_file, 'rb') as file:
                self._cat_heatmaps = pickle.load(file)
        else:
            self._cat_heatmaps = dict()

    def process(self, inputs, outputs):
        """
        Override the base process method. This provides
        the same processing with the addition of the filter step
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                instances = instances_to_coco_json(instances, input["image_id"])
                instances = self.filter_instances(input["image_id"],
                                                  (input['width'], input['height']),
                                                  instances)
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def filter_instances(self, image_id, image_size, instances):
        """
        Return the filtered list of instances. Any instance whose
        bounding box area of the heightmap for the detected class does not
        meet the threshold is removed.
        """
        result = []
        for inst in instances:
            cat_id = self._reverse_id_mapping[inst['category_id']]
            if cat_id in self._cat_heatmaps.keys():
                heatmap = self._cat_heatmaps[cat_id]
                bbox_max = heatmap.get_bbox_max( inst['bbox'], image_size)
                if bbox_max >= self._threshold:
                    result.append(inst)
                else:
                    self._logger.info(f'Removing instance for image {image_id}, bbox_max = {bbox_max}, inst = {inst}')
            else:
                result.append(inst)
        return result

