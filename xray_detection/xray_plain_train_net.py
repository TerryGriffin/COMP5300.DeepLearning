# COMP.5300 Deep Learning
# Terry Griffin
#
# Training and evaluation script based on the Detectron2 example
# script plain_train_net.py
#
# The example script has been modified to
#  provide evaluation of a validation set during training
#  collection validation stats for TensorBoard
#  support the UML X-ray and NIH X-ray
#  include class heatmaps as a post-processing filter step

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from filtered_coco_evaluator import FilteredCOCOEvaluator
import time
import datetime

logger = logging.getLogger("xray_train")


def get_evaluator(cfg, dataset_name, output_folder=None, cat_heatmap_file = None):
    """
    Use a custom evaluator to include heatmap processing
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return FilteredCOCOEvaluator(dataset_name, cfg, True,
                                 output_folder, cat_heatmap_file, 0.2)

def do_test(cfg, model, cat_heatmap_file):
    """
    Run the model on the test sets and output the results
    """
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
            cat_heatmap_file
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def flatten_results(coco_eval_results):
    """
    Flatten the collection results for writing to a tensorboard collection
    """
    results = {}
    for label, scores in coco_eval_results.items():
        results.update({label + '_eval/' + k: v for k, v in scores.items()})
    return results


def do_train(cfg, model, cat_heatmap_file, resume=False):
    model.train()

    # select optimizer and learning rate scheduler based on the config
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # creat checkpointer
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    # create output writers. Separate TensorBoard writers are created
    # for train and validation sets. This allows easy overlaying of graphs
    # in TensorBoard.
    train_tb_writer = os.path.join(cfg.OUTPUT_DIR, 'train')
    val_tb_writer = os.path.join(cfg.OUTPUT_DIR, 'val')
    train_writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(train_tb_writer),
        ]
        if comm.is_main_process()
        else []
    )
    val_writers = [TensorboardXWriter(val_tb_writer)]


    train_dataset_name = cfg.DATASETS.TRAIN[0]
    train_data_loader = build_detection_train_loader(cfg)
    train_eval_data_loader = build_detection_test_loader(cfg, train_dataset_name)
    val_dataset_name = cfg.DATASETS.TEST[0]
    val_eval_data_loader = build_detection_test_loader(cfg, val_dataset_name, DatasetMapper(cfg,True))
    logger.info("Starting training from iteration {}".format(start_iter))
    train_storage = EventStorage(start_iter)
    val_storage = EventStorage(start_iter)

    # Create the training and validation evaluator objects.
    train_evaluator = get_evaluator(
        cfg, train_dataset_name, os.path.join(cfg.OUTPUT_DIR, "train_inference", train_dataset_name),
        cat_heatmap_file
    )
    val_evaluator = get_evaluator(
        cfg, val_dataset_name, os.path.join(cfg.OUTPUT_DIR, "val_inference", val_dataset_name),
        cat_heatmap_file
    )

    # initialize the best AP50 value
    best_AP50 = 0
    start_time = time.time()
    for train_data, iteration in zip(train_data_loader, range(start_iter, max_iter)):
         # stop if the file stop_running exists in the running directory
         if os.path.isfile('stop_running'):
             os.remove('stop_running')
             break

         iteration = iteration + 1

         # run a step with the training data
         with train_storage as storage:
            model.train()
            storage.step()

            loss_dict = model(train_data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()


            # periodically evaluate the training set and write the results
            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter):

                train_eval_results = inference_on_dataset(model, train_eval_data_loader,
                                                          train_evaluator)
                flat_results = flatten_results(train_eval_results)
                storage.put_scalars(**flat_results)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in train_writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

         # run a step with the validation set
         with val_storage as storage:
            storage.step()

            # every 20 iterations evaluate the dataset to collect the loss
            if iteration % 20 == 0 or iteration == max_iter:
                with torch.set_grad_enabled(False):
                     for input, i in zip(val_eval_data_loader , range(1)):
                        loss_dict = model(input)
                        losses = sum(loss for loss in loss_dict.values())
                        assert torch.isfinite(losses).all(), loss_dict

                        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # periodically evaluate the validation set and write the results
            # check the results against the best results seen and save the parameters for
            # the best result
            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                or iteration == max_iter):
                val_eval_results = inference_on_dataset(model, val_eval_data_loader,
                                                        val_evaluator)
                logger.info('val_eval_results {}', str(val_eval_results))
                results = val_eval_results.get('segm', None)
                if results is None:
                    results = val_eval_results.get('bbox', None)
                if results is not None and results.get('AP50',-1) > best_AP50:
                    best_AP50 = results['AP50']
                    logger.info('saving best results ({}), iter {}'.format(best_AP50, iteration))
                    checkpointer.save("best_AP50")

                flat_results = flatten_results(val_eval_results)
                storage.put_scalars(**flat_results)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0):
                for writer in val_writers:
                    writer.write()
                elapsed = time.time() - start_time
                time_per_iter = elapsed / (iteration - start_iter)
                time_left = time_per_iter * (max_iter - iteration)
                logger.info("ETA: {}".format(str(datetime.timedelta(seconds=time_left))))


def register_datasets():
    """
    Register the UML TB dataset and the NIH X-ray datasets
    """
    register_coco_instances("xray_ac_train", {}, "datasets/xray/annotations/xray_ac_train.json", "datasets/xray/xray_images")
    register_coco_instances("xray_ac_val", {}, "datasets/xray/annotations/xray_ac_val.json", "datasets/xray/xray_images")
    register_coco_instances("xray_ac_test", {}, "datasets/xray/annotations/xray_ac_test.json", "datasets/xray/xray_images")
    register_coco_instances("xray_cav_train", {}, "datasets/xray/annotations/xray_cav_train.json", "datasets/xray/xray_images")
    register_coco_instances("xray_cav_val", {}, "datasets/xray/annotations/xray_cav_val.json", "datasets/xray/xray_images")
    register_coco_instances("xray_cav_test", {}, "datasets/xray/annotations/xray_cav_test.json", "datasets/xray/xray_images")
    register_coco_instances("xray_ly_train", {}, "datasets/xray/annotations/xray_ly_train.json", "datasets/xray/xray_images")
    register_coco_instances("xray_ly_val", {}, "datasets/xray/annotations/xray_ly_val.json", "datasets/xray/xray_images")
    register_coco_instances("xray_ly_test", {}, "datasets/xray/annotations/xray_ly_test.json", "datasets/xray/xray_images")
    register_coco_instances("xray_pe_train", {}, "datasets/xray/annotations/xray_pe_train.json", "datasets/xray/xray_images")
    register_coco_instances("xray_pe_val", {}, "datasets/xray/annotations/xray_pe_val.json", "datasets/xray/xray_images")
    register_coco_instances("xray_pe_test", {}, "datasets/xray/annotations/xray_pe_test.json", "datasets/xray/xray_images")
    register_coco_instances("xray_4classes_train", {}, "datasets/xray/annotations/xray_4classes_train.json", "datasets/xray/xray_images")
    register_coco_instances("xray_4classes_train_balanced", {}, "datasets/xray/annotations/xray_4classes_train_balanced.json", "datasets/xray/xray_images")
    register_coco_instances("xray_4classes_val", {}, "datasets/xray/annotations/xray_4classes_val.json", "datasets/xray/xray_images")
    register_coco_instances("xray_4classes_test", {}, "datasets/xray/annotations/xray_4classes_test.json", "datasets/xray/xray_images")

    register_coco_instances("nih_bbox_full_train", {}, "datasets/ChestXray-NIHCC/annotations/nih_bbox_full_train.json", "datasets/ChestXray-NIHCC/images")
    register_coco_instances("nih_bbox_full_train_balanced", {}, "datasets/ChestXray-NIHCC/annotations/nih_bbox_full_train_balanced.json", "datasets/ChestXray-NIHCC/images")
    register_coco_instances("nih_bbox_full_val", {}, "datasets/ChestXray-NIHCC/annotations/nih_bbox_full_val.json", "datasets/ChestXray-NIHCC/images")
    register_coco_instances("nih_bbox_full_test", {}, "datasets/ChestXray-NIHCC/annotations/nih_bbox_full_test.json", "datasets/ChestXray-NIHCC/images")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_datasets()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, args.cat_heatmap_file)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, args.cat_heatmap_file)
    return do_test(cfg, model, args.cat_heatmap_file)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--cat-heatmap-file", metavar="FILE")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
