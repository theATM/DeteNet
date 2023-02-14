
import os, json, cv2, random


from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from repo.detectron2.engine import DefaultPredictor,  DefaultTrainer
from repo.detectron2.config import get_cfg


def register_datasets(name):
    # register_coco_instances("RSD-GOD", {}, "json_annotation.json", "RSD-GOD")
    register_coco_instances("RSD-COCO-train", {}, "RSD-COCO/train/_annotations.coco.json",
                            "RSD-COCO/train")
    register_coco_instances("RSD-COCO-val", {}, "RSD-COCO/valid/_annotations.coco.json",
                            "RSD-COCO/valid")
    register_coco_instances("RSD-COCO-test", {}, "RSD-COCO/test/_annotations.coco.json",
                            "RSD-COCO/test")

    from detectron2.structures import BoxMode


def train(name):

    import torch
    #ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    #utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    if not os.path.exists("./output"):
        os.makedirs("./output")
        print("Created Output Dir")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = ("RSD-COCO-train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 6000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.BACKBONE.FREEZE_AT = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    #trainer.test(cfg)




def eval(name):

    if not os.path.exists("./output_val"):
        os.makedirs("./output_val")
        print("Created Output Val Dir")

    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    #cfg.DATASETS.TRAIN = ("RSD-COCO-train",)
    cfg.DATASETS.TEST = ("RSD-COCO-test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    #cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = "runs/train4_6000/output/model_final.pth"
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator, DatasetEvaluators
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("RSD-COCO-test", output_dir="./output_val")
    val_loader = build_detection_test_loader(cfg, "RSD-COCO-test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`





def predict(name):

    if not os.path.exists("./output_pred"):
        os.makedirs("./output_pred")
        print("Created Output Pred Dir")

    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    #cfg.DATASETS.TRAIN = ("RSD-COCO-train",)
    cfg.DATASETS.TEST = ("RSD-COCO-test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    #cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = "runs/train3_3000/model_final.pth"
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import build_detection_test_loader
    from detectron2.data import MetadataCatalog, DatasetCatalog
    dataset_metadata = MetadataCatalog.get("RSD-COCO-test")
    dataset_dicts = build_detection_test_loader(cfg, "RSD-COCO-test")

    for d in dataset_dicts:
        im = cv2.imread(d[0]["file_name"])
        cv2.imshow('image', im)
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # "output_pred/"+im,
        cv2.imwrite('./output_pred/'+d[0]["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])





if __name__ == '__main__':
    register_datasets('gg')
    #train('gg')
    eval('gg')
    #predict('gg')
