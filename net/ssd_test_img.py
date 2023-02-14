import os

from click import UsageError

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

import torch
from vision.datasets.voc_dataset_OK import VOCDataset
from vision.ssd.data_preprocessing import TestTransform
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.ssd import MatchPrior
import logging
from torch.utils.data import DataLoader

from vision.nn.multibox_loss import MultiboxLoss


def test(loader, net, criterion, device):
	net.to(device)
	net.eval()
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	num = 0
	for _, data in enumerate(loader):
		images, boxes, labels = data
		images = images.to(device)
		boxes = boxes.to(device)
		labels = labels.to(device)
		num += 1


		with torch.no_grad():
			confidence, locations = net(images)
			regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
			loss = regression_loss + classification_loss
			print(f"Num {num} Loss {loss:.4f}")
			if(num == 66 or torch.isnan(loss)):
				print("NANI")

		running_loss += loss.item()
		running_regression_loss += regression_loss.item()
		running_classification_loss += classification_loss.item()
	return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == "__main__":
	
	if len(sys.argv) != 3:
		raise UsageError("Usage: python ssd_test_img.py <img_name> <model_path> <net_type>")

	model_path, net_type = sys.argv[1], sys.argv[2]
	timer = Timer()
	label_path = "models/rds-labels.txt"

	#img_name = img_path.split("/")[1]

	# --------------- #

	class_names = [name.strip() for name in open(label_path).readlines()]
	num_classes = len(class_names)
	if net_type == 'vgg16-ssd':
		net = create_vgg_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1-ssd':
		net = create_mobilenetv1_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1-ssd-lite':
		net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'mb2-ssd-lite':
		net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'sq-ssd-lite':
		net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
	else:
		print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
		sys.exit(1)
		
	net.load(model_path)

	if net_type == 'vgg16-ssd':
		predictor = create_vgg_ssd_predictor(net, candidate_size=200)
		config = vgg_ssd_config
	elif net_type == 'mb1-ssd':
		predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200,device='cpu')
		config = mobilenetv1_ssd_config
	elif net_type == 'mb1-ssd-lite':
		predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
		config = mobilenetv1_ssd_config
	elif net_type == 'mb2-ssd-lite':
		predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method="soft")
		config = mobilenetv1_ssd_config
	elif net_type == 'sq-ssd-lite':
		predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
		config = squeezenet_ssd_config
	else:
		print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
		sys.exit(1)

	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
						format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	print("Loading Trained Model is Done!\n")

	image_sets_path = './RSD-GOD/testingsets'
	batch_size = 8
	num_workers = 0
	use_cuda = False
	DEVICE = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
	net.to(DEVICE)
	net.device = 'cpu'
	EVALUATE = True

	if EVALUATE is True:
		print("Starting Evaluation...\n")
		print(f"Dataset : {image_sets_path}")
		test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
		target_transform = MatchPrior(config.priors, config.center_variance,
									  config.size_variance, 0.5)
		criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
								 center_variance=0.1, size_variance=0.2, device=DEVICE)
		val_dataset = VOCDataset(image_sets_path, transform=test_transform,
								 target_transform=None, is_test=True)
		val_loader = DataLoader(val_dataset, batch_size,
								num_workers=num_workers,
								shuffle=False)

		logging.info(val_dataset)
		val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
		logging.info(
			f"Validation Loss: {val_loss:.4f}, " +
			f"Validation Regression Loss {val_regression_loss:.4f}, " +
			f"Validation Classification Loss: {val_classification_loss:.4f}"
		)

	DETECT = False
	if DETECT is True:
		print("Starting Detection...\n")

		net.cpu()
		for img_path in sorted(os.listdir(image_sets_path)):
			orig_image = cv2.imread(image_sets_path +'/'+ img_path)
			image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
			color = np.random.uniform(0, 255, size = (10, 3))

			timer.start()
			boxes, labels, probs = predictor.predict(image, 10, 0.4)
			interval = timer.end()
			print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

			fps = 1/interval

			for i in range(boxes.size(0)):
				box = boxes[i, :]
				label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

				i_color = int(labels[i])
				box = [round(b.item()) for b in box]

				cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color[i_color], 2)

				cv2.putText(orig_image, label,
							(box[0] - 10, box[1] - 10),
							cv2.FONT_HERSHEY_SIMPLEX,
							1,  # font scale
							color[i_color],
							2)  # line type

			print(orig_image.shape)
			cv2.imwrite('outputs/test2/' + img_path, orig_image)


		print("Check the result!")
