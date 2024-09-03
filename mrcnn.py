# Import libraries

from __future__ import annotations
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import random
import torchvision.models.detection
import numpy as np
import cv2 as cv
import torch

# Import methods
import data_prep
import calculations


def trainModel(device: torch.DeviceObjType, model, dirs, batchSize=4, epochs=1):
    """
    Prepares and trains the provided model on the given images.
    """
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(
            'e49.torch', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('e49.torch'))

    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    model.train()

    for e in range(47, epochs):
        random.shuffle(dirs)
        for i in range(0, len(dirs), batchSize):
            dirs, images, targets = data_prep.loadData(
                dirs, i, i + batchSize)
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            # Report training loss for batch
            print(i, 'loss:', losses.item())
            if i % 140 == 0:
                torch.save(model.state_dict(), f't1/e{e}s{str(i)}.torch')
                print("Save model to:", f'./t1/e{e}s{str(i)}.torch')

                model.eval()
                with torch.no_grad():
                    pred = model(images)
                print(pred)
                model.train()

        torch.save(model.state_dict(), f't1/e{e}.torch')
        print(f'Finished epoch {e}')


def predSample(device, model, label_ref, sample, seePred: bool = False):
    """
    Performs a single prediction for the passed image file using the provided model and returns
    the predicted masks. Optionally displays the predicted masks with vPred = True.
    """
    model.to(device)
    model.eval()

    image = cv.imread(sample)

    image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
    image = image.swapaxes(1, 3).swapaxes(2, 3)
    image.to(device)

    with torch.no_grad():
        pred = model(image)

    image = image[0].swapaxes(0, 2).swapaxes(
        0, 1).detach().cpu().numpy().astype(np.uint8)

    if seePred:
        data_prep.visualizePred(image, pred, label_ref)

    return pred


def evalModel(device, model: torch.models.detection.MaskRCNN, imgs: list[str], label_ref: dict, confidence: float):
    """
    Calculates the average mean intersection over union (mIOU) score for the provided model
    using the validation set, for model performance above the provided confidence value.
    """
    model.to(device)
    model.eval()

    dirs, images, truths = data_prep.loadData(
        imgs, 0, len(imgs), noResize=True)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = [img.to(device) for img in images]
    confusion = np.zeros((len(label_ref), len(label_ref)))
    for i in range(len(dirs)):
        pred = predSample(device, model, label_ref, dirs[i])
        calculations.updateConfusion(
            pred, truths[i], confidence, dirs[i], confusion)

    score = calculations.getmIOU(confusion)
    np.savetxt(f'eval_{confidence}_confusion.txt', confusion, delimiter=', ')
    return score


if __name__ == '__main__':
    from label_definitions import LABEL_NAMES

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    imgSize = [600, 600]
    batchSize = 4
    epochs = 50
    t_imgs_dirs = data_prep.loadImageDirs('train')
    v_imgs_dirs = data_prep.loadImageDirs('val')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights='DEFAULT')

    # Modify model to predict the needed number of classes
    num_classes = 25
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    # Load model state
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(
            'e49.torch', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('e49.torch'))

    # trainModel(device, model, t_imgs_dirs, batchSize, epochs)
    # predSample(device, model, LABEL_NAMES, 'test/1317440_slide-011.jpg', seePred=True)
    # predSample(device, model, LABEL_NAMES, 'val/1307323_slide-004.jpg', seePred=True)
    score = evalModel(device, model, v_imgs_dirs, LABEL_NAMES, 0.5)
    print(score)

    # images = cv.imread('train/3561_slide-083.jpg')
    # images = cv.imread('test/1317440_slide-011.jpg')
    # images = cv.imread('test/203627_slide-019.jpg')
    # images = cv.resize(images, (600, 600), cv.INTER_LINEAR)
