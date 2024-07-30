from __future__ import annotations
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import random
import torchvision.models.detection
import scipy.io as sio
import os
import numpy as np
import cv2 as cv
import torch


def loadImages(imgdir: str) -> list:
    """
    Returns a list of all file paths in directory dir.
    """
    return [f'{imgdir}/{path}' for path in os.listdir(f'{imgdir}')]


def getMasks(labels: np.ndarray, imgSize: list[int] = [600, 600]) -> list:
    """
    Returns a list of masks for the given label array, resized to the given imgSize.
    """
    masks = []
    mask_classes = []
    for dim in range(24):
        slice_ids = np.unique(labels[:, :, dim])[1:]
        if slice_ids.size > 0:
            mask = (labels[:, :, dim] > 0).astype(np.uint8)
            mask = cv.resize(mask, imgSize, cv.INTER_NEAREST)
            masks.append(mask)
            mask_classes.append(dim + 1)
    return masks, mask_classes


def loadData(imgs: list[str], start: int, end: int, imgSize: list[int] = [600, 600]) -> None:
    """
    Loads batchSize number of images into Torch tensors, with annotated ground
    truth masks.
    """
    batch_imgs = []
    batch_data = []

    for i in range(start, min(end, len(imgs))):

        # Read and resize image to correct dimensions

        # index = random.randint(0, len(imgs) - 1)
        img = cv.imread(imgs[i])
        img = cv.resize(img, imgSize, cv.INTER_LINEAR)

        # Load and prepare masks from .mat file
        full_file_dir = imgs[i].split('/')[-1]
        fileName = full_file_dir[:-4]
        labels = sio.loadmat(f'labels/{fileName}.mat')['labels']
        masks, classes = getMasks(labels, imgSize)

        # Build binding box record of dimension [len(masks), 4] (x, y, x + w, y + h)
        num_boxes = len(masks)
        boxes = torch.zeros([num_boxes, 4], dtype=torch.float32)
        for j in range(num_boxes):
            x, y, w, h = cv.boundingRect(masks[j])
            boxes[j] = torch.tensor([x, y, x + w, y + h])

        # Compile image, label mask, and mask bounding boxes into one package
        data = {}
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        img = torch.as_tensor(np.array(img), dtype=torch.float32)
        classes = torch.as_tensor(np.array(classes), dtype=torch.int64)
        data['boxes'], data['masks'], data['labels'] = boxes, masks, classes

        batch_imgs.append(img)
        batch_data.append(data)

    batch_imgs = torch.stack(batch_imgs, 0)
    batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)

    return batch_imgs, batch_data


def trainModel(device, model, imgs, batchSize=4, epochs=1):
    """
    Prepares and trains the provided model on the given images
    """
    model.load_state_dict(torch.load('t1/e46.torch'))
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    model.train()

    for e in range(47, epochs):
        random.shuffle(imgs)
        for i in range(0, len(imgs), batchSize):
            images, targets = loadData(imgs, i, i + batchSize)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

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


def evalModel(device, model, label_ref):

    # in_features = model.roi_heads.box_predictor.cls_score.in_features

    # model.roi_heads.box_predictor = FastRCNNPredictor(
    #     in_features, num_classes=25)

    model.load_state_dict(torch.load('t1/e29.torch'))
    model.to(device)
    model.eval()

    # images = cv.imread('train/3561_slide-083.jpg')
    images = cv.imread('test/1317440_slide-011.jpg')
    # images = cv.imread('test/203627_slide-019.jpg')
    # images = cv.resize(images, (600, 600), cv.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = [image.to(device) for image in images]

    with torch.no_grad():
        pred = model(images)

    im = images[0].swapaxes(0, 2).swapaxes(
        0, 1).detach().cpu().numpy().astype(np.uint8)
    im2 = im.copy()
    for i in range(len(pred[0]['masks'])):
        msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
        scr = pred[0]['scores'][i].detach().cpu().numpy()
        lab = int(pred[0]['labels'][i].detach().cpu().numpy())
        if scr > 0.75:
            im2[:, :, 0][msk > 0.5] = random.randint(0, 255)
            im2[:, :, 1][msk > 0.5] = random.randint(0, 255)
            im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
            print(label_ref[lab])
            cv.imshow(str(scr), im2)
            cv.waitKey(0)


if __name__ == '__main__':
    from label_definitions import LABEL_NAMES

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    imgSize = [600, 600]
    batchSize = 4
    epochs = 50
    imgs = loadImages('train')

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights='DEFAULT')
    num_classes = 25
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    trainModel(device, model, imgs, batchSize, epochs)
    # evalModel(device, model, LABEL_NAMES)
