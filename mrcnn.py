from __future__ import annotations
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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


def loadData(imgs: list[str], batchSize: int = 2, imgSize: list[int] = [600, 600]) -> None:
    """
    Loads batchSize number of images into Torch tensors, with annotated ground
    truth masks.
    """
    batch_imgs = []
    batch_data = []

    for i in range(batchSize):

        # Read and resize image to correct dimensions
        img = cv.imread(imgs[i])
        img = cv.resize(img, imgSize, cv.INTER_LINEAR)

        # Load and prepare masks from .mat file
        masks = []
        file_full = imgs[i].split('/')[-1]
        fileName = file_full[:-4]
        labels = sio.loadmat(f'labels/{fileName}.mat')['labels']
        label_ids = np.unique(labels)

        for dim in range(24):
            slice_ids = np.unique(labels[:, :, dim])[1:]
            if slice_ids.size > 0:
                mask = (labels[:, :, dim] > 0).astype(np.uint8)
                mask = cv.resize(mask, imgSize, cv.INTER_NEAREST)
                masks.append(mask)

                # Build binding box record of dimension [len(masks), 4] (x, y, x + w, y + h)
        num_boxes = len(masks)
        boxes = torch.zeros([num_boxes, 4], dtype=torch.float32)
        for i in range(num_boxes):
            x, y, w, h = cv.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x + w, y + h])

            # Compile image, label mask, and mask bounding boxes into one package
        data = {}
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        label_ids = torch.as_tensor(label_ids, dtype=torch.int64)
        data['boxes'], data['masks'], data['labels'] = boxes, masks, label_ids

        batch_imgs.append(img)
        batch_data.append(data)

    batch_imgs = torch.stack(batch_imgs, 0)
    batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)

    return batch_imgs, batch_data


if __name__ == '__main__':
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    batchSize = 1
    imgSize = [600, 600]

    imgs = loadImages('train')
    masks = loadData(imgs)
