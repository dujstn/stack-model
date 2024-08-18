import torch
import os
import cv2 as cv
import numpy as np
import scipy.io as sio
import random


def loadImageDirs(imgdir: str) -> list[str]:
    """
    Returns a list of all file paths in directory dir.
    """
    return [f'{imgdir}/{path}' for path in os.listdir(f'{imgdir}')]


def getMasks(labels: np.ndarray, imgSize: list[int] = [600, 600], noResize: bool = False) -> list:
    """
    Returns a list of masks for the given label array, resized to the given imgSize.
    If noResize is True, image resizing is skipped.
    """
    masks = []
    mask_classes = []
    for dim in range(24):
        slice_ids = np.unique(labels[:, :, dim])[1:]

        # Check if there are annotations (i.e. more than just background class)
        if slice_ids.size > 0:
            mask = (labels[:, :, dim] > 0).astype(np.uint8)

            if not noResize:
                mask = cv.resize(mask, imgSize, cv.INTER_NEAREST)
            masks.append(mask)
            mask_classes.append(dim + 1)
    return masks, mask_classes


def loadData(dirs: list[str], start: int, end: int, imgSize: list[int] = [600, 600], noResize: bool = False) -> list:
    """
    Loads end - start number of resized images into Torch tensors, with annotated ground
    truth masks. Image resizing is skipped if noResize is True.
    """
    batch_imgs, batch_data = [], []

    for i in range(start, min(end, len(dirs))):
        # Read and resize image to correct dimensions
        img = cv.imread(dirs[i])
        if not noResize:
            img = cv.resize(img, imgSize, cv.INTER_LINEAR)

        # Load and prepare masks from .mat file
        full_file_dir = dirs[i].split('/')[-1]
        fileName = full_file_dir[:-4]
        labels = sio.loadmat(f'labels/{fileName}.mat')['labels']
        masks, classes = getMasks(labels, imgSize, noResize)

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

    if noResize:
        batch_imgs = torch.nested.as_nested_tensor(batch_imgs)
    else:
        batch_imgs = torch.stack(batch_imgs, 0)
        batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)

    return dirs, batch_imgs, batch_data


def visualizePred(img: torch.tensor, pred: torch.tensor, label_ref: dict) -> None:
    og_img = img.copy()
    for i in range(len(pred[0]['masks'])):
        msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
        scr = pred[0]['scores'][i].detach().cpu().numpy()
        box = pred[0]['boxes'][i].detach().cpu().numpy()
        lab = int(pred[0]['labels'][i].detach().cpu().numpy())
        if scr > 0.85:
            img[:, :, 0][msk > 0.65] = random.randint(0, 255)
            img[:, :, 1][msk > 0.65] = random.randint(0, 255)
            img[:, :, 2][msk > 0.65] = random.randint(0, 255)
            cv.imshow(f'{str(scr)}, {label_ref[lab]}', np.hstack([og_img, img]))
            cv.waitKey(0)
