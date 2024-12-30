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


def eval_preprocess(path):
    image = cv.imread(path)
    image = cv.resize(image, (900, 600), cv.INTER_LINEAR)
    image = torch.as_tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)
    return image.to(device)


def evalModel(device, model, label_ref):

    model.load_state_dict(torch.load('t1/e49.torch'))
    model.to(device)
    model.eval()

    images = []
    paths = []
    for path in os.listdir('test/'):
        path = f'test/{path}'
        images.append(eval_preprocess(path))
        paths.append(path)

    with torch.no_grad():
        pred = model(images[:20])

    for i in range(len(pred)):
        for j in range(len(pred[i]['masks'])):
            im = images[i].swapaxes(0, 2).swapaxes(
                0, 1).detach().cpu().numpy().astype(np.uint8)
            im2 = im.copy()
            msk = pred[i]['masks'][j, 0].detach().cpu().numpy()
            scr = pred[i]['scores'][j].detach().cpu().numpy()
            lab = int(pred[i]['labels'][j].detach().cpu().numpy())
            pred_label = label_ref[lab]
            if scr > 0.75:

                red_overlay = np.zeros_like(im2)
                red_overlay[:, :, 2] = 255
                alpha = 0.60
                im2[msk > 0.5] = (1 - alpha) * im2[msk > 0.5] + \
                    alpha * red_overlay[msk > 0.5]

                cv.rectangle(im2, (675, 475), (900, 600), (255, 0, 0), -1)
                cv.putText(im2, pred_label, (700, 525),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.putText(im2, str(scr), (700, 575),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                try:
                    os.mkdir(f'output/{paths[i].split('/')[1][:-4]}')
                except:
                    pass
                write_path = f"output/{paths[i].split('/')[1][:-4]}/{
                    pred_label}.jpg"
                cv.imwrite(write_path, im2)

        cv.rectangle(im, (675, 475), (900, 600), (255, 0, 0), -1)
        cv.putText(im, "original", (700, 525),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imwrite(f"output/{paths[i].split('/')[1][:-4]}/original.jpg", im)

        print(f'Finished {i}th image')


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

    # trainModel(device, model, imgs, batchSize, epochs)
    evalModel(device, model, LABEL_NAMES)
