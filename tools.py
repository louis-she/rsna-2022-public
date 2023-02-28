import os
import models
import fire
import torch
import timm
import pandas as pd
import utils
import numpy as np
import torch.nn.functional as F
from plugins import pfbeta
from copy import copy
import cv2
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import dicomsdl
from pydicom.pixel_data_handlers import apply_voi_lut
import pydicom


def _func(item):
    _, item = item
    path = Path("/home/featurize/data/rsna-breast-cancer-detection/train_images") / item.patient_id / f"{item.image_id}.dcm"
    image = utils.load_image_from_dicom(path)
    image = utils.crop_roi(image)
    np.savez_compressed(f"/home/featurize/rsna_datasets/roi/{item.image_id}", data=image)


def generate_dataset(image_size: int = None):
    df = pd.read_csv("/home/featurize/data/rsna-breast-cancer-detection/train.csv", dtype={"patient_id": str, "image_id": str})
    os.makedirs(f"/home/featurize/rsna_datasets/roi", exist_ok=True)

    iterable = df.iterrows()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=lambda: utils.init_decoder()) as pool:
        list(tqdm(
            pool.imap(_func, iterable),
            total=len(df),
        ))


def post_processing(oof_df: str, fold: int, thresholds: str):
    _oof_df = pd.read_csv(oof_df, dtype={"patient_id": str, "image_id": str})
    df = pd.read_csv("./train.csv", dtype={"patient_id": str, "image_id": str})
    df = df.loc[df.fold == fold].reset_index()
    start, end, step = thresholds.split(":")
    start, end, step = float(start), float(end), float(step)

    for thres in np.arange(start, end, step):
        oof_df = copy(_oof_df)
        oof_df["hard_pred"] = oof_df.pred > thres
        for patient_id in oof_df.patient_id.unique():
            patient_df = df.loc[df.patient_id == patient_id]
            if patient_df.loc[patient_df.laterality == 0].shape[0] > 1:
                image_ids = patient_df.loc[patient_df.laterality == 0].image_id
                oof_df.loc[oof_df.image_id.isin(image_ids), "hard_pred"] = oof_df.loc[oof_df.image_id.isin(image_ids)].hard_pred.any()
            if patient_df.loc[patient_df.laterality == 1].shape[0] > 1:
                image_ids = patient_df.loc[patient_df.laterality == 1].image_id
                oof_df.loc[oof_df.image_id.isin(image_ids), "hard_pred"] = oof_df.loc[oof_df.image_id.isin(image_ids)].hard_pred.any()
        print(f"thres: {thres:.2f}, score: {pfbeta(oof_df.label, oof_df.hard_pred, 1)}")


def mix_model(file):
    weights_file = Path(file)

    test_image = torch.rand(1, 1, 1536, 1536)
    net = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=False, in_chans=3, num_classes=1)

    ori_net = models.Convnext(False, False, "nano", 3)
    weights = torch.load(weights_file, map_location="cpu")

    ori_net.load_state_dict(weights)
    ori_net.eval()
    res = ori_net(test_image, None).mean().item()
    print(f"ori model inference res: {res}")

    weights["head.norm.weight"] = weights["cancer_pool.norm.weight"]
    weights["head.norm.bias"] = weights["cancer_pool.norm.bias"]
    weights["head.fc.weight"] = weights["cancer_fc.weight"]
    weights["head.fc.bias"] = weights["cancer_fc.bias"]
    del weights["cancer_pool.norm.weight"]
    del weights["cancer_pool.norm.bias"]
    del weights["cancer_fc.weight"]
    del weights["cancer_fc.bias"]

    net.load_state_dict(weights)
    new_res = net(test_image.repeat(1, 3, 1, 1)).mean()
    print(f"current model inference res: {new_res}")
    assert res == new_res
    torch.save(weights, weights_file.as_posix() + ".new")


if __name__ == '__main__':
    fire.Fire()
