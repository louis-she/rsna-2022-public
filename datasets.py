import os
from pathlib import Path
import random
import torch
import numpy as np
import cv2
import utils
import pytorch_tao as tao
from albumentations.pytorch import ToTensorV2
import albumentations as A

logger = tao.get_logger(__name__)

pixel_wise_aug = [
    A.HueSaturationValue(),
    A.CLAHE(),
    A.RandomContrast(),
    A.RandomGamma(),
    A.RandomBrightness(),
    A.Blur(),
    A.MedianBlur(),
    A.ImageCompression(),
]


class Dataset(torch.utils.data.Dataset):
    def aug_v0(self):
        augs = []
        augs.append(A.RandomRotate90(p=0.5))
        augs.append(A.HorizontalFlip(p=0.5))
        augs.append(A.VerticalFlip(p=0.5))
        # augs.append(A.Rotate(limit=90, p=0.5))
        return A.Compose(augs)

    def aug_v1(self):
        augs = []
        augs.append(A.RandomRotate90(p=0.5))
        augs.append(A.HorizontalFlip(p=0.5))
        augs.append(A.VerticalFlip(p=0.5))
        augs.append(A.Rotate(limit=90, p=0.5))
        return A.Compose(augs)

    def aug_v2(self):
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.OneOf(pixel_wise_aug),
                        A.SomeOf(pixel_wise_aug, n=2),
                        A.SomeOf(pixel_wise_aug, n=3),
                        A.SomeOf(pixel_wise_aug, n=4),
                        A.SomeOf(pixel_wise_aug, n=5),
                    ],
                    p=0.7,
                ),
            ]
        )

    def aug_none(self):
        return A.NoOp()

    def __init__(self, df, aug="none", training=False):
        self.df = df
        self.aug = getattr(self, "aug_" + aug)()
        self.training = training
        self.neg_df = self.df.loc[self.df.cancer == 0]
        self.pos_df = self.df.loc[self.df.cancer == 1]
        self.fp_df = self.df.loc[(self.df.cancer == 0) & (self.df.pred > 0.3)]
        self.dataset_randomness = np.cumsum(tao.args.dataset_randomness)

    def __len__(self):
        if self.training:
            # RSNA 正样本数 x 1.5 x neg_ratio
            return len(self.df.loc[(self.df.fold < 10) & (self.df.cancer == 1)]) * tao.args.neg_ratio * 2
        return len(self.df)

    def load_image(self, item):
        return utils.load_original_image(item)

    def get_subset(self):
        # 0.1  prob totally random
        # 0.3  prob from fold  0 - 10   RSNA 数据集
        # 0.1  prob from fold 10 - 20   CMMD 中国乳腺摄影数据库
        # 0.15 prob from fold 20 - 30   INbreast 质量可以
        # 0.1 prob from fold 30 - 40    King-Abdulaziz-University-Mammogram-Dataset 质量一般
        # 0.25  prob from fold 40 - 50  vindr mammo 外部数据质量最佳
        r = random.random()
        if r < self.dataset_randomness[0]:
            sub_df = self.df
        elif r < self.dataset_randomness[1]:
            sub_df = self.df.loc[self.df.fold < 10]  # rsna
        elif r < self.dataset_randomness[2]:
            sub_df = self.df.loc[(self.df.fold >= 10) & (self.df.fold < 20)]  # CMMD
        elif r < self.dataset_randomness[3]:
            sub_df = self.df.loc[(self.df.fold >= 20) & (self.df.fold < 30)]  # INbreast
        elif r < self.dataset_randomness[4]:
            sub_df = self.df.loc[(self.df.fold >= 30) & (self.df.fold < 40)]  # King-Abdulaziz-University-Mammogram-Dataset
        else:
            sub_df = self.df.loc[(self.df.fold >= 40) & (self.df.fold < 50)]  # vindr mammo
        return sub_df

    def __getitem__(self, idx):
        if self.training:
            sub_df = self.get_subset()
            if sub_df.shape[0] == 0:
                sub_df = self.df
            if idx % tao.args.neg_ratio == 0:
                item = sub_df.loc[sub_df.cancer == 1].sample(1).iloc[0]
            else:
                item = sub_df.loc[sub_df.cancer == 0].sample(1).iloc[0]
        else:
            item = self.df.iloc[idx]

        image = self.load_image(item)
        # crop RoI -> pad square -> resize to tao.args.image_size -> aug(rotate, flip)
        image = utils.crop_roi(image)
        long_edge = max(image.shape[:2])
        pad_fn = A.PadIfNeeded(long_edge, long_edge, border_mode=cv2.BORDER_CONSTANT, position="random" if self.training else "center", value=0, always_apply=True, p=1.0)
        image = pad_fn(image=image)["image"]
        image = cv2.resize(image, tao.args.image_size)
        image = self.aug(image=image)["image"]
        image = image / 255.0

        target = item.cancer

        return {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "target": torch.tensor([target], dtype=torch.float),
            "patient_id": str(item.patient_id),
            "image_id": str(item.image_id),
            "laterality": str(item.laterality),
            "site_id": str(item.site_id),
        }
