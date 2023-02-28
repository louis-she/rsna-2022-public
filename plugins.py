import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Iterator
from ignite.metrics import Metric
import pytorch_tao as tao
import torch
import uuid
import ignite.distributed as idist
import pandas as pd
from pytorch_tao.plugins import TrainPlugin, BasePlugin
from torch.optim.swa_utils import AveragedModel, SWALR
import utils
import numpy as np
import io
from ignite.engine import Events

logger = tao.get_logger(__name__)


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    if ctp + cfp == 0:
        return 0
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


class BestModelAnalysis(TrainPlugin):
    def __init__(self, metric_name, model, dataloader):
        self.checkpoint_filename = utils.sync_masters(
            "/tmp/" + uuid.uuid4().hex[:8] + ".pt"
        )
        self.best_score = 0
        self.best_epoch = 0
        self.dataloader = dataloader
        self.metric_name = metric_name
        self.model = model

    @idist.one_rank_only()
    @tao.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(self):
        try:
            score = self.trainer.val_engine.state.metrics[self.metric_name]
        except KeyError:
            return
        if score > self.best_score:
            logger.info(
                "Best model found at epoch %d, %s: %.4f",
                self.trainer.state.epoch,
                self.metric_name,
                score,
            )
            self.best_score = score
            self.best_epoch = self.trainer.state.epoch
            torch.save(self.model.state_dict(), self.checkpoint_filename)

    @tao.on(Events.COMPLETED)
    def on_completed(self):
        logger.info(
            f"best score: {self.best_score:.4f} at epoch {self.best_epoch}, generating report..."
        )
        try:
            self.model.load_state_dict(
                torch.load(self.checkpoint_filename, map_location="cpu")
            )
        except:
            logger.info("do not find checkpint, quit analyze")
            return
        self.model.cuda()
        self.model.eval()
        self._labels = []
        self._predictions = []
        self._image_ids = []
        self._patient_ids = []
        self._laterality = []
        with torch.no_grad():
            for batch in self.dataloader:
                for i, (image, label, image_id, patient_id, laterality) in enumerate(zip(
                    batch["image"],
                    batch["target"],
                    batch["image_id"],
                    batch["patient_id"],
                    batch["laterality"],
                )):
                    aux_features = {}
                    for key in batch["aux_features"].keys():
                        aux_features[key] = torch.stack([batch["aux_features"][key][i]] * 7).cuda()

                    horizontal_flip = torch.flip(image, [2])
                    vertical_flip = torch.flip(image, [1])
                    horizontal_vertical_flip = torch.flip(image, [1, 2])
                    rotate90 = torch.rot90(image, 1, [1, 2])
                    rotate180 = torch.rot90(image, 2, [1, 2])
                    rotate270 = torch.rot90(image, 3, [1, 2])
                    tta_images = torch.stack(
                        [
                            image,
                            horizontal_flip,
                            vertical_flip,
                            horizontal_vertical_flip,
                            rotate90,
                            rotate180,
                            rotate270,
                        ],
                        dim=0,
                    )
                    probs = (
                        torch.sigmoid(self.model(tta_images, aux_features)["cancer"].squeeze()).cpu().tolist()
                    )
                    self._predictions.append(probs)
                    self._labels.append(label.item())
                    self._image_ids.append(image_id)
                    self._patient_ids.append(patient_id)
                    self._laterality.append(laterality)
        labels = utils.all_gather_list(self._labels)
        predictions = utils.all_gather_list(self._predictions)
        predictions = list(zip(*predictions))
        image_ids = utils.all_gather_list(self._image_ids)
        patient_ids = utils.all_gather_list(self._patient_ids)
        laterality = utils.all_gather_list(self._laterality)

        if idist.get_rank() > 0:
            return None

        oof_df = pd.DataFrame.from_dict(
            {
                "image_id": image_ids,
                "patient_id": patient_ids,
                "laterality": laterality,
                "label": labels,
                "prediction_ori": predictions[0],
                "prediction_hf": predictions[1],
                "prediction_vf": predictions[2],
                "prediction_hvf": predictions[3],
                "prediction_r90": predictions[4],
                "prediction_r180": predictions[5],
                "prediction_r270": predictions[6],
            }
        )

        tao.tracker.add_tabular(f"best_tta", oof_df)
        os.unlink(self.checkpoint_filename)


class Pfbeta(Metric):
    def __init__(self):
        super().__init__(
            output_transform=lambda x: (
                x["y_pred"],
                x["y"],
                x["image_id"],
                x["patient_id"],
                x["laterality"],
                x["site_id"]
            )
        )
        self.max_score = 0
        self.num = 0

    def update(self, output):
        y_pred, y, image_id, patient_id, laterality, site_id = output
        y_pred = torch.sigmoid(y_pred)
        self._labels.append(y.squeeze(1).cpu())
        self._predictions.append(y_pred.squeeze(1).cpu())
        self._image_ids += image_id
        self._site_ids += site_id
        self._patient_ids += patient_id
        self._laterality += laterality

    def compute(self):
        labels = idist.all_gather(torch.concat(self._labels))
        predictions = idist.all_gather(torch.concat(self._predictions))
        image_ids = utils.all_gather_list(self._image_ids)
        site_ids = utils.all_gather_list(self._site_ids)
        patient_ids = utils.all_gather_list(self._patient_ids)
        laterality = utils.all_gather_list(self._laterality)

        if idist.get_rank() > 0:
            return None

        self.num += 1

        predictions = predictions.cpu().tolist()
        labels = labels.cpu().tolist()

        all_scores = []
        all_thresholds = []
        for threshold in np.concatenate(
            [np.arange(0.1, 0.8, 0.05), np.arange(0.8, 1, 0.01)], axis=0
        ):
            threshold_score = pfbeta(
                labels, (torch.tensor(predictions) > threshold).tolist(), 1
            )
            logger.info(
                f"naive score with threshold {threshold:.2f}: {threshold_score}"
            )
            all_scores.append(threshold_score)
            all_thresholds.append(threshold)

        max_idx = np.argmax(all_scores)
        thres_score = all_scores[max_idx]
        tao.tracker.add_points(
            {
                "naive_threshold": all_thresholds[max_idx],
                "naive_scores": all_scores[max_idx],
            }
        )

        # 使用 patient 聚合预测结果
        oof_df = pd.DataFrame.from_dict(
            {
                "image_id": image_ids,
                "patient_id": patient_ids,
                "laterality": laterality,
                "label": labels,
                "prediction": predictions,
                "site_id": site_ids
            }
        )

        patient_df = {"patient_id": [], "preds": [], "labels": [], "laterality": [], "site_ids": []}
        for patient_id in oof_df.patient_id.unique():
            for laterality in range(2):
                sub_df = oof_df.loc[
                    (oof_df.patient_id == patient_id)
                    & (oof_df.laterality == str(laterality))
                ]
                agg_pred = sub_df.prediction.max()
                patient_df["patient_id"].append(patient_id)
                patient_df["laterality"].append(laterality)
                patient_df["preds"].append(agg_pred)
                patient_df["labels"].append(sub_df.iloc[0].label)
                patient_df["site_ids"].append(str(sub_df.iloc[0].site_id))
        patient_df = pd.DataFrame.from_dict(patient_df)
        patient_df_site0 = patient_df.loc[patient_df.site_ids == "0.0"].reset_index()
        patient_df_site1 = patient_df.loc[patient_df.site_ids == "1.0"].reset_index()
        for _patient_df, _name in [(patient_df_site0, "site0"), (patient_df_site1, "site1"), (patient_df, "all")]:
            all_scores = []
            all_thresholds = []
            for threshold in np.concatenate(
                [np.arange(0.1, 0.8, 0.05), np.arange(0.8, 1, 0.01)], axis=0
            ):
                threshold_score = pfbeta(_patient_df.labels, _patient_df.preds > threshold, 1)
                logger.info(f"[{_name}] score with threshold {threshold:.2f}: {threshold_score}")
                all_scores.append(threshold_score)
                all_thresholds.append(threshold)

            max_idx = np.argmax(all_scores)
            thres_score = all_scores[max_idx]

            tao.tracker.add_points(
                {
                    "thresholds": all_thresholds[max_idx],
                    f"scores_{_name}": all_scores[max_idx],
                }
            )
        tao.tracker.add_histogram("preds_hist", oof_df.prediction, bins=100)
        tao.tracker.add_histogram("patient_preds_hist", patient_df.preds, bins=100)
        tao.tracker.add_tabular(f"oof_df.{self.num}", oof_df)
        tao.tracker.add_tabular(f"patient_df.{self.num}", patient_df)
        return thres_score

    def reset(self):
        self._labels = []
        self._predictions = []
        self._image_ids = []
        self._patient_ids = []
        self._laterality = []
        self._site_ids = []


class SWA(TrainPlugin):

    def __init__(self, swa_lr: float, swa_start: int, swa_freq: int = 1):
        super().__init__()
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr

    def model(self):
        return self.swa_model

    def is_enabled(self):
        return self.trainer.state.epoch > self.swa_start

    # @tao.on(Events.EPOCH_COMPLETED)
    # def update_bn(self):
    #     torch.optim.swa_utils.update_bn(self.loader, self.swa_model)

    @tao.on(Events.STARTED)
    def create_swa_model(self):
        logger.info("Creating swa model ...")
        self.swa_model = AveragedModel(self.trainer.model)
        self.swa_scheduler = SWALR(self.trainer.optimizer, swa_lr=self.swa_lr)

    @tao.on(lambda self: Events.ITERATION_COMPLETED(every=self.swa_freq))
    def update_parameters(self):
        if self.is_enabled() and self.trainer.state.iteration % self.swa_freq == 0:
            self.swa_model.update_parameters(self.trainer.model)
            self.swa_scheduler.step()

    def state_dict(self):
        return {
            "swa_model": self.swa_model.state_dict(),
            "swa_scheduler": self.swa_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.swa_model.load_state_dict(state_dict["swa_model"])
        self.swa_scheduler.load_state_dict(state_dict["swa_scheduler"])
