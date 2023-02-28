# created with tao mini template
import os
from pathlib import Path
import random
from ignite import distributed as idist

idist.initialize("nccl")
import warnings

warnings.filterwarnings("ignore")

import itertools
import pytorch_tao as tao
from typing import List
from pytorch_tao.helper import seed_everything, stride_event_filter
from ignite.engine import Events
import pandas as pd
import torch
import models
import torch.nn.functional as F
from ignite.metrics import RunningAverage
import utils
import plugins
import timm
from timm.models.layers import convert_sync_batchnorm
from torch.optim.lr_scheduler import OneCycleLR
from ignite.metrics import Precision, Recall
from pytorch_tao.plugins import (
    Metric,
    OutputRecorder,
    ProgressBar,
    Scheduler,
    Checkpoint,
)
from pytorch_tao.trackers import NeptuneTracker
from sklearn.preprocessing import LabelEncoder
import datasets
from functools import partial
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler

# 1. define the parameters
@tao.arguments
class _:
    max_epochs: int = tao.arg(default=20)
    train_folds: List[int] = tao.arg(default=[0, 1])
    val_folds: List[int] = tao.arg(default=[2])
    batch_size: int = tao.arg(default=64)
    aug: str = tao.arg(default="none")
    optimizer: str = tao.arg(default="Adam@lr=3e-4")
    seed: int = tao.arg(default=777)
    model: str = tao.arg(default="resnet18")
    tta: str = tao.arg(default=None)
    base_dir: str = tao.arg(default="/home/featurize/rsna_datasets/1024")
    save_chk: int = tao.arg(default=None)
    val_filter: str = tao.arg(default="1:1:-1")
    neg_ratio: int = tao.arg(default=4)
    hard_ratio: float = tao.arg(default=0.0)
    accumulate: int = tao.arg(default=1)
    with_aux_targets: bool = tao.arg(default=False)
    with_aux_features: bool = tao.arg(default=False)
    image_size: List[int] = tao.arg(default=None)
    fp_threshold: float = tao.arg(default=0.3)
    nextvit_checkpoint: str = tao.arg(default="")
    use_checkpoint: bool = tao.arg(default=False)
    variant: str = tao.arg(default=None)
    enable_analysis: bool = tao.arg(default=False)
    in_chans: int = tao.arg(default=1)
    swa_lr: float = tao.arg(default=3e-5)
    swa_epochs: int = tao.arg(default=0)
    swa_freq: int = tao.arg(default=4)
    oversample_site0: float = tao.arg(default=1.0)
    debug: bool = tao.arg(default=False)
    mixup_warmup: float = tao.arg(default=0.3)
    early_stop_thres: float = tao.arg(default=0.1)
    dataset_randomness: List[float] = tao.arg(default=[0.1, 0.3, 0.1, 0.15, 0.1, 0.25])


seed_everything(tao.args.seed)

logger = tao.get_logger(__name__)

if tao.args.image_size is not None and "roi" not in tao.args.base_dir:
    logger.error("image_size is only valid for roi dataset, is the set not correct?")

try:
    import nextvit.classification.nextvit
except Exception as e:
    logger.warn(f"import nexvit failed with {e}")

tracker = NeptuneTracker(tao.name)
tao.set_tracker(tracker)

df = pd.read_csv(
    "/home/featurize/work/rsna/train.csv",
    dtype={"patient_id": str, "image_id": str, "laterality": str},
)

if tao.args.debug:
    df = df.loc[df.debug == 1].reset_index()
    tao.args.val_filter = "1:1:-1"

train_df = df.loc[df.fold.isin(tao.args.train_folds)]
val_df = df.loc[df.fold.isin(tao.args.val_folds)]

if tao.args.val_folds[0] in tao.args.train_folds:
    logger.warn("val fold is in train folds, enable fake validation mode")
    val_df = val_df.loc[val_df.debug == 1].reset_index()

if tao.args.oversample_site0 != 1.0:
    site0_pos = train_df.loc[
        (train_df.site_id == 0) & (train_df.cancer == 1)
    ].reset_index()
    site0_neg = train_df.loc[
        (train_df.site_id == 0) & (train_df.cancer == 0)
    ].reset_index()
    site0_pos_num = int(site0_pos.shape[0] * tao.args.oversample_site0)
    site0_neg_num = int(site0_neg.shape[0] * tao.args.oversample_site0)
    ori_sample_num = train_df.shape[0]
    ori_pos_sample = train_df.loc[train_df.cancer == 1].shape[0]
    train_df = pd.concat(
        [
            train_df.loc[train_df.site_id != 0],
            site0_pos.sample(site0_pos_num, random_state=tao.args.seed, replace=True),
            site0_neg.sample(site0_neg_num, random_state=tao.args.seed, replace=True),
        ]
    )
    logger.info(
        "[oversample_site0] ori sample num: {}, ori pos sample: {}, site0_pos_num, {}, site0_neg_num: {}, new sample num: {}, new pos sample: {}".format(
            ori_sample_num,
            ori_pos_sample,
            site0_pos_num,
            site0_neg_num,
            train_df.shape[0],
            train_df.loc[train_df.cancer == 1].shape[0],
        )
    )

# 数据预处理，删除训练集中的 false positive，对于同一个病人 + laterality，保留 oof_prob > positive_thres(0.3 default)
# 的样本，如果一个小于 0.3 的样本都没有，则直接保留 oof_prob 的最大值。
for patient_id, laterality in itertools.product(
    train_df.loc[(train_df.cancer == 1) & (train_df.fold < 10)].patient_id.unique(), ("0", "1")
):
    sub_df = train_df.loc[
        (train_df.patient_id == patient_id)
        & (train_df.laterality == laterality)
        & (train_df.cancer == 1)
    ]
    if sub_df.shape[0] == 0:
        continue
    to_delete_index = sub_df.loc[sub_df.oof_prob < tao.args.fp_threshold].index
    if len(to_delete_index) == sub_df.shape[0]:
        to_delete_index = sub_df.loc[sub_df.oof_prob != sub_df.oof_prob.max()].index
    train_df.drop(to_delete_index, inplace=True)

# 2. Create the necessary pytorch components
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

Loader = partial(idist.auto_dataloader, batch_size=tao.args.batch_size, num_workers=24)
train_loader = Loader(
    datasets.Dataset(
        train_df, aug=tao.args.aug, training=True
    ),
    shuffle=True,
)
val_loader = Loader(datasets.Dataset(val_df, aug="none"))

model_name, kwargs = utils.parse_arg(
    tao.args.model, {"pretrained": True, "num_classes": 1, "in_chans": 1}
)
if hasattr(models, model_name):
    model = getattr(models, model_name)(
        with_aux_features=tao.args.with_aux_features,
        with_aux_targets=tao.args.with_aux_targets,
        variant=tao.args.variant,
        use_checkpoint=tao.args.use_checkpoint,
        in_chans=tao.args.in_chans,
    )
else:
    model = timm.create_model(model_name, **kwargs)
optim_type, optim_args = utils.parse_arg(tao.args.optimizer)
optimizer = getattr(torch.optim, optim_type)(params=model.parameters(), **optim_args)

model = convert_sync_batchnorm(model)
model = idist.auto_model(model)

optimizer = idist.auto_optim(optimizer)
_stride_event_filter = stride_event_filter(tao.args.val_filter)

# 3. Create trainer and train / val forward functions
trainer = tao.Trainer(
    device=device,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    val_event=lambda _, e: _stride_event_filter(_, e) or e >= tao.args.max_epochs,
)


@trainer.train(
    amp=True,
    fields=["image", "target"],
    accumulate=tao.args.accumulate,
)
def _train(image, target):
    logits = model(image)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return dict(loss=loss)


infer = utils.Infer(model, tta=tao.args.tta)


@trainer.eval(
    fields=[
        "image",
        "target",
        "image_id",
        "patient_id",
        "laterality",
        "site_id"
    ]
)
def _eval(image, target, image_id, patient_id, laterality, site_id):
    _model = swa.model() if tao.args.swa_epochs > 0 and swa.is_enabled() else model
    logits = _model(image)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return dict(
        val_loss=loss,
        y_pred=logits,
        y=target,
        image_id=image_id,
        patient_id=patient_id,
        laterality=laterality,
        site_id=site_id,
    )


# 4. start to train
_scheduler = OneCycleLR(
    optimizer,
    optimizer.param_groups[0]["lr"],
    pct_start=0.05,
    epochs=tao.args.max_epochs,
    steps_per_epoch=len(train_loader),
)


@trainer.train_engine.on(Events.EPOCH_COMPLETED)
def _set_epoch(engine):
    if hasattr(train_loader.sampler, "set_epoch"):
        logger.info(f"called set_epoch of sampler by {engine.state.epoch + 1}")
        train_loader.sampler.set_epoch(engine.state.epoch + 1)


@trainer.train_engine.on(Events.EPOCH_STARTED)
def _early_stop(engine):
    if (
        trainer.state.epoch > 6
        and "pf1" in trainer.val_engine.state.metrics
        and trainer.val_engine.state.metrics["pf1"] is not None
        and trainer.val_engine.state.metrics["pf1"] < tao.args.early_stop_thres
    ):
        logger.info("pf1 is too low, early stop this run")
        idist.finalize()
        trainer.train_engine.terminate()


if tao.args.swa_lr is not None:
    swa = plugins.SWA(tao.args.swa_lr, tao.args.max_epochs, tao.args.swa_freq)
    trainer.use(swa)

trainer.use(Scheduler(_scheduler))
trainer.use(ProgressBar("loss"), at="train")
trainer.use(Metric("pf1", plugins.Pfbeta(), tune=True))
trainer.use(Metric("val_loss", RunningAverage(output_transform=lambda x: x["val_loss"]), direction="min"))
if len(tao.args.train_folds) != 3 and tao.args.enable_analysis:
    trainer.use(plugins.BestModelAnalysis("pf1", model, val_loader))
trainer.use(OutputRecorder("loss"), at="train")
if tao.args.save_chk:
    trainer.use(Checkpoint("pf1", {"model": model}, n_saved=tao.args.save_chk))

trainer.use(tracker)
trainer.fit(max_epochs=tao.args.max_epochs + tao.args.swa_epochs)

idist.finalize()
