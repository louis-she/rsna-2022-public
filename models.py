from collections import OrderedDict
import os
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch
import timm
from nextvit.classification.nextvit import nextvit_base, nextvit_small
from copy import deepcopy
try:
    import pytorch_tao as tao
except:
    tao = None
    pass


class EffnetAuxHead(torch.nn.Module):
    def __init__(self, with_aux, backbone="tf_efficientnetv2_b3"):
        super().__init__()
        self.with_aux = with_aux
        use_pretrained = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None
        effnet = timm.create_model(backbone, pretrained=use_pretrained, in_chans=1)
        self.backbone = torch.nn.Sequential(
            effnet.conv_stem, effnet.bn1, effnet.blocks, effnet.conv_head, effnet.bn2
        )
        self.feature_dim = effnet.classifier.in_features
        self.cancer = torch.nn.Linear(self.feature_dim, 1, bias=False)
        if self.with_aux:
            self.density = torch.nn.Linear(self.feature_dim, 1, bias=False)
            self.biopsy = torch.nn.Linear(self.feature_dim, 1, bias=False)
            self.invasive = torch.nn.Linear(self.feature_dim, 1, bias=False)
            self.BIRADS = torch.nn.Linear(self.feature_dim, 4, bias=False)
            self.difficult_negative_case = torch.nn.Linear(
                self.feature_dim, 1, bias=False
            )

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if not self.training or not self.with_aux:
            return self.cancer(x)
        elif self.with_aux:
            return {
                "cancer": self.cancer(x),
                "density": self.density(x),
                "biopsy": self.biopsy(x),
                "invasive": self.invasive(x),
                "BIRADS": self.BIRADS(x),
                "difficult_negative_case": self.difficult_negative_case(x),
            }


class Nano(torch.nn.Module):
    def __init__(self):
        super().__init__()
        use_pretrained = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None
        backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=use_pretrained, in_chans=1
        )
        self.stem = backbone.stem
        self.stage1 = backbone.stages[0]
        self.stage2 = backbone.stages[1]
        self.stage3 = backbone.stages[2]
        self.stage4 = backbone.stages[3]
        self.fc = torch.nn.Linear(80 + 160 + 320 + 640, 1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        feature = torch.cat(
            (
                F.adaptive_avg_pool2d(f1, 1),
                F.adaptive_avg_pool2d(f2, 1),
                F.adaptive_avg_pool2d(f3, 1),
                F.adaptive_avg_pool2d(f4, 1),
            ),
            dim=1,
        ).flatten(1)

        return self.fc(feature)


class Convnext(torch.nn.Module):

    def create_pool_head(self, ori_head, dim, num_classes):
        return deepcopy(ori_head[:3]), torch.nn.Linear(dim, num_classes)

    def __init__(self, with_aux_features, with_aux_targets, variant, in_chans, *args, **kwargs):
        super().__init__()
        self.in_chans = in_chans
        self.with_aux_features = with_aux_features
        self.with_aux_targets = with_aux_targets
        channels, name = {
            "nano": ((80, 120, 320, 640), "convnextv2_nano.fcmae_ft_in22k_in1k"),
            "base": ((128, 256, 512, 1024), "convnextv2_base.fcmae_ft_in22k_in1k"),
        }[variant]

        use_pretrained = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None
        backbone = timm.create_model(
            name, pretrained=use_pretrained, in_chans=in_chans, num_classes=1
        )
        self.stem = backbone.stem
        self.stages = backbone.stages
        feat_dim = channels[-1]
        if self.with_aux_features:
            feat_dim += 40
            self.meta_view = torch.nn.Embedding(6, 8)
            self.meta_age = torch.nn.Embedding(64, 8)
            self.meta_implant = torch.nn.Embedding(2, 8)
            self.meta_site_id = torch.nn.Embedding(2, 8)
            self.meta_laterality = torch.nn.Embedding(2, 8)
        if self.with_aux_targets:
            self.density_pool, self.density_fc = self.create_pool_head(backbone.head, feat_dim, 1)
            self.biopsy_pool, self.biopsy_fc = self.create_pool_head(backbone.head, feat_dim, 1)
            self.invasive_pool, self.invasive_fc = self.create_pool_head(backbone.head, feat_dim, 1)
            self.BIRADS_pool, self.BIRADS_fc = self.create_pool_head(backbone.head, feat_dim, 4)
            self.difficult_negative_case_pool, self.difficult_negative_case_fc = self.create_pool_head(backbone.head, feat_dim, 1)
        self.cancer_pool, self.cancer_fc = self.create_pool_head(backbone.head, feat_dim, 1)

    def forward(self, x, aux_features=None):
        if self.in_chans == 3:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        feature_map = self.stages(x)

        cf = self.cancer_pool(feature_map)
        if self.with_aux_targets:
            df = self.density_pool(feature_map)
            bf = self.biopsy_pool(feature_map)
            if_ = self.invasive_pool(feature_map)
            Bf = self.BIRADS_pool(feature_map)
            dncf = self.difficult_negative_case_pool(feature_map)

        if self.with_aux_features:
            vie_emb = self.meta_view(aux_features["view"]).squeeze(1)
            age_emb = self.meta_age(aux_features["age"]).squeeze(1)
            imp_emb = self.meta_implant(aux_features["implant"]).squeeze(1)
            sid_emb = self.meta_site_id(aux_features["site_id"]).squeeze(1)
            lat_emb = self.meta_laterality(aux_features["laterality"]).squeeze(1)

            cf = torch.cat((cf, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)
            if self.with_aux_targets:
                df = torch.cat((df, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)
                bf = torch.cat((bf, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)
                if_ = torch.cat((if_, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)
                Bf = torch.cat((Bf, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)
                dncf = torch.cat((dncf, vie_emb, age_emb, imp_emb, sid_emb, lat_emb), dim=1)

        ret = {"cancer": self.cancer_fc(cf)}
        if self.with_aux_targets:
            ret["density"] = self.density_fc(df)
            ret["biopsy"] = self.biopsy_fc(bf)
            ret["invasive"] = self.invasive_fc(if_)
            ret["BIRADS"] = self.BIRADS_fc(Bf)
            ret["difficult_negative_case"] = self.difficult_negative_case_fc(dncf)

        return ret["cancer"]


class NextViT(torch.nn.Module):
    def __init__(self, with_aux_features, with_aux_targets, use_checkpoint, *args, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.with_aux_features = with_aux_features
        self.with_aux_targets = with_aux_targets
        if "base" in tao.args.nextvit_checkpoint:
            model = nextvit_base()
        elif "small" in tao.args.nextvit_checkpoint:
            model = nextvit_small()
        else:
            raise RuntimeError()
        model.load_state_dict(torch.load(tao.args.nextvit_checkpoint)["model"])
        self.stem = model.stem
        self.features = model.features
        self.norm = model.norm
        self.avgpool = model.avgpool
        self.proj_head = torch.nn.Linear(1024, 1)

    def forward(self, x, aux_features):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x

# heng'v version
# class NextViT(torch.nn.Module):
#     def __init__(self, with_aux_features, with_aux_targets, variant, use_checkpoint, *args, **kwargs):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
#         self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
#         backbone = nextvit_small(pretrained=True, use_checkpoint=self.use_checkpoint)
#         self.cancer = torch.nn.Linear(1024, 1)

#     def forward(self, x, aux_features):
#         x = (x - self.mean) / self.std
#         x = self.encoder.forward_features(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = torch.flatten(x, 1, 3)
#         cancer = self.cancer(x)
#         return {"cancer": cancer}
