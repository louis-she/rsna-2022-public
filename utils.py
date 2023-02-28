import random
import cv2
import torch
import numpy as np
from pathlib import Path
import pydicom
from pydicom.filebase import DicomBytesIO
import torch.distributed as dist
import dicomsdl
try:
    import redis
    storage = redis.Redis()
except ImportError:
    pass


decoder = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crop_roi(image):
    copy_image = image.copy()
    copy_image[copy_image < 20] = 0
    return image[:, copy_image.sum(axis=0) > 1000]


def sync_masters(var):
    objects = [var]
    torch.distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def normalised_to_8bit(image):
    xmin = image.min()
    xmax = image.max()
    norm = np.empty_like(image, dtype=np.uint8)
    dicomsdl.util.convert_to_uint8(image, norm, xmin, xmax)
    return norm


def parse_arg(s: str, default_args=None):
    results = s.split("@")
    if len(results) == 1:
        main = results[0]
        args = {}
    else:
        main = results[0]
        args = [x.split("=") for x in results[1].split("&")]
    args = {key: eval(value) for key, value in args}
    if default_args is None:
        default_args = {}
    default_args.update(args)
    return main, default_args


def load_image(item, base_dir):
    return np.load(f"{base_dir}/{item.image_id}.npz")["data"]


def flatten(xss):
    return [x for xs in xss for x in xs]


def init_decoder():
    global decoder
    import nvjpeg2k
    decoder = nvjpeg2k.Decoder()


def load_dicom(path, force_slow=False):
    dcmfile = pydicom.dcmread(path)
    reverse = dcmfile.PhotometricInterpretation == "MONOCHROME1"
    if not force_slow and dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(path, 'rb') as f:
            raw = DicomBytesIO(f.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        return decoder.decode(hackedbitstream), reverse, dcmfile
    else:
        return dcmfile.pixel_array, reverse, dcmfile


def dicom_to_2k(dicom_file, output_dir):
    """将某个 dicom 文件转换成 2K jepg 图像，并保存为文件，方便 dali 读取解压

    返回 True 保存成功，False 保存失败
    """
    dicom_file = Path(dicom_file)
    output_dir = Path(output_dir)
    dcmfile = pydicom.dcmread(dicom_file)
    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(dicom_file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(output_dir / f"{dicom_file.name}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)
        return True
    else:
        return False


def get_ensemble_image(image, mode="flip+rotate"):
    assert len(image.shape) == 3 and (image.shape[0] == 3 or image.shape[0] == 1)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    images = [image]

    if "flip" in mode:
        images += [
            image[:, :, ::-1],
            image[:, ::-1, :]
        ]

    if "rotate" in mode:
        transpose_image = image.transpose(0, 2, 1)
        images += [
            transpose_image,
            transpose_image[:, :, ::-1],
            transpose_image[:, ::-1, :],
        ]

    return torch.from_numpy(np.stack(images, axis=0))


class Infer():

    def __init__(self, model, tta=None):
        self.model = model
        self.tta = tta

    def batch_infer(self, batch):
        if self.tta is None:
            return self.model(batch)
        logits = []
        for i in range(batch.shape[0]):
            images = get_ensemble_image(batch[i], self.tta)
            logit = self.model(images.cuda())
            logit = torch.mean(logit, dim=0, keepdim=True)
            logits.append(logit)
        return torch.cat(logits, dim=0)


def all_gather_list(data):
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return data
    data_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(data_list, data)
    return flatten(data_list)


def convert_sync_batchnorm(module, process_group=None):
    # convert both BatchNorm and BatchNormAct layers to Synchronized variants
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if isinstance(module, BatchNormAct2d):
            # convert timm norm + act layer
            module_output = SyncBatchNormAct(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group=process_group,
            )
            # set act and drop attr from the original module
            module_output.act = module.act
            module_output.drop = module.drop
        else:
            # convert standard BatchNorm layers
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output


def load_image_from_dicom(path):
    image, reverse, _ = load_dicom(path, force_slow=True)
    image = image.astype(np.float32)
    image = normalised_to_8bit(image)
    if reverse:
        image = 255 - image
    return image


def load_original_image(item):
    """ 返回 0 - 255 的 uint8 格式的单通道图片 """
    if item.fold < 10:
        # return load_image_from_dicom(f"/home/featurize/data/rsna-breast-cancer-detection/train_images/{item.patient_id}/{item.image_id}.dcm")
        return np.load(f"/home/featurize/rsna_datasets/roi/{item.image_id}.npz")["data"]
    elif item.fold >= 10 and item.fold < 20:
        # CMMD 数据集
        patient_id, suffix = item.image_id.split("_")
        path = list(Path(f"/home/featurize/work/rsna/data/CMMD/TheChineseMammographyDatabase/CMMD/{patient_id}").glob("**/*.dcm"))[int(suffix) - 1]
        return load_image_from_dicom(path)
    elif item.fold >= 20 and item.fold < 30:
        path = Path(f"/home/featurize/work/rsna/data/INbreast-2012/INbreast Release 1.0/AllDICOMs/{item.image_id}")
        return load_image_from_dicom(path)
    elif item.fold >= 30 and item.fold < 40:
        path = Path(f"/home/featurize/work/rsna/data/King-Abdulaziz-University-Mammogram-Dataset/{item.image_id}")
        return cv2.imread(path.as_posix()).mean(axis=2).astype(np.uint8)
    elif item.fold >= 40 and item.fold < 50:
        path = Path(f"/home/featurize/work/rsna/data/vindr-mammo/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/{item.image_id}")
        return load_image_from_dicom(path)
