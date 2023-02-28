import torch
import utils


class Infer():

    def __init__(self, model, tta=None):
        self.model = model
        self.tta = tta

    def batch_infer(self, batch):
        if self.tta is None:
            return self.model(batch)
        logits = []
        for i in range(batch.shape[0]):
            images = utils.get_ensemble_image(batch[i], self.mode)
            logit, _ = self.model(images.cuda())
            logit = torch.mean(logit, dim=0, keepdim=True)
            logits.append(logit)
        return torch.cat(logits, dim=0)

