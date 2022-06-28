from utils.inverter import StyleGANInverter
import torch


class StyleGANGenerator:
    def __init__(self):
        self.G = self.load_model()

    def _get_tensor_value(self, tensor):
        return tensor.cpu().detach().numpy()

    def load_model(self):
        inverter = StyleGANInverter(
        'styleganinv_bedroom256',
        learning_rate=0.01,
        iteration=100,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        regularization_loss_weight=2.0,
        logger=None)

        return inverter.G

    def synthesis(self, dlatent):
        if type(dlatent) != torch.Tensor:
            dlatent = torch.tensor(dlatent)

        if dlatent.ndim == 2:
            dlatent = dlatent[None]

        if torch.cuda.is_available():
            dlatent = dlatent.to('cuda')

        image = self.G.net.synthesis(dlatent)
        return image

    def process4imshow(self, image):
        image_ = self.G.postprocess(self._get_tensor_value(image))[0]
        return image_