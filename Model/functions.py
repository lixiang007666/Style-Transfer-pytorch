# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as model

style_layers, content_layers = [0, 5, 10, 19, 28], [25]
vgg = model.vgg19(pretrained=True).features
if torch.cuda.is_available() :
    vgg = vgg.cuda()

## ------------------- content loss function ------------------ ##
class ContentLoss(nn.Module) :

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, inputs) :
        self.loss = self.weight * self.criterion(inputs, self.target)
        outputs = inputs.clone()
        return outputs

## ------------------- gram matix function ------------------ ##
class Gram(nn.Module) :

    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, inputs) :
        batch_size, channels, width, height = inputs.size()
        features = inputs.view(batch_size * channels, width * height)
        return torch.mm(features, features.t()) / (batch_size * channels * width * height)

## ------------------- style loss function ------------------ ##
class StyleLoss(nn.Module) :

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, inputs) :
        gram_features = self.gram(inputs)
        self.loss = self.weight * self.criterion(gram_features, self.target)
        outputs = inputs.clone()
        return outputs

## ------------------- total variation denoising loss function ------------------ ##
class TotalVariationDenoisingLoss(nn.Module) :

    def __init__(self, weight) :
        self.weight = weight
        super(TotalVariationDenoisingLoss, self).__init__()

    def forward(self, inputs) :
        self.loss = self.weight * 0.5 * ((inputs[:, :, 1:, :] - inputs[:, :, :-1, :]).abs().mean() +
                            (inputs[:, :, :, 1:] - inputs[:, :, :, :-1]).abs().mean())
        outputs = inputs.clone()
        return outputs

## ------------------- train model ------------------ ##
class StyleTransferModel(nn.Module) :

    def __init__(self, style_img, content_img, base_model = vgg,
                 style_weight = 1e6, content_weight = 1, total_variation_denoising_weight = 10):
        super(StyleTransferModel, self).__init__()
        self.style_img = style_img
        self.content_img = content_img
        self.base_model = base_model
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_denoising_weight = total_variation_denoising_weight
        self.layers = nn.Sequential()

    def generate_layers(self) :
        content_loss_list, style_loss_list, total_variation_denoising_loss_list = [], [], []
        conv2d_index, maxpool2d_index, batchnorm2d_index, relu_index  = \
            1, 1, 1, 1
        name = "TotalVariationDenoisingLoss_{}".format(1)
        total_variation_denoising_loss = TotalVariationDenoisingLoss(self.total_variation_denoising_weight)
        self.layers.add_module(name, total_variation_denoising_loss)
        total_variation_denoising_loss_list.append(total_variation_denoising_loss)
        for idx, layer in enumerate(self.base_model) :
            if isinstance(layer, nn.Conv2d) :
                name = "Conv2d_{}".format(conv2d_index)
                self.layers.add_module(name, layer)

                if idx in content_layers :
                    name = "ContentLloss_{}".format(conv2d_index)
                    target = self.layers(self.content_img)
                    content_loss = ContentLoss(target, self.content_weight)
                    self.layers.add_module(name, content_loss)
                    content_loss_list.append(content_loss)
                elif idx in style_layers :
                    name = "StyleLoss_{}".format(conv2d_index)
                    target = self.layers(self.style_img)
                    target = Gram()(target)
                    style_loss = StyleLoss(target, self.style_weight)
                    self.layers.add_module(name, style_loss)
                    style_loss_list.append(style_loss)

                conv2d_index += 1

            elif isinstance(layer, nn.MaxPool2d) :
                name = "MaxPool2d_{}".format(maxpool2d_index)
                self.layers.add_module(name, layer)
                maxpool2d_index += 1

            elif isinstance(layer, nn.ReLU) :
                name = "Relu_{}".format(relu_index)
                self.layers.add_module(name, layer)
                relu_index += 1

            else :
                name = "BatchNorm2d_{}".format(batchnorm2d_index)
                self.layers.add_module(name, layer)
                batchnorm2d_index += 1

        return content_loss_list, style_loss_list, total_variation_denoising_loss_list

    def forward(self, inputs) :
        outputs = self.layers(inputs)
        return outputs