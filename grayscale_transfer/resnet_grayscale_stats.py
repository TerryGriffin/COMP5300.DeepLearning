# COMP.5300 Deep Learning
# Terry Griffin
#
# Compare the number of parameters between a three channel input
# model and the equivalent single channel input model for a number
# of ResNet models

from torchvision import models

from to_grayscale_model import resnet_grayscale_model

resnets = [models.resnet18,
          models.resnet34,
          models.resnet50,
          models.resnet101,
          models.resnet152]

for resnet in resnets:
    model = resnet()
    count_3channel = sum([p.numel() for p in model.parameters()])
    resnet_grayscale_model(model)
    count_1channel = sum([p.numel() for p in model.parameters()])
    print(resnet.__name__,': 3 channel: ', count_3channel, ' 1 channel ', count_1channel,
          ' savings ', (1.0 - count_1channel / count_3channel) * 100.0, '%')


