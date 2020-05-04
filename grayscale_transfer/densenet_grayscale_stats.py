# COMP.5300 Deep Learning
# Terry Griffin
#
# Compare the number of parameters between a three channel input
# model and the equivalent single channel input model for a number
# of DenseNet models

from torchvision import models

from to_grayscale_model import densenet_grayscale_model

nets = [models.densenet121,
           models.densenet161,
           models.densenet169,
           models.densenet201]

for net in nets:
    model = net()
    count_3channel = sum([p.numel() for p in model.parameters()])
    densenet_grayscale_model(model)
    count_1channel = sum([p.numel() for p in model.parameters()])
    print(net.__name__,': 3 channel: ', count_3channel, ' 1 channel ', count_1channel,
          ' savings ', (1.0 - count_1channel / count_3channel) * 100.0, '%')


