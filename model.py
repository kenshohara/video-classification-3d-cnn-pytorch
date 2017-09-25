import torch
from torch import nn

from models import resnet


def generate_model(opt):
    assert opt.model_name in ['resnet']

    if opt.model_name == 'resnet':
        assert opt.mode in ['score', 'feature']
        if opt.mode == 'score':
            last_fc = True
        elif opt.mode == 'feature':
            last_fc = False

        assert opt.model_depth in [18, 34, 50, 101]

        if opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, last_fc=last_fc)
        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, last_fc=last_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
