from fpn import *
from resnet import *
from head import *
import torch

class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig

        self.backbone = resnet50(pretrained=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = SharedHead(config.fpn_out_channels, config.class_num, config.use_GN_head, config.cnt_on_reg, config.prior)

        self.config = config

    def train(self, mode=True):
        """
        set training mode, frozen bn
        """
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in module.parameters():
                    p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")

        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        Ps = self.fpn([C3, C4, C5])

        cls, cnt, reg = self.head(Ps)

        return [cls, cnt, reg]


class DetectHead(nn.Module)::
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()

        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides

        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, x):
        cls, cnt, reg = x



