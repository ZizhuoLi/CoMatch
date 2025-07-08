from .resnet import ResNet_8_1_align


def build_backbone(config):
    if config['backbone_type'] == 'ResNet':
        if config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return ResNet_8_1_align(config['backbone'])
        else:
            raise ValueError(f"LOFTR.ALIGN_CORNER {config['align_corner']} not supported.")

    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
