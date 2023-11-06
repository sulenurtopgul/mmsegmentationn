# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RoadSegDataset(BaseSegDataset):
    """RoadSeg dataset.
    """
    METAINFO = dict(
        classes=('Ausgebaute Allwetterstrasse',
                 'Eisenbahn',
                 'Fußweg',
                 'Karawanenweg',
                 'Saumweg'),
        palette=[[77, 255, 0], 
                 [204, 0, 0],
                 [230, 128, 0],
                 [255, 0, 0],
                 [0, 204, 242]])
    
    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
