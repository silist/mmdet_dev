from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CityPersonsDataset(CocoDataset):

    CLASSES = ('pedestrian', )
