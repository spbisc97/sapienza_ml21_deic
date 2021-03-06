from .common import ImageClassificationDataset


class ISIC2018Dataset(ImageClassificationDataset):
    """ ISIC 2018 Skin Lesion Classification dataset.

    Dataset: https://challenge.isic-archive.com/landing/2018/47
    Paper: https://arxiv.org/abs/1902.03368

    See `ImageClassificationDataset` for documentation.
    """

    CLASSNAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def __init__(self, root, split, img_dir='ISIC2018_Task3_Training_Input',
                 transform=None, target_transform=None):

        super(ISIC2018Dataset, self).__init__(
            root=root,
            split=split,
            img_dir='ISIC2018_Task3_Training_Input' if img_dir is None else img_dir,
            transform=transform,
            target_transform=target_transform
        )


    @staticmethod
    def get_normalization_statistics():

        return [0.7778883,  0.53917843, 0.5600745], [0.13092259, 0.15768562, 0.17565714]
