CONE_CLASSES = ('seg_orange_cone', 'seg_yellow_cone', 'seg_large_orange_cone',
                'seg_blue_cone', 'seg_unknown_cone')

class ConeTrainingConfig:
    def __init__(self):
        self.classNames = CONE_CLASSES
        self.numClasses = len(self.classNames) + 1
        self.continuousId = {(aa + 1): (aa + 1) for aa in range(self.numClasses - 1)}

        self.trainImgs = 'Data/fsoco_segmentation_train/'
        self.trainAnn = 'Data/fsoco_segmentation_train/train_coco.json'
        self.valImgs = 'Data/fsoco_segmentation_train/'
        self.valAnn = 'Data/fsoco_segmentation_train/train_coco.json'

        self.imgSize = 544
        self.trainBs = 8
        self.bsPerGpu = 8

        self.warmupUntil = 500
        self.lrSteps = (0, 40000, 60000, 70000, 80000)

        self.lr = 0.001
        self.warmupInit = self.lr * 0.1

        self.posIouThre = 0.5
        self.negIouThre = 0.4

        self.confAlpha = 1
        self.bboxAlpha = 1.5
        self.maskAlpha = 6.125
        self.semanticAlpha = 1

        self.masksToTrain = 100

        self.backboneWeight = 'Yolact_minimal/weights/backbone_res50.pth'

        self.valInterval = 4000
        self.valNum = -1

        self.scales = [int(self.imgSize / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspectRatios = [1, 1 / 2, 2]
