from imgaug import augmenters as iaa
import imgaug as ia
import torchvision
import numpy as np
import glob
import os
from PIL import Image


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            # iaa.Fliplr(0.15),
            # iaa.Crop(px=(0, 10)),
            iaa.Sometimes(0.25, iaa.PerspectiveTransform(0.08)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.Affine(rotate=(-10, 10), mode='symmetric'),
            iaa.Sometimes(0.25,
                        iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                    iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
      
    def save_img(self, img, path):
        """
        Args:
            img: numpy array
        """
        img = Image.fromarray(img).convert("RGB")
        img.save(path)

    def Aug(self, folder, aug_number):
        data_len = len(os.listdir(folder))
        paths = list(glob.iglob(os.path.join(folder,"*.*")))
        count = 0
        while count + data_len < aug_number:
            # print(count + data_len)
            for p in paths:
                # print(count+data_len)
                if count + data_len >= aug_number:
                    break
                try:
                    img = Image.open(p).convert('RGB')
                    img = np.array(img)
                    img_name = "aug_"+str(count) +".jpg"
                    path = os.path.join(folder, img_name)
                    self.save_img(self.aug.augment_image(img), path)
                    count += 1
                except Exception as e:
                    print(e)
                


# transforms = ImgAugTransform()

# transforms.Aug('/mnt/01D3744DD971BD10/Projects/logo (copy)/data_backup_copy/3m', aug_number=30)