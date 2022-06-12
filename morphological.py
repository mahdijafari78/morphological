import numpy as np
from PIL import Image


class PhotoShop:

    def __init__(self, filepath):
        self.filepath = filepath
        self.image_src = self.binarize_this(image_file=self.filepath)

    def image2pixelarray(self, filepath):
        im = Image.open(filepath).convert('L')
        (width, height) = im.size
        greyscale_map = list(im.getdata())
        greyscale_map = np.array(greyscale_map)
        greyscale_map = greyscale_map.reshape((height, width))
        return greyscale_map

    def convert_binary(self, image_src, thresh_val):
        color_1 = 255
        color_2 = 0
        initial_conv = np.where((image_src <= thresh_val), image_src, color_1)
        final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
        return final_conv

    def binarize_this(self, image_file, thresh_val=127):
        image_src = self.image2pixelarray(image_file)
        image_b = self.convert_binary(image_src=image_src, thresh_val=thresh_val)
        return image_b

    def erode(self, erosion_level=3):
        erosion_level = 3 if erosion_level < 3 else erosion_level

        structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)


        orig_shape = self.image_src.shape
        image_pad = np.pad(array=self.image_src, pad_width=erosion_level - 2, mode='constant')
        flat_submatrices = np.array([
            image_pad[i:(i + erosion_level), j:(j + erosion_level)]
            for i in range(orig_shape[0]) for j in range(orig_shape[1])
        ])
        image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])
        image_erode = image_erode.reshape(orig_shape)

        return image_erode

    def dilate(self, dilation_level=3):
        dilation_level = 3 if dilation_level < 3 else dilation_level

        structuring_kernel = np.full(shape=(dilation_level, dilation_level), fill_value=255)

        orig_shape = self.image_src.shape
        pad_width = dilation_level - 2

        image_pad = np.pad(array=self.image_src, pad_width=pad_width, mode='constant')

        flat_submatrices = np.array([
            image_pad[i:(i + dilation_level), j:(j + dilation_level)]
            for i in range(orig_shape[0]) for j in range(orig_shape[1])
        ])

        image_dilate = np.array([255 if (i == structuring_kernel).any() else 0 for i in flat_submatrices])

        image_dilate = image_dilate.reshape(orig_shape)

        return image_dilate

    def morphological_open(self,level=3):
        self.image_src = self.erode(level)
        return self.dilate(level)

    def morphological_close(self,level=3):
        self.image_src = self.dilate(level)
        return self.erode(level)

    def save(self, array, file_path):
        Image.fromarray(array.astype(np.uint8)).save(file_path)

