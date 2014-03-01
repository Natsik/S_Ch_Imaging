__author__ = 'aynroot'

from ms2_c_func import MS2


def c_call(function):
    def c_call_wrapper(self):
        self.c_img = self.to_c_format()
        function(self)
        self.np_img = self.from_c_format()
        return self.np_img
    return c_call_wrapper


class ImageEditor(object):

    def __init__(self):
        self.np_img = None
        self.c_img = None
        self.np_shape = None

    def update_image(self, np_img):
        self.np_img = np_img
        self.np_shape = self.np_img.shape

    def to_c_format(self):
        if self.np_shape[2] == 3:
            self.c_img = self.np_img.flatten()
        elif self.np_shape[2] == 4:
            raise NotImplementedError
        return self.c_img

    def from_c_format(self):
        if self.np_shape[2] == 3:
            self.np_img = self.c_img.reshape(self.np_shape)
        elif self.np_shape[2] == 4:
            raise NotImplementedError
        return self.np_img

    @c_call
    def test(self):
        return self.c_img

    @c_call
    def erosion(self):
        new_c_img = self.c_img.copy()
        MS2.c_errosion_func(self.c_img, new_c_img)
        self.c_img = new_c_img

    @c_call
    def dilatation(self):
        new_c_img = self.c_img.copy()
        MS2.c_dilatation_func(self.c_img, new_c_img)
        self.c_img = new_c_img

    @c_call
    def inversion(self):
        new_c_img = self.c_img.copy()
        MS2.c_inversion_func(self.c_img, new_c_img)
        self.c_img = new_c_img
