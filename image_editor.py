__author__ = 'aynroot'


class ImageEditor(object):

    def __init__(self):
        self.np_img = None

    def update_image(self, np_img):
        self.np_img = np_img


    # TODO: call c functions here
    def erosion(self):
        raise NotImplementedError

    def dilatation(self):
        raise NotImplementedError

    def inversion(self):
        raise NotImplementedError
