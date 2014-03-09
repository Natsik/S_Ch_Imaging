__author__ = 'aynroot'

import os
import cPickle


def _maybe_make_dirs(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


class UserFiltersDumpNLoader(object):
    """ Class for saving and restoring user settings (for ex. filters) """

    def __init__(self, filename='./user_settings/settings'):
        self.settings_file_name = filename
        self.user_filters = None
        _maybe_make_dirs(filename)
        self.read_file()

    def read_file(self):
        try:
            with open(self.settings_file_name) as f:
                self.user_filters = cPickle.load(f)
        except:
            self.user_filters = {}

    def save_filter(self, matrix, divisor, name):
        self.user_filters[name] = (matrix, divisor)
        with open(self.settings_file_name, 'w') as f:
            cPickle.dump(self.user_filters, f)

