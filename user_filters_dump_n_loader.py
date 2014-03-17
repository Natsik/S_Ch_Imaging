__author__ = 'aynroot'

import os
import cPickle
FILTERS = 'filters'
GOLDEN_IMAGES = 'golden_images'
DIFF_IMAGES = 'diff_images'


def _maybe_make_dirs(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


class UserSettingsDumpNLoader(object):
    """ Class for saving and restoring user settings (for ex. filters) """

    def __init__(self, filename='./user_settings/settings'):
        self.settings_file_name = filename
        self.settings = None
        _maybe_make_dirs(filename)
        self.read_file()

    def read_file(self):
        try:
            with open(self.settings_file_name) as f:
                self.settings = cPickle.load(f)
        except:
            self.settings = {}

    def save_filter(self, matrix, divisor, name):
        if FILTERS not in self.settings:
            self.settings[FILTERS] = {}
        self.settings[FILTERS][name] = (matrix, divisor)

    def _save_settings(self):
        with open(self.settings_file_name, 'w') as f:
            cPickle.dump(self.settings, f)

    def get_filters(self):
        return self.settings.get(FILTERS, {})

    def get_golden_images_dir(self):
        return self._get_dirname(GOLDEN_IMAGES, './golden images')

    def get_diff_images_dir(self):
        return self._get_dirname(DIFF_IMAGES, './diffs')

    def _get_dirname(self, dir_type, default_value):
        dirname = self.settings.get(dir_type, default_value)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def _set_dirname(self, dir_type, dirname):
        self.settings[dir_type] = dirname
        self._save_settings()

    def set_golden_images_dir(self, dirname):
        self._set_dirname(GOLDEN_IMAGES, dirname)

    def set_diff_images_dir(self, dirname):
        self._set_dirname(DIFF_IMAGES, dirname)
