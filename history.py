__author__ = 'aynroot'


class History(object):
    """
    Stores history of modifications for current opened image.
    Allows revert or repeat image changes.
    """

    def __init__(self):
        self.states_for_undo = []
        self.states_for_redo = []

    def add_new_state(self, np_img):
        self.states_for_undo.append(np_img)
        self.states_for_redo = []

    def undo(self):
        state = self.states_for_undo.pop()
        self.states_for_redo.append(state)
        return state

    def redo(self):
        state = self.states_for_redo.pop()
        self.states_for_undo.append(state)
        return state

    def can_redo(self):
        return True if self.states_for_redo else False

    def can_undo(self):
        return True if len(self.states_for_undo) > 1 else False

    def reset(self):
        self.__init__()