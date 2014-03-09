__author__ = 'aynroot'


class History(object):
    """
    Stores history of modifications for current opened image.
    Allows revert or repeat image changes.
    """

    # TODO: decide what to do when branching (probably remove all future states and change them)

    def __init__(self):
        self.states = []
        self.current_state = -1

    def add_new_state(self, np_img):
        self.states.append(np_img)
        self.current_state += 1

    def undo(self):
        self.current_state -= 1
        return self.states[self.current_state]

    def redo(self):
        self.current_state += 1
        return self.states[self.current_state]

    def can_redo(self):
        return self.current_state < len(self.states) - 1

    def can_undo(self):
        return self.current_state > 0

    def reset(self):
        self.__init__()