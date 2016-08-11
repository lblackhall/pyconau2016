class MovingAverageFilter(object):

    def __init__(self, window):
        self.window = window
        self.data = []

    def step(self, measurement):
        self.data.append(measurement)
        if len(self.data) > self.window:
            self.data.pop(0)

    def current_state(self):
        return sum(self.data) / len(self.data)



