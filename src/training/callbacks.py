class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best = None
        self.count = 0

    def step(self, value):
        if self.best is None or value < self.best:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience
