class BaseTrain:
    def __init__(self, Net, config, data_train, logger, data_test=None):
        self.model = Net
        self.config = config
        self.data_train = data_train
        self.logger = logger
        self.data_test = data_test

