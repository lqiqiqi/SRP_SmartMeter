class BaseTrain:
    def __init__(self, Net, config, data, logger):
        self.model = Net
        self.config = config
        self.data = data
        self.logger = logger

