class Buffer:
    'Fixed sized buffer for collecting model data, for Eplus building model generation'
    def __init__(self,
                 buffer_size: int = 5000000000,
                 data_collection_period_start: Tuple[int, int] = [1,1],
                 data_collection_period_end: Tuple[int, int] = [12,31],
                 data_collection_method: str = 'random',
                 weather_region: str = 'rochester'):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.weather_region = weather_region
        self.data_collection_period_start = data_collection_period_start
        self.data_collection_period_end = data_collection_period_end
        self.episode = 0
        # prob 'Random'
        self.data_collection_method = data_collection_method
        #self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])

    def add(self, elem):
        #e = tuple(state)
        e = elem
        self.buffer.append(elem)

    def b_full(self):
        if len(self.buffer) == self.buffer_size:
            return True
        else:
            return False

    def percentage_full(self):
        return round(len(self.buffer) / self.buffer_size, 2)
