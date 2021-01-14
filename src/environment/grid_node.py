
class GridNode:
    def __init__(self, is_empty: bool, is_agent: bool) -> None:
        super().__init__()

        """ self.x_value = x_value
        self.y_value = y_value """
        self.is_empty = is_empty
        self.is_agent = is_agent
