
class GridNode:
    def __init__(self, is_empty: bool, is_agent: bool) -> None:
        super().__init__()

        self.is_empty = is_empty
        self.is_agent = is_agent
