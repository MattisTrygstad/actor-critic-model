class UniversalAction():
    """
    Superclass for representing an action any game. This contributes to decoupeling of environment and agent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action = ()

    def __str__(self) -> str:
        return str(self.action)
