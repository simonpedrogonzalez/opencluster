class CatalogNotFoundException(Exception):
    def __init__(self, message="No known catalog could be found"):
        self.message = message
        super().__init__(self.message)
