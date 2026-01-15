class ConfigDict:
    """A simple replacement for ml_collections.ConfigDict.
    Provides a dictionary-like object that allows attribute access to its keys.
    """
    def __init__(self, **kwargs):
        # Initialize with keyword arguments
        for key, value in kwargs.items():
            # If value is a dict, recursively convert to ConfigDict
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(**value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        # Allow dictionary-style access: config[key]
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        # Allow dictionary-style assignment: config[key] = value
        if isinstance(value, dict):
            setattr(self, key, ConfigDict(**value))
        else:
            setattr(self, key, value)
    
    def __contains__(self, key):
        # Check if key exists: key in config
        return hasattr(self, key)
    
    def get(self, key, default=None):
        # Get value for key if exists, else return default
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def to_dict(self):
        """Convert ConfigDict to a regular Python dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        # String representation of the ConfigDict
        return f"ConfigDict({self.to_dict()})"