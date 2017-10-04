import yaml


class Config:

    def __init__(self):
        self._values = {}

    def __getattr__(self, name):
        return self._values[name]

    def __setattr__(self, name, value):
        if name == '_values':
            return super().__setattr__(name, value)
        self._values[name] = value

    def load_config_dict(self, dct):
        '''Load a dictionary of configuration values.'''
        self._values.update(dct)

    def load_config_file(self, filename):
        '''Load config from a YAML file.'''
        with open(filename) as f:
            self._values.update(yaml.load(f))
