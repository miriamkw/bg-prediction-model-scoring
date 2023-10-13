import json
import pkg_resources


class ConfigManager:
    def __init__(self):
        self.config_file = pkg_resources.resource_filename('glupredkit', 'config.json')
        with open(self.config_file) as f:
            self.config = json.load(f)

    @property
    def use_mgdl(self):
        return self.config.get('use_mgdl', True)

    @use_mgdl.setter
    def use_mgdl(self, value):
        self.config['use_mgdl'] = value
        # Update the config.json file to persist the change
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def convert_value(self, value):
        if not self.use_mgdl:
            return value / 18.018
        return value


# You can create a global instance of the config manager.
config_manager = ConfigManager()
