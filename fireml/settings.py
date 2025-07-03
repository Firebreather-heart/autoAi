import os, pathlib,json


BASE_DIR = os.path.join(pathlib.Path(__file__).resolve().parent, "settings.json")


class Settings:
    output_directory:str
    log_directory:str
    working_directory:str

    def __init__(self):
        with open(BASE_DIR, 'r') as settings:
            settings_dict = json.load(settings)
        self.BASE_DIR = BASE_DIR 
        
        for key, value in settings_dict.items():
            setattr(self, key, value)

    def display(self):
        print(self.__dict__)

