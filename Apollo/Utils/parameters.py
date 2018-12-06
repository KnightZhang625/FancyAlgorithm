import argparse
from argparse import RawTextHelpFormatter

class Parameter(object):
    def __init__(self, model):
        self.parser = argparse.ArgumentParser(description=model, formatter_class=RawTextHelpFormatter)