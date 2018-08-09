import argparse


class Flag(object):
    def __init__(self):
        self.learning_rate = 0.001
        self.seq_len = 20
        self.batch_size = 16
