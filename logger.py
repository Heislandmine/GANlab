"""
学習過程・生成物の格納
"""

import os
import os.path
import time


class Logger:
    def __init__(self):
        self.train_name = self._get_time_stamp()

    def _get_time_stamp(self) -> str:
        fmt = "%Y_%m_%d_%H_%M_%S"
        time_stamp = time.strftime(fmt, time.localtime())
        return time_stamp
