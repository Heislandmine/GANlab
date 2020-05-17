"""
学習過程・生成物の格納
"""

import os
import os.path
import time


class Logger:
    def __init__(self):
        self.train_name = self._get_time_stamp()

        self.base_dir_name = "results/" + self.train_name
        self.log_dir_name = self.base_dir_name + "/log"
        self.images_dir_name = self.base_dir_name + "/images"
        self.weights_dir_name = self.base_dir_name + "/weights"
        self.train_conf_dir_name = self.base_dir_name + "/train_conf"

        self.log_dirs = [
            self.base_dir_name,
            self.log_dir_name,
            self.images_dir_name,
            self.weights_dir_name,
            self.train_conf_dir_name,
        ]

        self._make_log_dir(self.log_dirs)

    def _get_time_stamp(self) -> str:
        fmt = "%Y_%m_%d_%H_%M_%S"
        time_stamp = time.strftime(fmt, time.localtime())
        return time_stamp

    def _make_log_dir(self, log_dirs: list) -> None:
        for dir_name in log_dirs:
            if not self._check_dir_exsist(dir_name):
                os.makedirs(dir_name)

    def _check_dir_exsist(self, dir_name: str) -> bool:
        return os.path.exists(dir_name)
