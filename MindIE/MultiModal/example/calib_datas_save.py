#!/usr/bin/env python
# coding=utf-8
import inspect
import os
from functools import wraps

import torch
from torch import distributed as dist


global global_args_list


def save_args_decorator():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            calib_datas_path = os.getenv("CALIB_DATAS_PATH")
            if calib_datas_path is None:
                return func(self, *args, **kwargs)
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            params = list(bound_args.arguments.values())[1:]  # 去掉self

            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0  # 单进程环境

            try:
                # 尝试访问全局变量
                global_args_list
            except NameError:
                # 如果未定义则初始化
                global_args_list = []
            global_args_list.append(params)

            if rank == 0:
                # 保存到pt文件
                torch.save(global_args_list, os.path.join(calib_datas_path, "calib_datas.pt"))

            return func(self, *args, **kwargs)

        return wrapper

    return decorator
