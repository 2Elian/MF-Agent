#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/22 22:14
# @Author  : lizimo@nuist.edu.cn
# @File    :run.py
# @Description: 主程序入口: 用于docker部署 一键式服务 --> 包括索引构建、模型服务注册、cache加载、检索器服务注册、根据config中data_name加载初赛/复赛数据集 --> 最终输出结果json
# config中配置一个run_type --> 值为dev_single/dev_batch/prod_single/prod_batch