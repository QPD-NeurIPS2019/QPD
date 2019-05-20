# -*- coding: utf-8 -*-
import multiprocessing
import os
import time
from datetime import datetime


def subprocess(seed, map_name, buffer_size, a_lr, c_lr, clip_method, clip_value, channel_merge, multi_channel, input_length,
               a_l2, c_l2, battle_num, explore_num, dense_unit, end_epsilon):
    os.system('python main.py -s {} -mn {} -rb {} -a_lr {} -c_lr {} -cm {} -cv {}'
              ' -chm {} -mc {} -a_l2 {} -c_l2 {} -bn {} -en {} -du {} -ee {} -il {} -wn 2 > out{}_{}_{}_{}.log 2>&1 &'
              .format(seed, map_name, buffer_size, a_lr, c_lr, clip_method, clip_value,
                      channel_merge, multi_channel, a_l2, c_l2, battle_num, explore_num, dense_unit, end_epsilon, input_length,
                      seed, explore_num, battle_num, end_epsilon))


def mainprocess():
    pool = multiprocessing.Pool(3)  # run 3 process at the same time
    # for i in range(3):
    #     # pool.apply_async(subprocess, args=(i, 'no', 0, ))
    #     # seed, map_name, buffer_size, actor_lr, critic_lr, clip_method, clip_value, channel_merge, multi_channel,
    #     # a_l2, c_l2, battle_num, explore_num
    #     pool.apply_async(subprocess, args=(i, '3m', 1000, 1E-3, 1E-3, 'global_norm', 5, 'concat', 'on',
    #                                        0.01, 0.01, 5000, 1000, ))

    # for lr in [1E-2, 5E-3, 1E-3, 5E-4, 3E-4, 5E-5, 1E-5, 1E-6]:
    #    for clip_value in [0.1, 0.25, 0.5, 1, 2, 5, 10]:
    # for epsilon in [0, 0.05]:
    #     for bn in [15000]:
    #         for en in [2000]:
    #             for i in range(3):
    #                 # pool.apply_async(subprocess, args=(i, 'no', 0, ))
    #                 # seed, map_name, buffer_size, actor_lr, critic_lr, clip_method, clip_value, channel_merge, multi_channel,
    #                 # a_l2, c_l2, battle_num, explore_num
    #                 pool.apply_async(subprocess, args=(i, '3s5z', 1000, 5E-4, 5E-4, 'global_norm', 5, 'concat', 'on',
    #                                                    0.0, 0.0, bn, en, 64, epsilon, ))
    #                 time.sleep(3)
    for epsilon in [0]:
        for bn in [50000]:
            for en in [2000]:
                for i in range(0, 1):
                    # pool.apply_async(subprocess, args=(i, 'no', 0, ))
                    # seed, map_name, buffer_size, actor_lr, critic_lr, clip_method, clip_value, channel_merge, multi_channel, input_length,
                    # a_l2, c_l2, battle_num, explore_num
                    pool.apply_async(subprocess, args=(i, '3s5z_vs_3s6z', 1000, 5E-4, 5E-4, 'global_norm', 5, 'concat', 'on', 16,
                                                       0.0, 0.0, bn, en, 64, epsilon, ))
                    time.sleep(3)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # main function
    mainprocess()
