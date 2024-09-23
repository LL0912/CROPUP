# -*- ecoding: utf-8 -*-
# @enviroment: pytorch 1.6.0 CUDA 9.0
# @Author: LeiLei leilei912@whu.edu.cn

def format_timedelta(start_time,end_time):
    duration = end_time - start_time
    total_seconds = int(duration)
    hours, remainder = divmod(total_seconds, 60*60)
    minutes, seconds = divmod(remainder, 60)
    print("time consuming:",duration)
    print("{}:{}:{}".format(hours, minutes, seconds))

