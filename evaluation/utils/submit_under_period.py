import sys
import time
import datetime
from datetime import timedelta
import os 
import argparse

def is_discount_time():
    # 获取当前时间（UTC时间）
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    # 将UTC时间转换为北京时间（UTC+8）
    beijing_time = now_utc + timedelta(hours=8)
    
    # 定义优惠时间段（北京时间00:30-8:30）
    discount_start = beijing_time.replace(hour=0, minute=30, second=0, microsecond=0)
    discount_end = beijing_time.replace(hour=8, minute=30, second=0, microsecond=0)
    print(f"当前时间：{beijing_time}", flush=True)
    
    # 如果当前时间在优惠时间段内
    if discount_start <= beijing_time < discount_end:
        return True
    else:
        return False



def submit_program(program):
    # 这里是你提交程序的代码
    print("提交程序中...", flush=True)
    # 如果超过北京时间8.30，就退出程序
    # 使用装饰器，如果超过8.30，就退出程序
    
    os.system(program)
    # 例如：os.system("your_program_command")

def sleep_until_next_discount():
    # 获取当前时间（UTC时间）
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    # 将UTC时间转换为北京时间（UTC+8）
    beijing_time = now_utc + timedelta(hours=8)
    
    # 计算下一个优惠时间段的开始时间
    if beijing_time.hour < 0 or (beijing_time.hour == 0 and beijing_time.minute < 30):
        # 如果当前时间在00:00-00:30之间，下一个优惠时间段是今天的00:30
        next_discount_start = beijing_time.replace(hour=0, minute=30, second=0, microsecond=0)
    else:
        # 否则，下一个优惠时间段是明天的00:30
        next_discount_start = (beijing_time + timedelta(days=1)).replace(hour=0, minute=30, second=0, microsecond=0)
    
    # 计算需要睡眠的时间（秒）
    sleep_duration = (next_discount_start - beijing_time).total_seconds()
    
    # 睡眠直到下一个优惠时间段开始
    print(f"当前时间不在优惠时间段内，睡眠 {sleep_duration} 秒...", flush=True)
    time.sleep(sleep_duration)

def main(program):
    while True:
        if is_discount_time():
            
            submit_program(program)
            break  # 提交程序后退出循环
        else:
            sleep_until_next_discount()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_cmd", type=str, required=True)
    parser.add_argument("--is_bonus",  default=False, action="store_true")
    
    args = parser.parse_args()
    if args.is_bonus:
        main(args.program_cmd)
    else:
        submit_program(args.program_cmd)