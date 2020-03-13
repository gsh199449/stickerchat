import csv
import select
import socket
import subprocess
import sys


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def get_available_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = str(child.communicate()[0], 'utf8')
    reader = csv.reader(text.split('\n')[:-1])
    stats = []
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        print(
            'index:{} memory:{} compute_percentage:{} memory_percentage:{}'.format(line[0], memory, compute_percentage,
                                                                                   memory_percentage))
        stats.append({'id': line[0], 'memory': memory})
        if memory < 200 and compute_percentage < 5 and memory_percentage < 5:
            print('\033[32m gpu ' + line[0] + ' is available \033[0m')
            return int(line[0])
    stats.sort(key=lambda e: e['memory'])
    print('\033[31m can\'t find an available gpu, please change another server!!!! \033[0m')
    return int(get_input_with_timeout('use which GPU?', 5, stats[0]['id']))


def get_free_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = str(child.communicate()[0], 'utf8')
    reader = csv.reader(text.split('\n')[:-1])
    stats = []
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        stats.append({'id': line[0], 'memory': memory})
        if memory < 200 and compute_percentage < 5 and memory_percentage < 5:
            return int(line[0])
    stats.sort(key=lambda e: e['memory'])
    return int(stats[0]['id'])


def get_input_with_timeout(prompt: str, time: int = 10, default_input=None):
    """

    :param prompt: input prompt
    :param time: timeout in seconds
    :param default_input: when timeout this function will return this value
    :return: input value or default_input
    """
    print(bcolors.GREENBACK + prompt + bcolors.ENDC)
    print(bcolors.GREENBACK + ' Time Limit %d seconds' % time + bcolors.ENDC)
    i, o, e = select.select([sys.stdin], [], [], time)
    if i:
        return sys.stdin.readline().strip()
    else:
        return default_input


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0;37;40m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5;41;42m'
    GREENBACK = '\033[0;40;42m'
    REDBACK = '\033[0;42;101m'


if __name__ == '__main__':
    get_available_gpu()
