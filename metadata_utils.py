import math
from shapely.geometry.polygon import Polygon
import numpy as np


BLOCK_SIZE = 4


class Partition:
    def __init__(self):
        pass

    partitionId = 0
    numRecords = 0
    filesize = 0
    filename = ""
    x1, y1, x2, y2 = 0, 0, 0, 0
    mbr = Polygon()
    nblocks = 1


def get_partitions(filename, block_size):

    partitions = []
    with open(filename) as f:
        lines = f.readlines()
        lines.pop(0)
        for line in lines:
            values = line.split('\t')
            partition = Partition()
            try:
                # partition.partitionId = int(values[0])
                partition.numRecords = int(values[2])
                partition.filesize = int(values[3])
                partition.filename = values[1].replace("'", "")
                partition.x1, partition.y1, partition.x2, partition.y2 = float(values[5]), float(values[6]), float(values[7]), float(values[8])
                partition.nblocks = math.ceil(float(values[3]) / (block_size * 1024 * 1024))
                partition.mbr = Polygon([(partition.x1, partition.y1), (partition.x1, partition.y2), (partition.x2, partition.y2), (partition.x2, partition.y1)])
            except Exception as e:
                print(filename)
                print(e)
            partitions.append(partition)
    return partitions


def get_total_area(partitions):
    total_area = 0
    for p in partitions:
        total_area += abs(p.x2 - p.x1) * abs(p.y2 - p.y1) * p.nblocks
    return total_area


def get_total_margin(partitions):
    total_margin = 0
    for p in partitions:
        total_margin += abs(p.x2 - p.x1) + abs(p.y2 - p.y1) * p.nblocks
    return total_margin


def get_total_overlap(partitions):
    total_overlap = 0
    for p1 in partitions:
        total_overlap += p1.mbr.area * p1.nblocks * (p1.nblocks - 1) / 2
        for p2 in partitions:
            if p1 is not p2 and p1.mbr.intersects(p2.mbr):
                total_overlap += p1.mbr.intersection(p2.mbr).area * p1.nblocks * p2.nblocks

    return total_overlap


def get_size_std(partitions):
    sizes = []
    for p in partitions:
        sizes.append(p.filesize/1024/1024)
    size_arr = np.array(sizes)
    size_std = np.std(size_arr)
    return size_std


def get_disk_util(partitions):
    value = 0
    count = 0
    for p in partitions:
        p_util = p.filesize / (float) (128 * 1024 * 1024)
        value += p_util
        count += p.nblocks
    util = value / count
    return util


def get_cost(partitions):
    cost = 0
    ratio = pow(10, -4)
    w = 180 + 180
    h = 83 + 90
    q = math.sqrt(ratio * w * h)
    for p in partitions:
        cost += (abs(p.x2 - p.x1) + q) * (abs(p.y2 - p.y1) + q) * p.nblocks / (w * h)
    return cost
