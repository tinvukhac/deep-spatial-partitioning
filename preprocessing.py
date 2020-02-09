import metadata_utils

gindexs = ['rsgrove', 'kdtree', 'str', 'zcurve', 'grid', 'rrstreebb']


def extract_and_flatten(filename):
    hist_f = open(filename)
    hist_vector = []
    lines = hist_f.readlines()[1:]
    for line in lines:
        values = line.split('\t')
        hist_vector.append(int(values[3]))

    return hist_vector


def get_best_partitioning_scheme(dirname):
    areas = []
    margins = []
    overlaps = []
    stds = []
    disk_utils = []

    for gindex in gindexs:
        master_file = dirname + '/_master.' + gindex
        partitions = metadata_utils.get_partitions(master_file, metadata_utils.BLOCK_SIZE)
        areas.append(metadata_utils.get_total_area(partitions))
        margins.append(metadata_utils.get_total_margin(partitions))
        overlaps.append(metadata_utils.get_total_overlap(partitions))
        stds.append(metadata_utils.get_size_std(partitions))
        disk_utils.append(metadata_utils.get_disk_util(partitions))

    return areas.index(min(areas)), margins.index(min(margins)), overlaps.index(min(overlaps)), stds.index(
        min(stds)), disk_utils.index(max(disk_utils))


def main():
    hist_vector = extract_and_flatten('data/raw/histogram/points_dataset.hist')
    print ('Features X = ')
    print (hist_vector)

    best_area, best_margin, best_overlap, best_std, best_disk_util = get_best_partitioning_scheme(
        'data/raw/metadata/points_dataset_masters')

    print ('Labels ys = ')
    print ('{},{},{},{},{}'.format(best_area, best_margin, best_overlap, best_std, best_disk_util))
    print ('Or:')
    print ('{},{},{},{},{}'.format(gindexs[best_area], gindexs[best_margin], gindexs[best_overlap], gindexs[best_std],
                                   gindexs[best_disk_util]))


if __name__ == '__main__':
    main()
