import os

import model
import preprocessing

gindexs = ['rsgrove', 'kdtree', 'str', 'zcurve', 'grid', 'rrstreebb']
jar_file = 'beast-uber-spark-0.2.2-SNAPSHOT.jar'
spark_master = 'spark://ec-hn.cs.ucr.edu:7077'

dataset_path = ''
output_path = ''
histogram_file = ''
hist_rows = 50
hist_cols = 50
num_layers = 3
num_units = 10


def main():
    print ('Using deep learning for Big Spatial Data partitioning')

    print ('Step 1: Computing dataset histogram...')
    os.system(
        "spark-submit --master {} {} histogram {} {} 'iformat:point(1,2)' separator:, shape:{},{} -overwrite".format(
            spark_master, jar_file, dataset_path, histogram_file, hist_rows, hist_cols))

    print ('Step 2: Extracting histogram vector...')
    histogram_vector = preprocessing.extract_and_flatten(histogram_file)

    print ('Step 3: Predicting the best partitioning technique...')
    best_index = model.predict_using_saved_model('fc', num_layers, num_units, histogram_vector, hist_rows, hist_cols)

    print ('Step 4: Partition the input dataset using the suggested partitioning technique...')
    os.system("spark-submit --master {} {} index {} {} gindex:{} 'iformat:point(1,2)' separator:, -overwrite".format(
        spark_master, jar_file, dataset_path, output_path, best_index))


if __name__ == '__main__':
    main()
