import os
from glob import glob
import json

if __name__ == '__main__':
    data_dir = '.'

    def bar(phase='trainA'):
        trainX_list = glob(os.path.join(data_dir, f'{phase}_npy', '*'))
        trainY_list = glob(os.path.join(data_dir, f'{phase}_gt_npy', '*'))

        return [{'image': x, 'label': y} for x, y in zip(trainX_list, trainY_list)]

    trainA_list = bar(phase='trainA')
    trainB_list = bar(phase='trainB')

    valA_list = bar(phase='valA')
    valB_list = bar(phase='valB')

    json_to_save = {}

    json_to_save['trainA'] = trainA_list
    json_to_save['trainB'] = trainB_list
    json_to_save['valA'] = valA_list
    json_to_save['valB'] = valB_list

    out_dir = '.'

    with open(os.path.join(out_dir, 'trainval_list_0.json'), mode='w') as f:
        json.dump(json_to_save, f)
