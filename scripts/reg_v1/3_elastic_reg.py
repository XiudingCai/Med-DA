"""

"""

import os
import pandas as pd
import SimpleITK as sitk
from shutil import copy
import ants
import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='base path')
    parser.add_argument('-o', type=str, required=True, help='save path')
    parser.add_argument('-n', type=str, required=False, default='SyN', help='name of registration method')
    return parser.parse_args()


def ant_registration(image, evaluater, name='SyN'):
    """

    :param f_path:
    :param m_path:
    :param s_path:
    :param evaluater:
    :param name:
    :return:
    """
    split_symbol = '/'
    print('---------' * 10)
    print('processing image', image['moving'])
    # registration
    fi = ants.image_read(image['fixed'])
    mi = ants.image_read(image['moving'])
    reged = ants.registration(
        fixed=fi,
        moving=mi,
        type_of_transform=name,
    )
    # saving reged img
    s_path = os.path.join(image['save_path'], split_symbol.join(image['moving'].split(split_symbol)[-3:]))
    os.makedirs(os.path.dirname(s_path), exist_ok=True)
    ants.image_write(reged['warpedmovout'], s_path)
    print('\tregistered image', image['moving'])

    # apply transform to moving images
    for to_moving in image['to_moving']:
        moving = ants.apply_transforms(fixed=fi, moving=ants.image_read(to_moving),
                                       transformlist=reged['fwdtransforms'])
        s_path = os.path.join(image['save_path'], split_symbol.join(to_moving.split(split_symbol)[-3:]))
        os.makedirs(os.path.dirname(s_path), exist_ok=True)
        ants.image_write(moving, s_path)
        print('\ttransformed image', to_moving)

    # copy images
    for to_copy in image['to_copy']:
        s_path = os.path.join(image['save_path'], split_symbol.join(to_copy.split(split_symbol)[-3:]))
        os.makedirs(os.path.dirname(s_path), exist_ok=True)
        try:
            copy(to_copy, s_path)
        except FileExistsError as err:
            print(err)
        print('\tcopy image', to_copy)

    # print('image saved to ' % s_path)
    # metric = eval(fi, mi, reged, evaluater)
    metric = 0
    return metric


def eval(f, m, reged, evaluater):
    """

    :param f:
    :param m:
    :param reged:
    :param evaluater:
    :return:
    """
    result = {}
    for item in evaluater:
        result[item] = '%.6f_%.6f' % (evaluater[item](f, m), evaluater[item](f, reged['warpedmovout']))
    return result


if __name__ == '__main__':
    args = get_params()
    base_dir = args.i
    save_path = args.o
    method_name = args.n

    result = []
    log = []

    images = []

    # generating images paris
    for name in os.listdir(os.path.join(base_dir, 'MRI', 'bone_labels')):
        images.append({
            'fixed': os.path.join(base_dir, 'CT', 'images', name.replace('bone', 'image')),
            'moving': os.path.join(base_dir, 'MRI', 'images', name.replace('bone', 'image')),

            'to_moving': [os.path.join(base_dir, 'MRI', 'bone_labels', name),
                          os.path.join(base_dir, 'MRI', 'tumor_labels', name.replace('bone', 'tumor'))],

            'to_copy': [os.path.join(base_dir, 'CT', 'images', name.replace('bone', 'image')),
                        os.path.join(base_dir, 'CT', 'gtv_labels', name.replace('bone', 'gtv')),
                        os.path.join(base_dir, 'CT', 'bone_labels', name)],

            'save_path': save_path
        })

    columns = ['image_name', 'mi']
    evaluater = {
        'mi': ants.image_mutual_information,
    }

    for image in images:
        log.append('fixed file name %s' % image['fixed'])
        print('fixed file name %s' % image['fixed'])
        metric = ant_registration(
            image,
            evaluater,
            method_name
        )
        print('finished %s' % image['moving'])
        # result_line = [metric[item] for item in columns[1:]]
        # result_line.insert(0, '%s' % image['moving'])
        # result.append(result_line)

    # result_pd = pd.DataFrame(result, columns=columns)
    # result_pd.to_csv(os.path.join(save_path, 'metirc_value.csv'), index=False)
    # with open(os.path.join(save_path, 'log.log'), 'w', encoding='utf-8') as f:
    #     f.writelines(log)
