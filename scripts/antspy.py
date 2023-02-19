import os
import ants
import numpy as np
import time

# fix_path = 'img_fix.png'
# move_path = 'img_move.png'

import cv2

for idx, img_name in enumerate(os.listdir('/mnt/z/CodingHere/GAN/DCLGAN-main/datasets/ct2mr/niiA')):

    # img = cv2.imread(f'/mnt/z/CodingHere/GAN/DCLGAN-main/datasets/ct2mr/paired/{img_name}', 0)
    #
    # fixed_image = img[:, :512]
    # moving_image = img[:, 512:]
    #
    # cv2.imwrite('a.png', fixed_image)
    # cv2.imwrite('b.png', moving_image)

    fix_path = f'./niiA/{img_name}'
    move_path = f'./niiB/{img_name}'

    types = ['Translation', 'Rigid', 'Similarity', 'QuickRigid', 'DenseRigid', 'BOLDRigid', 'Affine', 'AffineFast',
             'BOLDAffine',
             'TRSAA', 'ElasticSyN', 'SyN', 'SyNRA', 'SyNOnly', 'SyNCC', 'SyNabp', 'SyNBold', 'SyNBoldAff', 'SyNAggro',
             'TVMSQ']

    # 保存为png只支持unsigned char & unsigned short，因此读取时先进行数据类型转换
    fix_img = ants.image_read(fix_path, pixeltype='unsigned char')
    move_img = ants.image_read(move_path, pixeltype='unsigned char')

    for t in types:
        start = time.time()

        out = ants.registration(fix_img, move_img, type_of_transform=t)
        print(out)
        reg_img = out['warpedmovout']  # 获取配准结果

        out_name = img_name.replace('.nii', f'_{t}.nii')
        reg_img.to_file('niiB-reg/' + out_name)

        print(t + ' : ', time.time() - start, '\n')
