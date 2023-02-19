class Crop:

    @staticmethod
    def get_dim_max(img, dim, i):
        """
            获取指定维度指定位置的最大值
        :param img: 三维numpy数组
        :param dim: 指定维度
        :param i: 指定的位置
        :return: 指定维度指定位置的最大值
        """
        if dim == 0:
            return img[i, :, :].max()
        elif dim == 1:
            return img[:, i, :].max()
        elif dim == 2:
            return img[:, :, i].max()
        else:
            raise Exception('Dimensions cannot be greater than 3!')

    @staticmethod
    def dim_crop(img, dim, label):
        """
            用于裁切单一维度的数据
        :param img: 三维numpy数组
        :param dim: 第几个维度
        :param label: 裁切标志
        :return:
        """
        min_dim = -1
        max_dim = -1
        for i in range(img.shape[dim]):
            if Crop.get_dim_max(img, dim, i) != label:
                min_dim = i
                break

        for i in range(img.shape[dim] - 1, -1, -1):
            if Crop.get_dim_max(img, dim, i) != label:
                max_dim = i
                break

        # 校正
        min_dim = 0 if min_dim == -1 else min_dim
        max_dim = img.shape[dim] - 1 if max_dim == -1 else max_dim

        len_dim = max_dim - min_dim
        return min_dim, len_dim

    @staticmethod
    def crop(img, label=0, square=False, expand=0):
        """
            crop
        :param img: img，三维的numpy数组
        :param label: 裁切的标志
        :param square: 是否需要裁剪成正方形
        :param expand: 扩张的数量
        :return:
        """
        # 按slice找
        min_z, len_z = Crop.dim_crop(img, 0, label)
        # 按x轴找
        min_x, len_x = Crop.dim_crop(img, 1, label)
        # 按y轴找
        min_y, len_y = Crop.dim_crop(img, 2, label)

        if square:
            if len_x > len_y:
                return min_z, len_z, min_x, len_x, min_y, len_x
            else:
                return min_z, len_z, min_x, len_y, min_y, len_y
        else:
            return min_z, len_z, min_x, len_x, min_y, len_y
