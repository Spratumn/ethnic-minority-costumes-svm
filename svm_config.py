CELL_NUMS = (8, 12)    # LBP划分区域数量（WN, HN）
BIN_SIZE = 256         # 直方图分箱数量
LBP_LIB_FLAG = 1      # 是否使用sklearn库提供的接口进行LBP提取，1：使用，0：使用自己实现的方法(结果相同，速度相对慢一些)
NORMAL = True         # 是否对特征进行归一化处理
PCA_RATE = 640         # PCA降维特征参数， >0 执行PCA， <=0 不执行PCA
PCA_LIB_FLAG = 1      # 是否使用sklearn库提供的接口进行PCA，1：使用，0：使用自己实现的方法(结果相同，速度相对慢一些)