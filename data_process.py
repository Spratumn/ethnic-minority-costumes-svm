import os

# 将原始提供的图像数据进行同一整理：
#   1. 将图像统一存放在’images‘文件夹，并统一命名格式；
#   2. 记录图像文件的类别信息，存到annotations.csv；
#   3. 记录数据集的类别对应关系，存入readme.txt
def gather_dataset(oriDir, saveDir):
    classnames = os.listdir(oriDir)
    imageDir = os.path.join(saveDir, 'images')
    annoPath = os.path.join(saveDir, 'annotations.csv')
    readmePath = os.path.join(saveDir, 'readme.txt')

    classMap = {}
    with open(readmePath, 'w') as f:
        for i, classname in enumerate(classnames):
            f.write(f'{classname}:{i}\n')
            classMap[classname] = i
    
    annoFile = open(annoPath, 'w')
    sampleIdx = 1
    for classname in classnames:
        srcImageDir = os.path.join(oriDir, classname)
        imagenames = os.listdir(srcImageDir)
        for imagename in imagenames:
            srcImagePath = os.path.join(srcImageDir, imagename)
            savename = f'{str(sampleIdx).zfill(7)}.jpg'
            saveImagePath = os.path.join(imageDir, savename)
            os.rename(srcImagePath, saveImagePath)
            annoFile.write(f'./images/{savename},{classMap[classname]}\n')
            sampleIdx += 1
    
    annoFile.close()

if __name__ == '__main__':
    pass
    # gather_dataset(oriDir='./datasets/origin', saveDir='./datasets')