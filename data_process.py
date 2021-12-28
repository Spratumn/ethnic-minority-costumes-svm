import os



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


gather_dataset(oriDir='./datasets/origin', saveDir='./datasets')