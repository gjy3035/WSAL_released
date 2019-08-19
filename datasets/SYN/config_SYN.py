import os
from config import cfg


root = cfg.DATA.ROOT + 'RAND_CITYSCAPES'
raw_img_path = os.path.join(root, 'RGB')
raw_mask_path = os.path.join(root, 'CityIdLabels')

#processed_path = os.path.join(root, 'processed')
#processed_train_path = os.path.join(processed_path, 'train')
#processed_val_path = os.path.join(processed_path, 'val')



palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

ignored_label = 255

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
label_colors_voc = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
label_colors = [
				(128, 64,128), (244, 35,232), ( 70, 70, 70), # 0-2
				(102,102,156), (190,153,153), (153,153,153), # 3-5
				(250,170, 30), (220,220,  0), (107,142, 35), # 6-8
				(152,251,152), ( 70,130,180), (220, 20, 60), # 9-11
				(255,  0,  0), (  0,  0,142), (  0,  0, 70), # 12-14
                (  0, 60,100), (  0, 80,100), (  0,  0,230), # 15-17
                (119, 11, 32),(0,0,0),(255,255,255)                                # 18==
               ]
