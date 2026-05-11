import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import glob
import model as models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 是否展示结果
show_result = False

# 加载模型和权重
model = models.Change_detection()
model = nn.DataParallel(model)
pretrain_deeplab_path = "./Com/model_best.pth"
checkpoint = torch.load(pretrain_deeplab_path, map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()

# 初始化评估参数
TP, TN, FP, FN = 0, 0, 0, 0

for file in tqdm.tqdm(glob.glob('./test_dataset_com/A/*')):
    filename = os.path.basename(file)
    testCase1_01 = f'./test_dataset_com/A/{filename}'
    testCase1_02 = f'./test_dataset_com/B/{filename}'
    gt_path = f'./test_dataset_com/label/{filename}'

    img1 = Image.open(testCase1_01)
    img2 = Image.open(testCase1_02)
    gt = Image.open(gt_path)

    # 初始化结果存储
    gt_show = np.zeros((512, 512), dtype=np.uint8)

    # 将图像调整到固定大小
    temp_img1 = img1.resize((512, 512))
    temp_img2 = img2.resize((512, 512))
    temp_gt = gt.resize((512, 512))

    # 处理图像和标签
    temp_img1 = np.array(temp_img1, dtype=np.uint8)
    temp_img2 = np.array(temp_img2, dtype=np.uint8)
    temp_gt = np.array(temp_gt, dtype=np.uint8)

    temp_gt[temp_gt > 0] = 1  # 将标签二值化

    temp_img1 = TF.to_tensor(temp_img1)
    temp_img2 = TF.to_tensor(temp_img2)
    temp_img1 = TF.normalize(temp_img1, mean=[0.44758545, 0.44381796, 0.37912835], std=[0.21713617, 0.20354738, 0.18588887])
    temp_img2 = TF.normalize(temp_img2, mean=[0.34384388, 0.33675833, 0.28733085], std=[0.1574003, 0.15169171, 0.14402839])

    inputs1, inputs2 = temp_img1.cuda(), temp_img2.cuda()

    with torch.no_grad():
        inputs1 = Variable(inputs1.unsqueeze(0))
        inputs2 = Variable(inputs2.unsqueeze(0))
        output_map = model(inputs1, inputs2)

    # 获取预测结果
    param = 1  # 用于平衡Precision和Recall
    output_map[:, 1, :, :] += param
    pred = output_map.argmax(dim=1, keepdim=True).cpu().numpy().squeeze()
    pred = (pred * 255).astype(np.uint8)

    # 将预测结果保存到 gt_show
    gt_show[:, :] = pred

    # 计算评价指标
    confmatrix_TP = (pred > 0) * (temp_gt > 0)
    confmatrix_TN = (pred == 0) * (temp_gt == 0)
    confmatrix_FP = (pred > 0) * (temp_gt == 0)
    confmatrix_FN = (pred == 0) * (temp_gt > 0)

    TP += np.sum(confmatrix_TP)
    TN += np.sum(confmatrix_TN)
    FP += np.sum(confmatrix_FP)
    FN += np.sum(confmatrix_FN)

    # 显示结果
    if show_result:
        plt.subplot(1, 4, 1)
        plt.imshow(img1)
        plt.title('Image 1')
        plt.subplot(1, 4, 2)
        plt.imshow(img2)
        plt.title('Image 2')
        plt.subplot(1, 4, 3)
        plt.imshow(gt, cmap='gray')
        plt.title('Ground Truth')
        plt.subplot(1, 4, 4)
        plt.imshow(gt_show, cmap='gray')
        plt.title('Prediction')
        plt.show()

# 计算最终指标
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
overall_acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1_score)
print("IoU      :", iou)
print("OA       :", overall_acc)
