# import cv2
# import numpy as np
# from monai.transforms import ScaleIntensityRangePercentiles
# img = cv2.imread("C:\\Users\\lenovo\\Desktop\\00000001_000.png")#读取x图像
# if img is not None:
#     scaler = ScaleIntensityRangePercentiles(lower=1 , upper=99, b_min=0,b_max=255,clip=True)
#     img_adjusted = scaler(img).astype("uint8")
#     cv2.imshow("img_adjusted",img_adjusted)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     Type = type(img)
#     Shape = img.shape
#     print(Type)
#     print(Shape)
#     print("max", np.max(img))
#     print("min", np.min(img))
#     print("dtype",img.dtype)
# else:
#     print("no image")
# import running
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets,transforms
# from tqdm import tqdm
# import os
# import
#
# device = torch.device("cuda" if torch.cuda.is_available()else"cpu")
# train_transform = transforms.Compose([
#     transforms.Resize(224,224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(224,224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# ])
#
# trainset = datasets.ImageFolder(root='D/learn_pytorch/data/train',transform=train_transform)
# testset = datasets.ImageFolder(root='D/learn_pytorch/data/test',transform=test_transform)
# train_loader = DataLoader(trainset,batch_size=32,num_workers=2,shuffle=True)
# test_loader = DataLoader(testset,batch_size=32,num_workers=0,shuffle=False)
# def train(model,train_loader,criterion,optimizer,num_epochs):
#     best_acc = 0
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0
#         for inputs,labels in tqdm(train_loader,desc=f"epoch:{epoch+1}/{num_epochs}",unit="batch"):
#             inputs,labels = inputs.to(device),labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(inputs),labels)
#             loss.backward()#反向传播
#             optimizer.step()#更新参数
#             running_loss+=loss.item()*inputs.size(0)#用loss乘以批次大小，得到该批次的loss
#         epoch_loss = running.loss/len(train_loader.dataset)
#         print(f"epoch[{epoch+1}/{num_epochs},Train_loss:{epoch_loss:.4f}]")
#         accuracy = evaluate(model,test_loader,criterion)
#         if accuracy>best_acc:
#             best_acc = accuracy
#             save_model(model,save_path)
#             print("model saved with best acc",best_acc)
#
# def evaluate(model,test_loader,criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total  = 0
#     with torch.no_grad():
#         for inputs,labels in test_loader:
#             inputs,labels = inputs.to(device),labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs,labels)
#             test_loss += loss.item()*inputs.size(0)
#             _,predicted = torch.max(outputs,1)
#             total +=total + labels.size(0)
#             correct = correct + (predicted ==labels).sum().item()
#     avg_loss = test_loss/len(test_loader.dataset)
#     accuracy = 100.0*correct/total
#     print(f"test loss:{avg_loss:.4f},Accuracy:{accuracy:.4f}%")
#     return accuracy
# def save_model(model,save_path):
#     torch.save(model.state_dict(),save_path)
#
# if __name__ == "__main__":
#     num_epochs = 10
#     learning_rate = 0.001
#     num_class = 3
#     save_path = r"model_path\best.pth"
#     model = simplecnn(num_class).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimiter = optim.Adam(model.parameters(),lr=learning_rate)
#     train(model,train_loader,criterion,optimiter,num_epochs)
#     evaluate(model,test_loader,criterion)

# import napari
# import SimpleITK as sitk
#
# img = sitk.ReadImage("DET0000301_avg.nii.gz")
# arr = sitk.GetArrayFromImage(img)
# viewer = napari.Viewer()
# viewer.add_image(arr, name="image")
# napari.run()
#
# import SimpleITK as sitk
# import numpy as np
# import matplotlib.pyplot as plt
#
# def get_img_info(sitk_img):
#     """读取sitk图像，返回数组，shape,dtype,min,max"""
#     arr = sitk.GetArrayFromImage(sitk_img)
#     return arr,arr.shape,arr.dtype,float(arr.min()),float(arr.max())
# def intensity_map(arr,modality):
#     arr = arr.astype(np.float32)
#     if modality == "ct":
#         vmin, vmax = -1000,400
#         arr_clipped = np.clip(arr,vmin,vmax)
#     elif modality in ["xray","mri","ultrasound"]:
#         p1 = np.percentile(arr,1)
#         p99 = np.percentile(arr,99)
#         arr_clipped = np.clip(arr,p1,p99)
#         vmin,vmax = p1,p99
#     else:
#         raise ValueError("modality must be ct/xray/mri/ultrasound")
#     arr_norm = (arr_clipped - vmin) /(vmax - vmin + 1e-8)
#     return arr_norm,vmin,vmax
# def load_and_show(file_path,modality):
#     img = sitk.ReadImage(file_path)
#     arr,shape,dtype,raw_min,raw_max = get_img_info(img)
#     print("="*50)
#     print(f"[{modality.upper()}] path: {file_path}")
#     print(f"shape(x,y,z):{shape}")
#     print(f"dtype{dtype}")
#     print(f"原始数值范围:min={raw_min:.2f},max={raw_max:.2f}")
#
#     arr_norm,vmin,vmax = intensity_map(arr,modality)
#     print(f"映射使用范围 vmin={vmin:.2f},vamx={vmax:.2f}")
#     print(f"映射后范围: min={arr_norm.min():.2f},max={arr_norm.max():.2f}")
#
#     z_mid = shape[0]//4
#     print(shape[0])
#     slice_img = arr_norm[z_mid,:,:]
#
#     plt.figure(figsize=(5,5))
#     plt.imshow(slice_img,cmap="gray")
#     plt.title(f"{modality} | sliec z={z_mid}")
#     plt.axis("off")
#
#     plt.show()
#     return arr_norm
# if __name__ == "__main__":
#     load_and_show("DET0000301_avg.nii.gz",modality="ct")
#     # load_and_show("",modality="xray")
#     # load_and_show("",modality="mri")
#     # load_and_show("",modality="ultrasound")




import pydicom


ds = pydicom.dcmread("IMG-0037-00001.dcm")
print(ds.Modality,ds.Rows,ds.Columns,ds.BitsStored,ds.PhotometricInterpretation)
arr = ds.pixel_array
print(arr.dtype,arr.min(),arr.max())
print(ds.RescaleSlope,ds.RescaleIntercept,ds.PixelSpacing,ds.SliceThickness)
#
# slope = float(ds.RescaleSlope)      # often 1
# intercept = float(ds.RescaleIntercept)  # often - 1024 for CT
# hu = ds.pixel_array.astype("float32") * slope + intercept



# PHI_TAGS = ["PatientName", "PatientID", "PatientBirthDate",
#             "AccessionNumber", "InstitutionName", "ReferringPhysicianName"]
# for t in PHI_TAGS:
#     val = getattr(ds, t, "")
#     print(t, "=", repr(val))   # must be empty/removed on a de-identified set


#week2
#!/usr/bin/env python3
"""
DICOM 基础读片实验
--------------------------------
公开样例：
  data/CT_small.dcm   pydicom 官方测试集的真实 CT 切片
  data/MR_small.dcm   pydicom 官方测试集的真实 MR 切片
  data/CT_small_MONOCHROME1.dcm  由 CT_small.dcm 派生（仅翻转 (0028,0004) 标签，
                                 用来模拟 DX/MG 那类真正的 MONOCHROME1 文件的显示行为）

运行：  python3 dicom_basics.py
输出：  终端报告 + figures/*.png
"""

import os
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------- 1. 读标签
def report_tags(path):
    ds = pydicom.dcmread(path)
    name = os.path.basename(path)
    print("=" * 68)
    print(f"文件: {name}")
    print("-" * 68)
    # 用户要求四个必备字段
    print(f"{'Modality':<26}{ds.Modality!r}")
    print(f"{'Rows':<26}{ds.Rows}")
    print(f"{'Columns':<26}{ds.Columns}")
    print(f"{'BitsStored':<26}{ds.BitsStored}")
    print(f"{'PhotometricInterpretation':<26}{ds.PhotometricInterpretation!r}")

    # 补充字段（有助于解释取值范围）
    for kw in ("BitsAllocated", "PixelRepresentation", "SamplesPerPixel"):
        print(f"{kw:<26}{ds.get(kw, '<absent>')}")

    # ------------------------------------------------------- 3. 定量字段
    print("-" * 68)
    print("[几何 / 灰度定标标签]")
    slope = ds.get("RescaleSlope", None)
    inter = ds.get("RescaleIntercept", None)
    print(f"{'RescaleSlope':<26}{slope if slope is not None else '<absent>'}")
    print(f"{'RescaleIntercept':<26}{inter if inter is not None else '<absent>'}")

    if "PixelSpacing" in ds:
        ps = [float(v) for v in ds.PixelSpacing]
        print(f"{'PixelSpacing (row,col) mm':<26}{ps}")
    else:
        print(f"{'PixelSpacing':<26}<absent>")

    print(f"{'SliceThickness mm':<26}"
          f"{ds.get('SliceThickness', '<absent>')}")
    if "WindowCenter" in ds:
        print(f"{'WindowCenter / Width':<26}"
              f"{ds.get('WindowCenter')} / {ds.get('WindowWidth')}")

    return ds


# ------------------------------------------------------- 2. 像素数组
def report_pixels(ds, name):
    arr = ds.pixel_array
    print("-" * 68)
    print("[pixel_array]")
    print(f"{'shape':<26}{arr.shape}")
    print(f"{'dtype':<26}{arr.dtype}")
    print(f"{'min / max':<26}{arr.min()} / {arr.max()}")
    print(f"{'unique values':<26}{len(np.unique(arr))}")

    is_8bit_display = arr.dtype == np.uint8 and arr.max() <= 255
    print(f"{'落在 0-255?':<26}"
          f"{'是' if is_8bit_display else '否'}"
          f"   -> {'8 位显示级数据' if is_8bit_display else '存储级原始值，不是显示灰度'}")

    # 转换到物理量（HU / 任意单位）
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    phys = arr.astype(np.float32) * slope + inter
    unit = "HU" if ds.Modality == "CT" and "RescaleIntercept" in ds else "a.u."
    print(f"{'重标定后范围 (' + unit + ')':<26}{phys.min():.1f} / {phys.max():.1f}")
    return arr, phys, unit


# ------------------------------------------------------- 4. 显示
def show_raw(arr, title, fname, note=""):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    # vmin/vmax 固定为数据全局范围，保证“原始像素”不被错误拉伸
    ax.imshow(arr, cmap="gray", vmin=float(arr.min()), vmax=float(arr.max()))
    ax.set_title(title + ("\n" + note if note else ""), fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def show_pair(a, b, ta, tb, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=130)
    for ax, img, t in zip(axes, (a, b), (ta, tb)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def normalize_display(arr):
    """把任意位深 / 有符号数据映射到 0-1，模拟查看器的线性归一化显示。"""
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-8)


# ---------------------------------------------------------------- main
def main():
    print("\n########## CT 切片 ##########")
    ct = report_tags(os.path.join(DATA, "CT_small.dcm"))
    ct_arr, ct_phys, ct_unit = report_pixels(ct, "CT_small.dcm")

    print("\n########## MR 切片 ##########")
    mr = report_tags(os.path.join(DATA, "MR_small.dcm"))
    mr_arr, mr_phys, mr_unit = report_pixels(mr, "MR_small.dcm")

    # ------------------------------------------------------ 4. 显示阶段
    print("\n########## 显示 ##########")
    show_raw(ct_arr, "CT raw stored values (int16, -1024 offset)",
             "01_ct_raw.png",
             "rescaled to HU: %.0f .. %.0f" % (ct_phys.min(), ct_phys.max()))
    # 用 HU 正确加窗（软组织窗 W:400 L:40）
    w, l = 400.0, 40.0
    lo, hi = l - w / 2, l + w / 2
    ct_win = np.clip((ct_phys - lo) / (hi - lo), 0, 1)
    show_raw(ct_win, "CT windowed to HU (W=400, L=40)", "02_ct_window_hu.png")

    show_raw(mr_arr, "MR raw stored values (arbitrary units)",
             "03_mr_raw.png",
             "no rescale tags -> intensity is not calibrated")

    # MONOCHROME1
    m1 = pydicom.dcmread(os.path.join(DATA, "CT_small_MONOCHROME1.dcm"))
    print("\n########## MONOCHROME1 ##########")
    print(f"PhotometricInterpretation = {m1.PhotometricInterpretation!r}")
    m1_arr = m1.pixel_array
    print(f"pixel_array dtype={m1_arr.dtype}, range="
          f"{m1_arr.min()} / {m1_arr.max()}  (与 MONOCHROME2 版本完全相同)")
    raw_norm = normalize_display(m1_arr)
    inv_norm = 1.0 - raw_norm
    show_pair(raw_norm, inv_norm,
              "MONOCHROME1: shown as MONOCHROME2 (min->black)  = WRONG",
              "MONOCHROME1: inverted (min->white)  = CORRECT",
              "04_monochrome1_raw_vs_inverted.png")

    print("\n完成。图在", FIG)


if __name__ == "__main__":
    main()





















