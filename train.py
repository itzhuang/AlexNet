import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 这只是定义了device，回来还要把要送进gpu的东西送到gpu？
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),   #随机水平翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),   # transforms.ToTensor（）会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}  # 参数mean和std分别表示图像每个通道的均值和方差
                                  # 将每一个通道的数据先计算出其方差与均值，然后再将其每一个通道内的每一个数据减去均值，再除以方差，得到归一化后的结果

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.abspath(os.getcwd())  # get data root path   #绝对路径
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = os.path.join(data_root, "data_set")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #返回一个对象
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])  #1每个类别需要单独成立一个文件夹 2每个类别里面的图片需要按顺序排列（无论使用英语还是数字
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items()) #反转后数字在前面，输入数字可以直接输出类别名
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4) #将python对象编码成Json字符串
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str) #json_file里写json_str

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw) #创建了一个Dateloader对象

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout) #可迭代对象，file是输出进度条位置
        for step, data in enumerate(train_bar):
            images, labels = data  #list:2 (32.3,224,224) (32,)
            optimizer.zero_grad() #防止梯度累积
            outputs = net(images.to(device))  # (32,5)二维，batchsize，每个分类概率
            loss = loss_function(outputs, labels.to(device)) # loss((batchsize,class),class)第二个class之后当索引值 输出是一维的tensor
            loss.backward()   # 这一步反向传播后梯度才存在
            optimizer.step()  #这一步更新值，变化量=梯度X学习率

            # print statistics
            # 这的runningloss是dataloader迭代次数累加（每个epoch）的loss
            running_loss += loss.item() #取出单元素张量的元素值并返回该值，保持原元素类型不变。精度比直接取高？

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # 最大值索引
                predict_y = torch.max(outputs, dim=1)[1]  #max函数返回最大值和索引，第一个位置是最大值，第二个是索引
                # eq返回[true,T,T,F]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))  #除train steps是因为running loss累计太多了？

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()