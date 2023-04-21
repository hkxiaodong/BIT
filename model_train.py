import torch
import torch.nn as nn
from model import BMT
from block import *
from Dataset_MVSA_single import get_dataset as MVSA_S
from Dataset_TumEmo import get_dataset as TumEmo
from Dataset_MVSA_m import get_dataset as MVSA_M
from Dataset_MHF import get_dataset as MHF
import torch.optim as optim
import os
import numpy as np
import clip
from opts import AverageMeter
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cuda


dataset = 'MHF' # ''' TumEmo or MVSA_S or MVSA_M or MHF

if dataset == 'TumEmo':
    batch_size = 16
    cls_num = 7
    epoch = 5  #
    index = -1 # TumEmo 数据集在去-1的效果比较好
    cls_emb = True
    memory = 30
    depth = 5
    train_data, test_data = TumEmo(batch_size=batch_size)

if dataset == 'MVSA_S':
    batch_size = 16
    cls_num = 3
    epoch = 20  #
    index = 0
    depth = 3
    memory = 30
    cls_emb = True
    train_data, test_data = MVSA_S(batch_size=batch_size)

if dataset == 'MVSA_M':
    batch_size = 16
    cls_num = 3
    epoch = 6  #
    index = -1 # MVSA_M 数据集在取index=0的效果比较好
    depth = 3
    memory = 30
    train_data, test_data = MVSA_M(batch_size=batch_size)

if dataset == 'MHF':
    batch_size = 16
    cls_num = 2
    epoch = 6  #
    index = 0
    depth = 3
    memory = 30
    cls_emb = True
    train_data, test_data = MHF(batch_size=batch_size)

for i in range(4, 7):
    print(i)
    clip_pth = 'ViT-B/16'
    if clip_pth in ['ViT-B/16', 'ViT-B/32']:
        dim = 512
    elif clip_pth in ['ViT-L/14']:
        dim = 768
    #    pad_size = 77
    model, _ = clip.load(clip_pth)
    clip_model = model.cuda()
    lamd = 'v2'
    fusion = True
    depth = i
    net = BMT(dim=dim, num_heads=8, memory_slots=memory, depth=depth,
              cls_num=cls_num, Fusion=fusion, lamd=lamd, index=index, cls_emb=cls_emb).cuda()

    XE_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.86)
    def train_(epoch, total_epoch):
        # 每次输入barch_idx个数据
        net.train()
        tasks_top1 = AverageMeter()
        tasks_losses = AverageMeter()

        for batch_idx, data in enumerate(train_data):
            image,  text, emo_label = data[0], data[1], data[2]

            image,  emo_label = image.to(device), emo_label.to(device)

            clip_model.eval()
            text_tokens = clip.tokenize(text, truncate=True).cuda()
            text_f = clip_model.get_text_feature(text_tokens)
            text_f = torch.as_tensor(text_f, dtype=torch.float32)
            image_f = clip_model.get_image_feature(image)
            image_f = torch.as_tensor(image_f, dtype=torch.float32)

            output = net(image_f, text_f)
            loss = XE_loss(output, emo_label.long())
            emo_res = output.max(1)[1]  # emo 预测结果
            cor = emo_res.eq(emo_label).sum().item()
            tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
            tasks_losses.update(loss.item(), emo_label.size(0))

            #######################################
            if batch_idx % 50 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                      .format(epoch + 1, total_epoch, batch_idx + 1, len(train_data), tasks_losses.val, tasks_top1.val,
                              tasks_top1.avg))
                print(emo_res)
                print(emo_label.int())
            #################
            # Backward and optimize#
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

        print("Epoch [{}/{}], Loss Avg: {:.4f}, Acc Avg: {:.4f}"
              .format(epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg))
        return tasks_top1.avg


    def test_(epoch, total_epoch, train_acc):
        # 每次输入barch_idx个数据
        net.eval()
        global best_acc
        best_acc = 60  # 大于72就进行保存
        tasks_top1 = AverageMeter()
        tasks_losses = AverageMeter()
        prediction = []
        truth = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_data):
                image, text, emo_label = data[0], data[1], data[2]

                image, emo_label = image.to(device), emo_label.to(device)


                clip_model.eval()
                text_tokens = clip.tokenize(text, truncate=True).cuda()
                text_f = clip_model.get_text_feature(text_tokens)
                text_f = torch.as_tensor(text_f, dtype=torch.float32)
                image_f = clip_model.get_image_feature(image)
                image_f = torch.as_tensor(image_f, dtype=torch.float32)

                output = net(image_f, text_f)
                loss = XE_loss(output, emo_label.long())
                emo_res = output.max(1)[1]  # emo 预测结果

                prediction.append(emo_res.cpu().numpy())
                truth.append(emo_label.cpu().numpy())

                cor = emo_res.eq(emo_label).sum().item()

                tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
                tasks_losses.update(loss.item(), emo_label.size(0))
                if batch_idx % 50 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                          .format(epoch + 1, total_epoch, batch_idx + 1, len(test_data), tasks_losses.val, tasks_top1.val,
                                  tasks_top1.avg))

        print('Test result：')
        print("Epoch [{}/{}], Loss Avg: {:.4f}, Acc Avg: {:.4f}"
              .format(epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg))

        # print('Test set Epoch: {} \tavr Loss: {:.6f}\t avr emo_Acc: {:.2f}%({}/{})'.format(
        #             epoch, emo_loss_ / len(test_data.dataset), 100. * emo_correct / len(test_data.dataset),
        #             emo_correct, len(test_data.dataset)))
        # 保存
        acc = round(tasks_top1.avg, 2)
        train_acc = round(train_acc, 2)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'train_acc': train_acc,
                'epoch': epoch,
            }

            if not os.path.isdir(f'D:/paper_3/{dataset}_memory={memory}_lamd={lamd}'):
                os.mkdir(f'D:/paper_3/{dataset}_memory={memory}_lamd={lamd}')

            torch.save(state, f'D:/paper_3/{dataset}_memory={memory}_lamd={lamd}/{acc}_{train_acc}_{epoch}_depth={depth}_memory={memory}_index={index}.pth')
            prediction = np.array(prediction).flatten()
            truth = np.array(truth).flatten()

            np.savez(f'D:/paper_3/{dataset}_memory={memory}_lamd={lamd}/{acc}_{train_acc}_{epoch}_depth={depth}_memory={memory}_index={index}.npz', pred=prediction,
                     labels=truth)

            best_acc = acc
            print('Save done')
        # print(emo_res)
        # print(emo_label.int())



    for epoch_ in range(epoch):
        train_acc = train_(epoch_, epoch)
        test_(epoch_, epoch, train_acc)
        print("第%d个epoch的学习率：%f" % (epoch_, optimizer.param_groups[0]['lr']))
        lr_scheduler.step()