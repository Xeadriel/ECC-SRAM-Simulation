import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import storage
from torchvision import datasets
from torchvision import transforms
from quantize import QuanNet, QuanConv2d, QuanLinear
from prune import PruningTrainPlugin, sparsity_of_tensors
import torch.nn.utils.prune as pt_prune
import errorApply_Mantissa as EA

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cpu")

# nn.Module -> QuanNet
class LeNet5(QuanNet):
    def __init__(self, n_bits: int):
        super().__init__(n_bits)

        # Conv2d -> QuanConv2d
        self.conv1 = QuanConv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = QuanConv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = QuanConv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1))
        # Linear -> QuanLinear
        self.fc1 = QuanLinear(in_features=120, out_features=84)
        self.fc2 = QuanLinear(in_features=84, out_features=10)

    # stack layers in forward_() instead of forward()
    def forward_(self, x):
        x = self.conv1(x)
        x = self.conv1.relu(x)  # F.relu() -> self.[layer].relu()
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = self.conv2.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = self.conv3.relu(x)

        x = x.view(x.size()[0], x.size()[1])
        x = self.fc1(x)
        x = self.fc1.relu(x)

        x = self.fc2(x)

        return x

def load_data_set():
    path = r"datasets/mnist_data_set"
    # 预处理
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data_set = datasets.MNIST(path, train=True, download=True, transform=process)
    test_data_set = datasets.MNIST(path, train=False, download=True, transform=process)

    # 拆分训练集和测试集
    train_data, test_data = [], []
    for i in range(len(train_data_set.data)):
        train_data.append(train_data_set[i][0])
    for i in range(len(test_data_set.data)):
        test_data.append(test_data_set[i][0])
    train_data, train_label, test_data, test_label = torch.stack(train_data), train_data_set.targets, torch.stack(test_data), test_data_set.targets

    train_data, train_label, test_data, test_label = train_data.to(device), train_label.to(device), test_data.to(device), test_label.to(device)
    return train_data, train_label, test_data, test_label

def train(net, train_data, train_label, n_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1)
    criterion = criterion.to(device)

    loss_list = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        print("epoch: {}, training_loss: {}".format(
            epoch + 1,
            loss
        ))
        loss_list.append(float(loss))

    plt.plot(np.arange(n_epochs), loss_list)
    plt.savefig("lenet5_loss.png")

def global_prune(net):
    # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html?highlight=prune
    parameters_to_prune = (
        (net.conv1, 'weight'),
        (net.conv2, 'weight'),
        (net.conv3, 'weight'),
        (net.fc1, 'weight'),
        (net.fc2, 'weight'),
    )

    # remove 90% weights
    pt_prune.global_unstructured(parameters_to_prune, pruning_method=pt_prune.L1Unstructured, amount=0.9)
    for p in parameters_to_prune:
        pt_prune.remove(p[0], p[1])  # remove mask

    print("Sparsity in conv1.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv1.weight], 0)))
    print("Sparsity in conv2.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv2.weight], 0)))
    print("Sparsity in conv3.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv3.weight], 0)))
    print("Sparsity in fc1.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.fc1.weight], 0)))
    print("Sparsity in fc2.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.fc2.weight], 0)))
    print("Global sparsity: {:.2f}%".format(100. * sparsity_of_tensors([net.conv1.weight, net.conv2.weight, net.conv3.weight, net.fc1.weight, net.fc2.weight], 0)))

def prune_train(net, train_data, train_label, n_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1)

    criterion = criterion.to(device)

    # plugin this code snippet
    pft = PruningTrainPlugin()
    pft.set_net_named_parameters(net.named_parameters())

    loss_list = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        # plugin this code snippet
        pft.after_optimizer_step()

        print("train after prune, epoch: {}, training_loss: {}".format(
            epoch + 1,
            loss
        ))
        loss_list.append(float(loss))

    plt.plot(np.arange(n_epochs), loss_list)
    plt.savefig("lenet5_prune_loss.png")

def acc(net, test_data, test_label):
    correct = 0
    with torch.no_grad():
        for i in range(test_data.size()[0]):
            data, target = test_data[i], test_label[i]
            data = data.view(1, data.size()[0], data.size()[1], data.size()[2])
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / test_data.size()[0]
    error = 1 - accuracy
    print(accuracy)
    return accuracy
    # print('\nTest set: Accuracy: {}/{} ({:.0f}%), Error: {}/{} ({:.0f}%)\n'.format(
    #     correct,
    #     test_data.size()[0],
    #     100. * accuracy,
    #     test_data.size()[0] - correct,
    #     test_data.size()[0],
    #     100. * error
    # ))

def main_work_flow():
   train_data, train_label, test_data, test_label = load_data_set()
   n_epochs = 100
   print("sparse quantized model:")
   net = LeNet5(16)
   net = net.to(device)
   net.load_state_dict(torch.load("lenet5_dense_float.pt"), strict=False)

   net.float_mode()
   net.quantize(train_data)
   net.quan_mode()
   acc(net, test_data, test_label)
   for name in net.state_dict():
       print(name)
       print(net.state_dict()[name])
   torch.save(net.state_dict(), "lenet5_dense_quantized_16.pt")

    # step4, used the sparse_quantized model
   print("Evaluate sparse quantized model:")
   net = LeNet5(16)
   net = net.to(device)
   net.load_state_dict(torch.load("lenet5_dense_quantized_16.pt"), strict=False)
   net.quan_mode()
   acc(net, test_data, test_label)

   accor_data_all_runs = []

   for run in range(10):

     bet_data=(1,0.7943282347242815,0.6309573444801932,0.5011872336272722,0.3981071705534972,0.31622776601683794,0.251188643150958,0.19952623149688797,0.15848931924611134,0.12589254117941673,0.1,0.07943282347242814,0.0630957344480193,0.05011872336272722,0.03981071705534971,0.0316227766016838,0.025118864315095794,0.019952623149688786,0.015848931924611134,0.012589254117941668,0.01, 0.007943282347242814, 0.00630957344480193, 0.005011872336272719, 0.003981071705534969, 0.003162277660168376,
    0.002511886431509577, 0.0019952623149688768, 0.0015848931924611108, 0.0012589254117941649, 0.0009999999999999979,
    0.0007943282347242797, 0.0006309573444801917, 0.0005011872336272709, 0.0003981071705534961, 0.00031622776601683696,
    0.0002511886431509572, 0.00019952623149688728, 0.00015848931924611077, 0.00012589254117941623,
    9.999999999999958e-05, 7.94328234724279e-05, 6.309573444801917e-05, 5.011872336272715e-05, 3.9810717055349695e-05,
    3.9810717055349695e-05)
    #  bet_data=(1,0.6309573444801932)
   # bet_data= bet_data1[::-1]
     bt_data = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
     accor_data =np.zeros((len(bet_data),len(bt_data)))


   # for name in net.state_dict():
   #     print(name)
   #     print(net.state_dict()[name])

   # net = LeNet5(16)
   # net = net.to(device)
   # dict=torch.load("lenet5_sparse_quantized_16.pt")
   # print(dict)


     numb=1
     i = 0 #for bet in(0.1,0.07943282347242814,0.0630957344480193,0.0501187233627272,0.03981071705534969,0.031622776601683764,0.02511886431509577,0.019952623149688768,0.01584893192461111,0.012589254117941649,0.00999999999999998,0.007943282347242796,0.006309573444801917,0.00501187233627271,0.003981071705534961,0.0031622776601683694,0.002511886431509572,0.001995262314968873,0.0015848931924611076,0.0012589254117941623):
     for bet in bet_data:

         j=0
         for bt in bt_data:
            net = LeNet5(16)
            net = net.to(device)
            dict = torch.load("lenet5_dense_quantized_16.pt")
            for key in ("conv1.quan_weight", "conv2.quan_weight", "conv3.quan_weight"):
                p = dict[key].int()

                p.numpy()
            # p.dtype = 'int8'
                fc2_q_c = EA.error_Apply(p, 4, bet, [bt], 0)
                # print([bt])
                # print(key)
                # print(fc2_q_c)
                # fc2_q_c_cuda = fc2_q_c.to(device)
                dict[key] = torch.tensor(fc2_q_c).to(device)

            for key in ("fc1.quan_weight", "fc2.quan_weight"):
                p = dict[key].int()

                p.numpy()
            # p.dtype = 'int8'
                fc2_q_c = EA.error_Apply(p, 2, bet, [bt], 0)
                # fc2_q_c_cuda = fc2_q_c.to(device)
            # print(fc2_q_c)
                dict[key] = torch.tensor(fc2_q_c).to(device)
            # print("Bit error rate:",bet)
            net.load_state_dict(dict, strict=False)
            net.quan_mode()
            accor_data[i,j]=acc(net, test_data, test_label)
            j=j+1
         i=i+1
         # print(accor_data)
     accor_data_all_runs.append(accor_data)

     # 保存每次的数据到txt文件
     with open(f'16data_run_{run}.txt', 'w') as f:
         f.write("\t".join(map(str, bet_data)) + "\n")
         for i, bt in enumerate(bt_data):
             f.write(str(bt) + "\t" + "\t".join(map(str, accor_data[:, i])) + "\n")


# 计算平均准确性
   avg_accor_data = np.mean(accor_data_all_runs, axis=0)

   with open(f'16data_avg.txt', 'w') as f:
       f.write("\t".join(map(str, bet_data)) + "\n")
       for i, bt in enumerate(bt_data):
           f.write(str(bt) + "\t" + "\t".join(map(str, avg_accor_data[:, i])) + "\n")

   for i, bt in enumerate(bt_data):
       plt.plot(bet_data, avg_accor_data[:, i], label=f'Bit[{bt}]')  # 修改了这里的label参数

   plt.xscale('log')
   plt.gca().invert_xaxis()  # 倒序x轴
   plt.xlabel('Bit Error Rate')
   plt.ylabel('Accurate Recognition')
   plt.legend(loc='best')
   plt.title('Bit Error Rate and Accurate Recognition')
   plt.grid(True, which='both', linestyle='--', linewidth=0.5)
   plt.show()
   # 绘制折线图

   #   with open('data.txt', 'w') as f:
   #     f.write("\t".join(map(str, bet_data)) + "\n")  # 写入 bet_data 作为列名
   #     for i, bt in enumerate(bt_data):
   #         f.write(str(bt) + "\t" + "\t".join(map(str, accor_data[:, i])) + "\n")  # 写入每行的 bt_data 和对应的 accor_data
   #
   #
   # # 绘制折线图
   # for i, bt in enumerate(bt_data):
   #     plt.plot(bet_data, accor_data[:, i], label=f'bt={bt}')
   #
   # plt.xscale('log')  # 由于bet_data是对数尺度的，所以这里我们使用log尺度
   # plt.xlabel('Bit Error Rate')
   # plt.ylabel('Accurate Recognition')
   # plt.legend('Protect Bit')
   # plt.title('Bit Error Rate and Accurate Recognition')
   # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
   # plt.show()

   # bet1 = np.array(bet_data)
   # bt1 = np.array(bt_data)
   # table_str = "bet\t" + "\t".join(map(str, bet1)) + "\n"
   # for i in range(len(bt1)):
   #     table_str += str(bt1[i]) + "\t" + "\t".join(map(str, accor_data[i,j])) + "\n"
   #
   # # 将表格数据写入txt文件
   # with open("4bit11.txt", "w") as txt_file:
   #     txt_file.write(table_str)

    #fc2_q=dict["conv1.quan_weight"].int()
    #fc2_q.numpy()
    #fc2_q.dtype='int8'
    #print(fc2_q.shape)
    # for i, dim1 in enumerate(fc2_q):
    #     print(i)
    #     print(dim1)
    # error を加える

   # fc2_q_c=EA.error_Apply(fc2_q, 4, 0.03, [], 0)
   # print(fc2_q_c)
   # dict["conv1.quan_weight"]=torch.tensor(fc2_q_c)
   # print(dict["fc2.quan_bias"])
    #dict["fc2.quan_weight"] = torch.ones([10, 84])
    #dict["fc2.bias"] = torch.ones(10)
   # net.load_state_dict(dict, strict=False)
    #net.load_state_dict(torch.load("lenet5_dense_float.pt"), strict=False)
    #net.float_mode()
    #  net.quantize(train_data)
    #acc(net, test_data, test_label)

   #  net.quan_mode()
   #  acc(net, test_data, test_label)
   # # for name in net.state_dict():
        #  print(name)
       #print(net.state_dict()[name])
       # print(net.state_dict()[name].size())
    # conv1t=torch.load("lenet5_sparse_quantized.pt")['conv1.quan_weight']
    # print(conv1t)
    # conv1.quan_bias
    # conv2.quan_weight
    # conv2.quan_bias
    # conv3.quan_weight
    # conv3.quan_bias
    # fc1.quan_weight
    # fc1.quan_bias
    # fc2.quan_weight
    # fc2.quan_bias
    # step5, save model in yaml format
  #  storage.save_quan_model("lenet5_sparse_quantized.pt", "lenet5_sparse_quantized_float.yml", "lenet5_sparse_quantized_int.yml")
  #  storage.save_sparse_model("lenet5_sparse_quantized.pt", "lenet5_sparse_quantized_float_coo.yml", "lenet5_sparse_quantized_int_coo.yml", form="coo")

if __name__ == '__main__':
    main_work_flow()
