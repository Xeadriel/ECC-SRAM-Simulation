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
import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt

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
    # print(correct)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%), Error: {}/{} ({:.0f}%)\n'.format(
        correct,
        test_data.size()[0],
        100. * accuracy,
        test_data.size()[0] - correct,
        test_data.size()[0],
        100. * error
    ))

    return accuracy

def main_work_flow():
    train_data, train_label, test_data, test_label = load_data_set()
    # n_epochs = 100

    # # step1, train a dense model with float nn parameters
    # net = LeNet5(8)
    # net = net.to(device)
    # net.float_mode()
    # net.load_state_dict(torch.load("lenet5_dense_float.pt", map_location=torch.device('cpu')), strict=False)
    # for name in net.state_dict():
    #     print(name)
    #     print(net.state_dict()[name])
    # train(net, train_data, train_label, n_epochs=n_epochs)
    # acc(net, test_data, test_label)
    # torch.save(net.state_dict(), "lenet5_sparse_float.pt")
    # torch.save(net.state_dict(), "lenet5_dense_float.pt")

    #     # step2, prune the dense model and retrain
    # print("sparse float model:")
    # net = LeNet5(8)
    # net = net.to(device)
    # net.load_state_dict(torch.load("lenet5_dense_float.pt", map_location=torch.device('cpu')), strict=False)
    # net.float_mode()
    # global_prune(net)  # your own prune strategy
    # prune_train(net, train_data, train_label, n_epochs=n_epochs)  # retrain
    # acc(net, test_data, test_label)
    # torch.save(net.state_dict(), "lenet5_sparse_float.pt")

    #     # step3, quantize the model
    # print("sparse quantized model:")
    # net = LeNet5(8)
    # net = net.to(device)
    # net.load_state_dict(torch.load("lenet5_dense_float.pt",map_location=torch.device('cpu')), strict=False)

    # net.float_mode()
    # net.quantize(train_data)
    # net.quan_mode()
    # acc(net, test_data, test_label)
    # for name in net.state_dict():
    #     print(name)
    #     print(net.state_dict()[name])
    # torch.save(net.state_dict(), "lenet5_dense_quantized.pt")

    #     # step4, used the sparse_quantized model
    # print("Evaluate sparse quantized model:")
    # net = LeNet5(8)
    # net = net.to(device)
    # net.load_state_dict(torch.load("lenet5_dense_quantized.pt", map_location=torch.device('cpu')), strict=False)
    # net.quan_mode()
    # acc(net, test_data, test_label)
    # for name in net.state_dict():
    #     print(name)
    #     print(net.state_dict()[name])

    #     net = LeNet5(8)
    #     net = net.to(device)
    #     dict=torch.load("lenet5_sparse_quantized.pt", map_location=torch.device('cpu'))
    #     print(dict) 


    numb=1
    #for bet in(0.1,0.07943282347242814,0.0630957344480193,0.0501187233627272,0.03981071705534969,0.031622776601683764,0.02511886431509577,0.019952623149688768,0.01584893192461111,0.012589254117941649,0.00999999999999998,0.007943282347242796,0.006309573444801917,0.00501187233627271,0.003981071705534961,0.0031622776601683694,0.002511886431509572,0.001995262314968873,0.0015848931924611076,0.0012589254117941623):
    for bet in (0.01,0.007943282347242814,0.00630957344480193,0.005011872336272719,0.003981071705534969,0.003162277660168376,0.002511886431509577,0.0019952623149688768,0.0015848931924611108,0.0012589254117941649,0.0009999999999999979,0.0007943282347242797,0.0006309573444801917,0.0005011872336272709,0.0003981071705534961,0.00031622776601683696,0.0002511886431509572,0.00019952623149688728,0.00015848931924611077,0.00012589254117941623,9.999999999999958e-05,7.94328234724279e-05,6.309573444801917e-05,5.011872336272715e-05,3.9810717055349695e-05,3.9810717055349695e-05):
        net = LeNet5(8)
        net = net.to(device)
        dict = torch.load("lenet5_sparse_quantized.pt", map_location=torch.device('cpu'))
        for key in ("conv1.quan_weight", "conv2.quan_weight", "conv3.quan_weight"):
            p = dict[key].int()
            print(p.dtype)
            p.numpy()
        # p.dtype = 'int8'
            # print("###############")
            # print(p)
            # print("#################################")
            # print("#################################")
            # fc2_q_c = EA.error_Apply(p, 4, bet, [bt], 0)
            return
            # print(fc2_q_c)
            # print("##############")
            # return
            dict[key] = torch.tensor(fc2_q_c)

        for key in ("fc1.quan_weight", "fc2.quan_weight"):
            p = dict[key].int()
            p.numpy()
        # p.dtype = 'int8'
            fc2_q_c = EA.error_Apply(p, 2, bet, [bt], 0)
        # print(fc2_q_c)
            dict[key] = torch.tensor(fc2_q_c, dtype=torch.uint32)
        # print("Bit error rate:",bet)
        net.load_state_dict(dict, strict=False)
        net.quan_mode()
        acc(net, test_data, test_label)
        #  print(bet)



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




####### error injection #########

def inject_bit_errors(tensor: torch.Tensor, p_error=1e-5, 
                      protected_range=(23, 31), p_protected=1e-8, ignore_range=None):
    """
    Randomly flips bits in a float32 tensor to simulate hardware bit errors.
    
    Args:
        tensor: torch.Tensor (float32, CPU or CUDA)
        p_error: probability of flipping each bit (for unprotected bits)
        protected_range: (low_bit, high_bit) inclusive bit index range [0..31] protected
        p_protected: probability of flipping each bit in protected range
    Returns:
        torch.Tensor with bit-flip errors injected
    """
    device = tensor.device
    x = tensor.clone().detach().view(torch.int32)  # reinterpret bits

    # generate random flips for all 32 bits
    rand = torch.rand((*x.shape, 32), device=device)
    
    # build flip mask for each bit position
    flip_mask = (rand < p_error).to(torch.int32)
    
    # protected bit range has lower flip probability
    low, high = protected_range
    if low <= high:
        protected_mask = (rand < p_protected).to(torch.int32)
        flip_mask[..., low:high+1] = protected_mask[..., low:high+1]

    # convert bitwise masks to integer masks
    bit_values = (1 << torch.arange(32, device=device, dtype=torch.int32))
    bit_masks = (flip_mask * bit_values).sum(dim=-1)

    # flip bits using XOR
    corrupted = torch.bitwise_xor(x, bit_masks)

    # ignore certain ranges of bits
    if ignore_range is not None:
        low, high = ignore_range
        if low <= high:
            corrupted[..., low:high+1] = x[..., low:high+1]
    
    # reinterpret back to float
    return corrupted.view(torch.float32)

def inject_bit_errors_int(
    tensor: torch.Tensor,
    p_error: float = 1e-5,
    protected_range: Tuple[int, int] = (23, 31),
    p_protected: float = 1e-8,
    ignore_range: Optional[Tuple[int, int]] = None,
    nbits: int = 32,
) -> torch.Tensor:
    """
    Flip random bits in an integer tensor to simulate hardware bit errors.

    Args:
        tensor: integer torch.Tensor (dtype int32 or int64 preferred).
        p_error: probability of flipping each (unprotected) bit.
        protected_range: (low_bit, high_bit) inclusive bit index range [0..nbits-1]
                         that should use p_protected instead of p_error.
        p_protected: probability of flipping each bit inside protected_range.
        ignore_range: optional inclusive bit range (low, high) which must be left
                      exactly as the original input
        nbits: number of bits considered per integer (default 32).

    Returns:
        New tensor (same dtype & device) with bit flips injected.
    """
    nbits = 32
    device = tensor.device

    rand = torch.rand((*tensor.shape, nbits), device=device)
    bits = torch.zeros((*tensor.shape, nbits), dtype=torch.int32, device=device)
    idx = torch.arange(nbits, device=device)

    low_p, high_p = protected_range
    protected_mask = (idx >= low_p) & (idx <= high_p)
    bits[..., protected_mask] = (rand[..., protected_mask] < p_protected).int()

    bits[..., ~protected_mask] = (rand[..., ~protected_mask] < p_error).int()

    if ignore_range is not None:
        low_i, high_i = ignore_range
        ignore_mask = (idx >= low_i) & (idx <= high_i)
        bits[..., ignore_mask] = 0

    bit_values = (1 << idx).to(torch.int32)
    bitmask = (bits * bit_values).sum(dim=-1).to(torch.int32)
    return tensor ^ bitmask

####### seed ########## 

def set_seed(seed: int):
    """Sets the random seed for PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

####### error correction #########

def hamming_encode_32(x: torch.Tensor) -> torch.Tensor:
    """
    Encode each 32-bit integer in tensor x using Hamming(38,32) SECDED code.
    Adds 6 Hamming parity bits and 1 overall parity bit for double-error detection.

    Input:
        x: torch.Tensor of dtype torch.int32 or torch.int64
           Arbitrary shape.

    Output:
        torch.Tensor of dtype torch.int64 containing encoded 39-bit code words.
    """

    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError("Input tensor must have integer dtype (int32 or int64)")

    x = x.to(torch.int64)
    code = torch.zeros_like(x)

    parity_positions = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.int64)
    overall_pos = 39
    total_bits = 39

    all_positions = torch.arange(1, total_bits + 1, dtype=torch.int64)
    data_positions = [p for p in all_positions.tolist() if p not in parity_positions.tolist() and p != overall_pos]

    for bit_idx, pos in enumerate(data_positions):
        mask = (x >> bit_idx) & 1
        code |= mask << (pos - 1)

    for p in parity_positions.tolist():
        indices = ((all_positions & p) != 0)
        parity_mask = torch.zeros_like(code)
        for i in all_positions[indices]:
            parity_mask ^= (code >> (i - 1)) & 1
        code |= (parity_mask & 1) << (p - 1)

    # overall parity: XOR of all bits in code (manual bit count)
    overall_parity = torch.zeros_like(code)
    tmp = code.clone()
    for _ in range(total_bits - 1):  # iterate bits
        overall_parity ^= tmp & 1
        tmp >>= 1

    code |= (overall_parity & 1) << (overall_pos - 1)
    return code

def hamming_extract_parity(ecc: torch.Tensor) -> torch.Tensor:
    """
    Extract parity bits (positions 1,2,4,8,16,32,39) using a single bitmask.
    Returns the raw masked value (bits still in their original positions).
    """

    if ecc.dtype != torch.int64:
        raise TypeError("Input must be int64 tensor (encoded Hamming words)")

    mask = (
        (1 << 0)  |  # position 1
        (1 << 1)  |  # position 2
        (1 << 3)  |  # position 4
        (1 << 7)  |  # position 8
        (1 << 15) |  # position 16
        (1 << 31) |  # position 32
        (1 << 38)    # position 39
    )

    return ecc & mask

def hamming_apply_previous_parity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Replace parity bits in A with those from B.
    Parity positions: 1, 2, 4, 8, 16, 32, 39 (1-indexed)

    Input:
        a, b: torch.Tensor of dtype torch.int64 (same shape)

    Output:
        torch.Tensor of dtype torch.int64 with parity bits from B applied to A.
    """

    if a.dtype != torch.int64 or b.dtype != torch.int64:
        raise TypeError("Inputs must be int64 tensors")

    mask = (
        (1 << 0)  |  # bit 1
        (1 << 1)  |  # bit 2
        (1 << 3)  |  # bit 4
        (1 << 7)  |  # bit 8
        (1 << 15) |  # bit 16
        (1 << 31) |  # bit 32
        (1 << 38)    # bit 39
    )

    # Clear parity bits in A, insert parity bits from B
    return (a & ~mask) | (b & mask)

def hamming_secded_correct(x: torch.Tensor):
    
    x = x.to(torch.int64)
    # x = torch.tensor([0b100000000000000000000000000110111110101], dtype=torch.int64)
    # x = torch.tensor([0b100001000101000000000000000110111110101], dtype=torch.int64)

    # check bit 1 xors all bits that p1 checks for and p1
    # print("######## x: ")
    # print(format(x.item(), '039b'))
    # print("######## x & 1:")
    # print(format((x & 1).item(), '039b'))
    # print("######## x & 2**2:")
    # print(format(((x >> 2) & 1).item(), '039b'))
    # print("######## x & 2**5:")
    # print(format(((x >> 4) & 1).item(), '039b'))
    # print("######## x & 2**7:")
    # print(format(((x >> 6) & 1).item(), '039b'))
    # print("######## x & 2**9:")
    # print(format(((x >> 8) & 1).item(), '039b'))
    # print("######## x & 2**11:")
    # print(format(((x >> 10) & 1).item(), '039b'))
    # print("######## x & 2**13:")
    # print(format(((x >> 12) & 1).item(), '039b'))
    
    # print("########")
    c1 = ((x & 1) ^ ((x >> 2) & 1) ^ ((x >> 4) & 1) 
          ^ ((x >> 6) & 1) ^ ((x >> 8) & 1) ^ ((x >> 10) & 1)
          ^ ((x >> 12) & 1) ^ ((x >> 14) & 1) ^ ((x >> 16) & 1)
          ^ ((x >> 18) & 1) ^ ((x >> 20) & 1) ^ ((x >> 22) & 1)
          ^ ((x >> 24) & 1) ^ ((x >> 26) & 1) ^ ((x >> 28) & 1)
          ^ ((x >> 30) & 1) ^ ((x >> 32) & 1) ^ ((x >> 34) & 1)
          ^ ((x >> 36) & 1))
    
    # print("######## c1 ")
    # print(format(c1.item(), '039b'))
    # print("########")

    c2 = ((x >> 1) & 1) ^ ((x >> 2) & 1) ^ ((x >> 5) & 1) ^ ((x >> 6) & 1)      \
        ^ ((x >> 9) & 1) ^ ((x >> 10) & 1) ^ ((x >> 13) & 1) ^ ((x >> 14) & 1)  \
        ^ ((x >> 17) & 1) ^ ((x >> 18) & 1) ^ ((x >> 21) & 1) ^ ((x >> 22) & 1) \
        ^ ((x >> 25) & 1) ^ ((x >> 26) & 1) ^ ((x >> 29) & 1) ^ ((x >> 30) & 1) \
        ^ ((x >> 33) & 1) ^ ((x >> 34) & 1) ^ ((x >> 37) & 1)                   \
    
    # print("######## c2 ")
    # print(format(c2.item(), '039b'))
    # print("########")

    c3 = ((x >> 3) & 1) ^ ((x >> 4) & 1) ^ ((x >> 5) & 1) ^ ((x >> 6) & 1)      \
        ^ ((x >> 11) & 1) ^ ((x >> 12) & 1) ^ ((x >> 13) & 1) ^ ((x >> 14) & 1)  \
        ^ ((x >> 19) & 1) ^ ((x >> 20) & 1) ^ ((x >> 21) & 1) ^ ((x >> 22) & 1) \
        ^ ((x >> 27) & 1) ^ ((x >> 28) & 1) ^ ((x >> 29) & 1) ^ ((x >> 30) & 1) \
        ^ ((x >> 35) & 1) ^ ((x >> 36) & 1) ^ ((x >> 37) & 1)                   \

    #  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0 1 1 1 1 1 0 1 0 1
    # 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
    #  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0 1 1 1 1 1 0 1 0 1
    #                                                                                                       0 1 1 check bits

    # print("######## c3 ")
    # print(format(c3.item(), '039b'))
    # print("########")

    c4 = ((x >> 7) & 1) ^ ((x >> 8) & 1) ^ ((x >> 9) & 1) ^ ((x >> 10) & 1)      \
        ^ ((x >> 11) & 1) ^ ((x >> 12) & 1) ^ ((x >> 13) & 1) ^ ((x >> 14) & 1)  \
        ^ ((x >> 23) & 1) ^ ((x >> 24) & 1) ^ ((x >> 25) & 1) ^ ((x >> 26) & 1) \
        ^ ((x >> 27) & 1) ^ ((x >> 28) & 1) ^ ((x >> 29) & 1) ^ ((x >> 30) & 1) \
        
    # print("######## c4 ")
    # print(format(c4.item(), '039b'))
    # print("########")

    c5 = ((x >> 15) & 1) ^ ((x >> 16) & 1) ^ ((x >> 17) & 1) ^ ((x >> 18) & 1)      \
        ^ ((x >> 19) & 1) ^ ((x >> 20) & 1) ^ ((x >> 21) & 1) ^ ((x >> 22) & 1)  \
        ^ ((x >> 23) & 1) ^ ((x >> 24) & 1) ^ ((x >> 25) & 1) ^ ((x >> 26) & 1) \
        ^ ((x >> 27) & 1) ^ ((x >> 28) & 1) ^ ((x >> 29) & 1) ^ ((x >> 30) & 1) \
        
    # print("######## c5 ")
    # print(format(c5.item(), '039b'))
    # print("########")

    c6 = ((x >> 31) & 1) ^ ((x >> 32) & 1) ^ ((x >> 33) & 1) ^ ((x >> 34) & 1)      \
        ^ ((x >> 35) & 1) ^ ((x >> 36) & 1) ^ ((x >> 37) & 1)
        
    # print("######## c6 ")
    # print(format(c6.item(), '039b'))
    # print("########")

     # overall parity bit (p7 = bit 39)
    overall_parity = ((x >> 38) & 1)
    computed_parity = (x & 1) ^ ((x >> 1) & 1) ^ ((x >> 2) & 1) ^ ((x >> 3) & 1) ^ ((x >> 4) & 1) \
    ^ ((x >> 5) & 1) ^ ((x >> 6) & 1) ^ ((x >> 7) & 1) ^ ((x >> 8) & 1) ^ ((x >> 9) & 1) \
    ^ ((x >> 10) & 1) ^ ((x >> 11) & 1) ^ ((x >> 12) & 1) ^ ((x >> 13) & 1) ^ ((x >> 14) & 1) \
    ^ ((x >> 15) & 1) ^ ((x >> 16) & 1) ^ ((x >> 17) & 1) ^ ((x >> 18) & 1) ^ ((x >> 19) & 1) \
    ^ ((x >> 20) & 1) ^ ((x >> 21) & 1) ^ ((x >> 22) & 1) ^ ((x >> 23) & 1) ^ ((x >> 24) & 1) \
    ^ ((x >> 25) & 1) ^ ((x >> 26) & 1) ^ ((x >> 27) & 1) ^ ((x >> 28) & 1) ^ ((x >> 29) & 1) \
    ^ ((x >> 30) & 1) ^ ((x >> 31) & 1) ^ ((x >> 32) & 1) ^ ((x >> 33) & 1) ^ ((x >> 34) & 1) \
    ^ ((x >> 35) & 1) ^ ((x >> 36) & 1) ^ ((x >> 37) & 1)
                                                                                                                                                                                                                                                                                                                                                                                                                   

    syndrome = (c6 << 5) | (c5 << 4) | (c4 << 3) | (c3 << 2) | (c2 << 1) | c1

    # print("syndrome: ", syndrome[0][0][0][4].item())
    # print("overall_parity:", overall_parity[0][0][0][4].item(), "computed_parity:", computed_parity[0][0][0][4].item())

    corrected_x = x.clone()
    # error_type = "none"

    single_error_mask = (syndrome != 0) & (overall_parity != computed_parity)
    # no_error_mask = (syndrome == 0) & (overall_parity == computed_parity)
    # double_error_mask = (syndrome != 0) & (overall_parity == computed_parity)
    # triple_or_parity_error_mask = (syndrome == 0) & (overall_parity != computed_parity)

    # Initialize
    corrected_x = x.clone()
    # error_type = torch.full(x.shape, -1, dtype=torch.int8)  # or any encoding for error type

    # Single-bit correction
    bit_index = syndrome - 1
    corrected_x[single_error_mask] ^= (1 << bit_index[single_error_mask])

    # Optionally encode error types numerically:
    # 0 = no error, 1 = single corrected, 2 = double detected, 3 = triple/parity
    # error_type[no_error_mask] = 0
    # error_type[single_error_mask] = 1
    # error_type[double_error_mask] = 2
    # error_type[triple_or_parity_error_mask] = 3
    # the rest of the triple bit errors will be missed

    # print("######## x ########")
    # print(format(x[0][0][0][4].item(), '039b'))
    # print("######## corrected_x ########")
    # print(format(corrected_x[0][0][0][4].item(), '039b'))
    # print("Error type:", error_type)

    return corrected_x

def hamming_extract_data(codewords: torch.Tensor) -> torch.Tensor:
    """
    Extracts the 32 data bits from each 39-bit Hamming(39,32) SECDED codeword tensor element.
    Input:
        codewords: torch.Tensor of dtype int64/int32, each representing a 39-bit codeword.
    Output:
        torch.Tensor of same shape containing only the 32-bit data.
    """
    parity_positions = torch.tensor([1, 2, 4, 8, 16, 32, 39], dtype=torch.int64)
    data_positions = [i for i in range(1, 40) if i not in parity_positions.tolist()]

    data = torch.zeros_like(codewords)
    for out_bit, in_bit in enumerate(data_positions):
        bit_val = (codewords >> (in_bit - 1)) & 1
        data |= bit_val << out_bit
    return data

def test8(pt_name="lenet5_sparse_quantized.pt", p_error=1e-2, p_protected=1e-8, protected_range=(23, 31), ignore_range=(8,31),
          test_data = None, test_label = None):
    net = LeNet5(8)
    net = net.to(device)
    dict = torch.load(pt_name, map_location=torch.device('cpu'))
    for key in ("conv1.quan_weight", "conv2.quan_weight", "conv3.quan_weight"):
        p = dict[key].int()

        # print(format(torch.max(p).item(), '039b'))
        ecc = hamming_encode_32(p)
        # print(format(ecc[0][0][0][4].item(), '039b'))
        parity = hamming_extract_parity(ecc)
        # print(format(parity[0][0][0][4].item(), '039b'))

        p = inject_bit_errors_int(p, p_error=p_error, protected_range=protected_range, p_protected=p_protected, ignore_range=ignore_range)
        # print(p[0][0][0][4])
        ecc = hamming_encode_32(p)
        # print(format(ecc[0][0][0][4].item(), '039b'))
        
        ecc = hamming_apply_previous_parity(ecc, parity)
        # print(format(ecc[0][0][0][4].item(), '039b'))
        
        ecc =  hamming_secded_correct(ecc)
        # print("corrected:")
        # , double_error =
        # print(format(ecc[0][0][0][0][4].item(), '039b'))
        # print(double_error[0][0][0][4])

        p = hamming_extract_data(ecc)
        # print(p[0][0][0][4])

        # return

        dict[key] = p

    for key in ("fc1.quan_weight", "fc2.quan_weight"):
        p = dict[key].int()

        ecc = hamming_encode_32(p)
        
        parity = hamming_extract_parity(ecc)

        p = inject_bit_errors_int(p, p_error=p_error, protected_range=protected_range, p_protected=p_protected, ignore_range=ignore_range)

        ecc = hamming_encode_32(p)

        ecc = hamming_apply_previous_parity(ecc, parity)

        ecc =  hamming_secded_correct(ecc)

        p = hamming_extract_data(ecc)

        dict[key] = p
    
    """
                                                                                                1       0   0 1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 0 1 0 0 0 0 0 0 1
    39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 0 1 1 0 1 0 0 0 1
                                                                                         1  0 0   1 0 1   0   
    
    39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 1 0 1 1 0 0 1 0 1
                                                                                            1 1 0 1 1 0 0 1 0 1
     
     
    39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0 1 1 1 1 1 0 1 0 1
                                                                                                1       0   0 1  
    
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0 1 0 1 1 1 0 1 1 0
    39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0 1 1 1 1 1 0 1 0 1
                                                                                                            1 1 check bits

    parity:
    o 32 16 8 4 2 1
    1  0  0 0 0 0 0

    """
    
    net.load_state_dict(dict, strict=False)
    net.quan_mode()
    return acc(net, test_data, test_label)

if __name__ == '__main__':
    set_seed(69)
    train_data, train_label, test_data, test_label = load_data_set()
    # # main_work_flow()

    # print(format(torch.tensor([int(254)], dtype=torch.int32).item(), '032b'))
    # print(format(inject_bit_errors_int(torch.tensor([int(254)], dtype=torch.int32),
    #                                    p_error=0.5, protected_range=(0, 7), p_protected=0,
    #                                    ignore_range=(16,31)).to(torch.uint32).item(), '032b'))

    results = {
        "8bit_sparse_diff": [],
        "8bit_dense_diff": [],
        "16bit_dense_diff": [],
        "4bit_dense_diff": [],
        "8bit_sparse_same": [],
        "8bit_dense_same": [],
        "16bit_dense_same": [],
        "4bit_dense_same": []
    }

    error_probs_diff = [0.00001, 0.0001,
                        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    error_probs_same = [0.00001, 0.0001,
                        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]

    ############################################
    ############# different errors #############
    ############################################
    for p_error in error_probs_diff:
        print(f"error prob: {p_error}")

        results = {"8bit_sparse": [], "8bit_dense": [], "16bit_dense": [], "4bit_dense": []}
        x_labels = {"8bit_sparse": [], "8bit_dense": [], "16bit_dense": [], "4bit_dense": []}

        for i in range(1, 8):
            accuracy = test8(
                pt_name="lenet5_sparse_quantized.pt", p_error=p_error, p_protected=1e-8,
                protected_range=(0, i), ignore_range=(8,31),
                test_data=test_data, test_label=test_label
            )
            results["8bit_sparse"].append(accuracy)
            x_labels["8bit_sparse"].append(i)

        for i in range(1, 8):
            accuracy = test8(
                pt_name="lenet5_dense_quantized.pt", p_error=p_error, p_protected=1e-8,
                protected_range=(0, i), ignore_range=(8,31),
                test_data=test_data, test_label=test_label
            )
            results["8bit_dense"].append(accuracy)
            x_labels["8bit_dense"].append(i)

        for i in range(1, 16):
            accuracy = test8(
                pt_name="lenet5_dense_quantized_16.pt", p_error=p_error, p_protected=1e-8,
                protected_range=(0, i), ignore_range=(16,31),
                test_data=test_data, test_label=test_label
            )
            results["16bit_dense"].append(accuracy)
            x_labels["16bit_dense"].append(i)

        for i in range(1, 4):
            accuracy = test8(
                pt_name="lenet5_dense_quantized_4.pt", p_error=p_error, p_protected=1e-8,
                protected_range=(0, i), ignore_range=(4,31),
                test_data=test_data, test_label=test_label
            )
            results["4bit_dense"].append(accuracy)
            x_labels["4bit_dense"].append(i)

        for key in results.keys():
            plt.figure()
            plt.plot(x_labels[key], results[key], marker='o')
            plt.title(f"{key} | different errors | p_error={p_error}")
            plt.xlabel("Protected bits")
            plt.ylabel("Accuracy")
            plt.grid(True)
            
            plt.savefig(f"accuracy_plots/{key}_different_errors_{p_error}.png")
            plt.close()

    ############################################
    ############### same error #################
    ############################################
    for p_error in error_probs_same:
        print(f"error prob: {p_error}")

        configs = [
            ("8bit_sparse", "lenet5_sparse_quantized.pt", (0,1), (8,31)),
            ("8bit_dense", "lenet5_dense_quantized.pt", (0,1), (8,31)),
            ("16bit_dense", "lenet5_dense_quantized_16.pt", (0,1), (16,31)),
            ("4bit_dense", "lenet5_dense_quantized_4.pt", (0,1), (4,31))
        ]

        for name, pt, prange, irange in configs:
            accuracy = test8(
                pt_name=pt, p_error=p_error, p_protected=p_error,
                protected_range=prange, ignore_range=irange,
                test_data=test_data, test_label=test_label
            )

            plt.figure()
            plt.plot([p_error], [accuracy], 'ro')
            plt.title(f"{name} | same error | p_error=p_protected={p_error}")
            plt.xlabel("Error Probability")
            plt.ylabel("Accuracy")
            plt.grid(True)
            
            plt.savefig(f"accuracy_plots/{name}_same_error_{p_error}.png")
            plt.close()
