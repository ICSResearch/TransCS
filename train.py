import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
import scipy.io as scio
import torch.optim as optim
import torchsummary.torchsummary
import torch.optim.lr_scheduler as LS

import models
import utils

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--bs", default=64, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
opt = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_p(config, net):
    batch_size = 1

    net = net.eval()
    file_no = [11]
    folder_name = ["SET11"]

    for idx, item in enumerate(folder_name):
        p_total = 0
        path = "{}/".format(config.val_path) + item
        with torch.no_grad():
            for i in range(file_no[idx]):
                name = "{}/({}).mat".format(path, i + 1)
                x = scio.loadmat(name)['temp3']
                x = torch.from_numpy(np.array(x)).to(config.device)
                x = x.float()
                ori_x = x

                h = x.size()[0]
                h_lack = 0
                w = x.size()[1]
                w_lack = 0

                if h % config.block_size != 0:
                    h_lack = config.block_size - h % config.block_size
                    temp_h = torch.zeros(h_lack, w).to(config.device)
                    h = h + h_lack
                    x = torch.cat((x, temp_h), 0)

                if w % config.block_size != 0:
                    w_lack = config.block_size - w % config.block_size
                    temp_w = torch.zeros(h, w_lack).to(config.device)
                    w = w + w_lack
                    x = torch.cat((x, temp_w), 1)

                x = torch.unsqueeze(x, 0)
                x = torch.unsqueeze(x, 0)

                idx_h = range(0, h, config.block_size)
                idx_w = range(0, w, config.block_size)
                num_patches = h * w // (config.block_size ** 2)

                temp = torch.zeros(num_patches, batch_size, config.channel, config.block_size, config.block_size)
                count = 0
                for a in idx_h:
                    for b in idx_w:
                        output = net(x[:, :, a:a + config.block_size, b:b + config.block_size])
                        temp[count, :, :, :, :, ] = output
                        count = count + 1

                y = torch.zeros(batch_size, config.channel, h, w)
                count = 0
                for a in idx_h:
                    for b in idx_w:
                        y[:, :, a:a + config.block_size, b:b + config.block_size] = temp[count, :, :, :, :]
                        count = count + 1

                recon_x = y[:, :, 0:h - h_lack, 0:w - w_lack]

                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = ori_x.to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                p = 10 * np.log10(1 / mse)
                p_total = p_total + p

            return p_total / file_no[idx]


def main():
    device = "cuda:" + opt.device
    config = utils.GetConfig(ratio=opt.rate, device=device)
    config.check()
    set_seed(22)
    print("Data loading...")
    torch.cuda.empty_cache()
    dataset_train = utils.train_loader(batch_size=opt.bs)
    net = models.HybridNet(config).to(config.device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=10e-4)

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.model, map_location=config.device))
            info = torch.load(config.info, map_location=config.device)
        else:
            net.load_state_dict(torch.load(config.model, map_location="cpu"))
            info = torch.load(config.info, map_location="cpu")

        start_epoch = info["epoch"]
        best = info["res"]
        print("Loaded trained model of epoch {:2}, res: {:8.4f}.".format(start_epoch, best))
    else:
        start_epoch = 1
        best = 0
        print("No saved model, start epoch = 1.")

    scheduler = LS.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    over_all_time = time.time()
    for epoch in range(start_epoch, 200):
        print("Lr: op {}; sc {}.".format(optimizer.param_groups[0]['lr'], scheduler.get_last_lr()))

        epoch_loss = 0
        dic = {"rate": config.ratio, "epoch": epoch,
               "device": config.device}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            xi = xi.to(config.device)
            optimizer.zero_grad()
            xo = net(xi)
            batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.device)
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]"
                           .format(config.batch_size * (idx + 1),
                                   dataset_train.__len__() * config.batch_size,
                                   batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]"
              .format(epoch, avg_loss))

        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        print("\rNow val..")
        p = val_p(config, net)
        print("{:5.3f}".format(p))
        if p > best:
            info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.model)
            torch.save(optimizer.state_dict(), config.optimizer)
            torch.save(info, config.info)
            print("*", "  Check point of epoch {:2} saved  ".format(epoch).center(120, "="), "*")
            best = p
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write("Epoch {:2.0f}, Loss of train {:8.10f}, Res {:2.3f}\n".format(epoch, avg_loss, best))
            output_file.write(old)
            output_file.close()

        scheduler.step()
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))
    print("Train end.")


def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(opt.device) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    main()
