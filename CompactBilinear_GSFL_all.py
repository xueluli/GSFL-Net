
from __future__ import division
import os

import torch
import torchvision

#import cub200
import pdb
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from CompactBilinearPooling1 import CompactBilinearPooling

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 200
input_dim = 512
input_dim1 = 512
input_dim2 = 512
output_dim = 8192
alpha5 = 0.01 # dis_center_change_reg
alpha6 = 0.01 # total_dis_center_change_regs
rho = 0


cluster_lbs = torch.load('/media/tiantong/Drive/Xuelu/CUB_200_2011/cluster_label.pth')
#generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
#sketch_matrix01 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim))
#sketch_matrix02 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim))
#torch.save(sketch_matrix01,'/cvdata/xuelu/CUB_200_2011/bilinear-cnn1/src/sketch_matrix1_16384.pth')
#torch.save(sketch_matrix02,'/cvdata/xuelu/CUB_200_2011/bilinear-cnn1/src/sketch_matrix2_16384.pth')

sketch_matrix01 = torch.load('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/sketch_matrix1.pth')
sketch_matrix02 = torch.load('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/sketch_matrix2.pth')

class BCNN(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(output_dim, 4096),
#            nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True))
#        discriminative_encoder
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(output_dim, 4096),
#            nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True))
#        decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Linear(4096, output_dim),
            torch.nn.Sigmoid())
        # Linear classifier.
        self.fc1 = torch.nn.Linear(4096, num_classes)
#        for param in self.features.parameters():
#            param.requires_grad = False

    def forward(self, X):
#        pdb.set_trace()
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        sketch_matrix1 = sketch_matrix01.cuda()
        sketch_matrix2 = sketch_matrix02.cuda()
        fft1 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix1), 1)
        fft2 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, 1, signal_sizes = (output_dim,)) * output_dim
#        pdb.set_trace()
#        XX = X.permute(0, 2, 3, 1).contiguous().view(-1, 512)
        
#        X = torch.matmul(X,X.transpose(1,2))/(28**2)  # Bilinear
#        assert X.size() == (N, 512, 512)
#        X = X.view(N, 512**2)
        X = cbp.sum(dim = 1).sum(dim = 1)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X))
        out1 = torch.nn.functional.normalize(X)
#        pdb.set_trace()
        
        x1 = self.encoder1(out1)
#        pdb.set_trace()
        x2 = self.encoder2(out1)
#        pdb.set_trace()
        y = self.decoder(x1+x2)        
        out = self.fc1(x2)        
        assert out.size() == (N, 200)
        return out, y, x1, x2, out1 
    
def mse_loss(input, target):
    return torch.sum((input-target)**2)/input.data.nelement()

def mycustomLoss(data, shared_decoderOp, shared_encoder, dis_encoder, labels, class_center, shared_center, out):
    alpha1 = 0.01
    alpha2 = 0.1
    alpha3 = 0.1
#  pdb.set_trace()
    aa = mse_loss(shared_decoderOp, data)
    bb = torch.nn.MSELoss()(dis_encoder[0, :], class_center[labels[0], :])
    cc = torch.nn.MSELoss()(shared_encoder[0, :], shared_center[labels[0], :])  
    dim = data.shape[0] 
    for i in range(1, dim):
        bb = bb+torch.nn.MSELoss()(dis_encoder[i, :], class_center[labels[i], :])
        cc = cc+torch.nn.MSELoss()(shared_encoder[i, :], shared_center[labels[0], :])     
    dd = torch.nn.CrossEntropyLoss()(out, labels)
    Loss = dd+alpha1*aa+alpha2*bb+alpha3*cc
#    pdb.set_trace()
    return Loss, aa, bb, cc, dd

class BCNNManager(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
#        pdb.set_trace()
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        # Load the model from disk.
#        self._net.load_state_dict(self.load_my_state_dict(torch.load(self._path['model'])))
        self._net.load_state_dict(torch.load(self._path['model']))
        print(self._net)
        # Criterion.
#        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
#        self._solver = torch.optim.SGD(filter(lambda p: p.requires_grad,self._net.parameters()), lr=self._options['base_lr'],
#            momentum=0.9, weight_decay=self._options['weight_decay'])
#        self._solver = torch.optim.SGD(self._net.parameters(), lr=self._options['base_lr'],
#            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._solver = torch.optim.Adam(filter(lambda p: p.requires_grad,self._net.parameters()), lr=self._options['base_lr'],
                                        weight_decay=self._options['weight_decay'])
#        self._solver = torch.optim.Adam(self._net.parameters(), lr=self._options['base_lr'],
#                                        weight_decay=self._options['weight_decay'])
#        self._solver = torch.optim.SGD([
#                                        {'params': self._net.module.features.parameters()},
#                                        {'params': self._net.module.encoder1.parameters(), 'lr': 0.05},
#                                        {'params': self._net.module.encoder2.parameters(), 'lr': 0.05},
#                                        {'params': self._net.module.decoder.parameters(), 'lr': 0.05},
#                                        {'params': self._net.module.fc1.parameters()}                                         
#                                        ], lr=self._options['base_lr'], momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.65, patience=5, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        data_dir1 = '/media/tiantong/Drive/Xuelu/CUB_200_2011/train'
        train_data = datasets.ImageFolder(root=data_dir1,transform=train_transforms)
        data_dir2 = '/media/tiantong/Drive/Xuelu/CUB_200_2011/test'
        test_data = datasets.ImageFolder(root=data_dir2,transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=8,
            shuffle=False, num_workers=4, pin_memory=True)
            
    def load_my_state_dict(self, state_dict):
        pretrained_dict = state_dict
        model_dict = self._net.state_dict()

        # 1. filter out unnecessary keys 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
#        pdb.set_trace()
        # 3. load the new state dict
        return model_dict
    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        class_center = torch.load('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/model/class_center.pth')
#        class_center = torch.zeros([num_classes, 4096], dtype=torch.float32).cuda()
        total_class_center = class_center
#        total_class_center = torch.zeros([num_classes, 4096], dtype=torch.float32).cuda()
#        shared_center = torch.zeros([4096], dtype=torch.float32).cuda()
        shared_center = torch.load('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/model/shared_center.pth')
#        shared_center = torch.zeros([num_classes, 4096], dtype=torch.float32).cuda()
        cluster_center = torch.zeros([5, 4096], dtype=torch.float32).cuda()
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            u = 0
            for X, labels in self._train_loader:
                u = u+1
#                if u == 375:
#                   pdb.set_trace()
                # Data.
                X = torch.autograd.Variable(X.cuda())
                labels = torch.autograd.Variable(labels.cuda(async=True))
                
                class_center = class_center.detach()
                total_class_center = total_class_center.detach()
                shared_center = shared_center.detach()
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                out, y, x1, x2, out1 = self._net(X)
#                pdb.set_trace()
                total_loss, aa, bb, cc, dd = mycustomLoss(out1, y, x1, x2, labels, class_center, shared_center, out)    
#                print('| Epoch %2d Iter %3d\tBatch loss %.4f\t' % (t+1,u,total_loss))
                epoch_loss.append(total_loss.item())
                # Prediction.
                
                _, prediction = torch.max(out.data, 1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels.data).item()
                # Backward pass.
                total_loss.backward()
                self._solver.step()
            
                for c in range(num_classes):
                    temp1 = torch.zeros([4096], dtype=torch.float32).cuda()  
                    temp1 = temp1.detach()
                    Lbv = torch.zeros([1,1], dtype=torch.float32).cuda()
                    dim = x2.shape[0]
                    for i in range(dim):
                        if labels[i] == c:
                           temp1 = temp1.add_(1*(class_center[c, :]-x2[i, :]))
#                       temp1 = temp1.add_(1*(class_center[c, :]-y[i, :]))
                           Lbv = Lbv + 1
                    delta_c_center = temp1/(1+Lbv)
                    class_center[c, :] = class_center[c, :]-alpha5*delta_c_center


                for c in range(num_classes):
                    temp2 = torch.zeros([4096], dtype=torch.float32).cuda() 
                    temp2 = temp2.detach()
                    Lbv = torch.zeros([1,1], dtype=torch.float32).cuda()
                    dim = y.shape[0]
                    for i in range(dim):
                        if labels[i] == c:
#                  temp1 = temp1.add_(1*(class_center[c, :]-x2[i, :]-x1[i, :]))
                           temp2 = temp2.add_(1*(total_class_center[c, :]-x2[i, :]))
                           Lbv = Lbv + 1

                    total_delta_c_center = temp2/(1+Lbv)
                    total_class_center[c, :] = total_class_center[c, :]-alpha6*total_delta_c_center
        
                for i in range(5):
                    temp3 = torch.zeros([4096], dtype=torch.float32).cuda()
                    temp3 = temp3.detach()
                    u1 = 0
                    for c in range(0,num_classes):    
                        if cluster_lbs[c] == i:
                           u1 = u1+1
                           temp3 = temp3.add_(total_class_center[c, :]).detach()
                    cluster_center[i, :] = temp3/u1
                for c in range(num_classes):
                    shared_center[c, :] = cluster_center[cluster_lbs[c], :] 
#                pdb.set_trace()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
#                print('*', end='')
# Save model onto disk.
                torch.save(self._net.state_dict(),
                           os.path.join('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/model',
                                        'Nd8192_DisShared_ft_vgg_16_epoch_%d.pth' % (t + 1)))
                                        
                torch.save(class_center,
                           os.path.join('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/model',
                                        'Nclass_center.pth'))
                torch.save(shared_center,
                           os.path.join('/media/tiantong/Drive/Xuelu/CUB_200_2011/bilinear-cnn1/model',
                                        'Nshared_center.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        torch.no_grad()
        num_correct = 0
        num_total = 0
        for X, labels in data_loader:
            # Data.

            X = torch.autograd.Variable(X.cuda())
            labels = torch.autograd.Variable(labels.cuda(async=True))
#            X = X.to(device)
#            y = y.to(device)

            # Prediction.
#            pdb.set_trace()
            out, y, x1, x2, out1 = self._net(X)
            _, prediction = torch.max(out.data, 1)
            num_total += labels.size(0)
            num_correct += torch.sum(prediction == labels.data).item()
#            del X, y
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total

def main():
    """The main function."""

    options = {
        'base_lr': 0.0001,
        'batch_size': 24,
        'epochs': 200,
        'weight_decay': 1e-5,
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'model': os.path.join(project_root, 'model', 'initiald8192_DisShared_vgg_16_epoch_60.pth'),
    }
    for d in path:
        if d == 'model':
            assert os.path.isfile(path[d])
#    pdb.set_trace()
#        else:
#            assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()

