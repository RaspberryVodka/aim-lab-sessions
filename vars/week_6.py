# Week 3 imports
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
from PIL import Image

# Week 4 imports
import torch.nn as nn
from torch.optim import SGD
from torchsummary import summary

# Week 5 imports
from torch.optim.lr_scheduler import ExponentialLR
# Week 6 imports 
import os
from torch.utils.tensorboard import SummaryWriter


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inp):
        shortcut = self.shortcut(inp)
        inp = nn.ReLU()(self.bn1(self.conv1(inp)))
        inp = nn.ReLU()(self.bn2(self.conv2(inp)))
        inp = inp + shortcut  # The magic bit that cannot be done with nn.Sequential!
        return nn.ReLU()(inp)
    


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, outputs):
        super().__init__()
        self.layer0_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.layer0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0_bn   = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU()

        self.layer1_res1 = resblock(64, 64, downsample=False)
        self.layer1_res2 = resblock(64, 64, downsample=False)

        self.layer2_res1 = resblock(64, 128, downsample=True)
        self.layer2_res2 = resblock(128, 128, downsample=False)

        self.layer3_res1 = resblock(128, 256, downsample=True)
        self.layer3_res2 = resblock(256, 256, downsample=False)

        self.layer4_res1 = resblock(256, 512, downsample=True)
        self.layer4_res2 = resblock(512, 512, downsample=False)

        self.gap         = nn.AdaptiveAvgPool2d(1)
        self.flat        = nn.Flatten() 
        self.fc          = nn.Linear(512, outputs)

    def forward(self, inp):
        inp = self.layer0_conv(inp)
        inp = self.layer0_pool(inp)
        inp = self.layer0_bn(inp)
        inp = self.layer0_relu(inp)
        
        inp = self.layer1_res1(inp)
        inp = self.layer1_res2(inp)
        
        inp = self.layer2_res1(inp)
        inp = self.layer2_res2(inp)
        
        inp = self.layer3_res1(inp)
        inp = self.layer3_res2(inp)
        
        inp = self.layer4_res1(inp)
        inp = self.layer4_res2(inp)
            
        inp = self.gap(inp)
        inp = self.flat(inp)
        inp = self.fc(inp)

        return inp
    

# convenience function
def get_resnet():
    return ResNet(1, ResBlock, outputs=10)



class EarlyStopper:
    def __init__(self, patience=1, tolerance=0):
        self.patience = patience           # How many epochs in a row the model is allowed to underperform    
        self.tolerance = tolerance         # How much leeway the model has (i.e. how close it can get to underperforming before it is counted as such)
        self.epoch_counter = 0             # Keeping track of how many epochs in a row were failed 
        self.max_validation_acc = np.NINF  # Keeping track of best metric so far

    def should_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.epoch_counter = 0
        elif validation_acc < (self.max_validation_acc - self.tolerance):
            self.epoch_counter += 1
            if self.epoch_counter >= self.patience:
                return True
        return False 
    

# Saves a model to file, and names it after the current epoch
def save_checkpoint(model, epoch, save_dir):
    filename = f"checkpoint_{epoch}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)
    
    
def train_model_final(device, model, epochs, dataloaders, optimiser, lr_scheduler, writer, early_stopper, checkpoint_frequency):
    msg = ""
    for epoch in range(epochs):        
        #######################TRAINING STEP###################################
        model.train()  # set model to training mode 
        train_dl = dataloaders['train'] # select train dataloader
        
        total_steps_train = len(train_dl)
        correct_train = 0
        total_train = 0
        loss_train = 0
        
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(image_batch)
            label_batch = label_batch.to(device)
            image_batch = image_batch.to(device).reshape(batch_sz, 1, 28, 28) 
            output = model(image_batch)
            loss_train = nn.CrossEntropyLoss()(output, label_batch)
                        
            optimiser.zero_grad()
            loss_train.backward()
            optimiser.step()
            
            preds_train = torch.argmax(output, dim=1)
            correct_train += int(torch.eq(preds_train, label_batch).sum())
            total_train += batch_sz
            minibatch_accuracy_train = 100 * correct_train / total_train
            
            #### Fancy printing stuff, you can ignore this! ######
            if (batch_num + 1) % 5 == 0:
                print(" " * len(msg), end='\r')
                msg = f'Train epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_train}], Loss: {loss_train.item():.5f}, Acc: {minibatch_accuracy_train:.5f}, LR: {lr_scheduler.get_last_lr()[0]:.5f}'
                print (msg, end='\r' if epoch < epochs else "\n",flush=True)
            #### Fancy printing stuff, you can ignore this! ######
        lr_scheduler.step()
        ########################################################################
        print("") # Create newline between progress bars
        #######################VALIDATION STEP##################################
        model.eval()  # set model to evaluation mode. This is very important, we do not want to update model weights in eval mode
        val_dl = dataloaders['val'] # select val dataloader
        
        total_steps_val = len(val_dl)
        correct_val = 0
        total_val = 0
        loss_val = 0
        
        for batch_num, (image_batch, label_batch) in enumerate(val_dl):
            batch_sz = len(image_batch)
            label_batch = label_batch.to(device)
            image_batch = image_batch.to(device).reshape(batch_sz, 1, 28, 28) 
            
            with torch.no_grad(): # no_grad disables gradient calculations, which are not needed when evaluating the model. This speeds up the calculations
                output = model(image_batch)
                loss_val = nn.CrossEntropyLoss()(output, label_batch)

                preds_val = torch.argmax(output, dim=1)
                correct_val += int(torch.eq(preds_val, label_batch).sum())
                total_val += batch_sz
                minibatch_accuracy_val = 100 * correct_val / total_val
                
                #### Fancy printing stuff, you can ignore this! ######
                if (batch_num + 1) % 5 == 0:
                    print(" " * len(msg), end='\r')
                    msg = f'Eval epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_val}], Loss: {loss_val.item():.5f}, Acc: {minibatch_accuracy_val:.5f}'
                    if early_stopper.epoch_counter > 0:
                        msg += f", Epochs without improvement: {early_stopper.epoch_counter}"
                    print (msg, end='\r' if epoch < epochs else "\n",flush=True)
                #### Fancy printing stuff, you can ignore this! ######
        ########################################################################
        print("")  # Create newline between progress bars
        
        # Log loss and accuracy metrics using the writer so we can see them in Tensorboard 
        epoch_train_acc = 100 * correct_train / total_train
        epoch_val_acc = 100 * correct_val / total_val
        
        writer.add_scalar(f'Loss/train', loss_train, epoch)
        writer.add_scalar(f'Acc/train', epoch_train_acc, epoch)
        writer.add_scalar(f'Loss/val', loss_val, epoch)
        writer.add_scalar(f'Acc/val', epoch_val_acc, epoch)
        
        # Check whether we need to save the model to a checkpoint file
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(model, epoch + 1, "./saved_models")

        # Check whether we should stop the training based on the validation accuracy
        if early_stopper.should_stop(epoch_val_acc):
            print(f"\nValidation accuracy has not improved for the last {early_stopper.epoch_counter} epochs, stopping training early at epoch {epoch + 1}/{epochs}")
            # if stopping, we also want to save the checkpoint so we don't lose anything between the last save
            save_checkpoint(model, epoch + 1, "./saved_models")
            return
        
def test_model(device, model, dataloaders):
    model.eval()
    correct = 0
    total = 0
    
    test_dl = dataloaders['test']
    total_steps = len(test_dl)
    msg = ""
    for batch_num, (image_batch, label_batch) in enumerate(test_dl):
        batch_sz = len(image_batch)
        label_batch = label_batch.to(device)
        image_batch = image_batch.to(device).reshape(batch_sz, 1, 28, 28)
        out = model(image_batch)
        preds = torch.argmax(out, dim=1)
        correct += int(torch.eq(preds, label_batch).sum())
        total += label_batch.shape[0]
        if (batch_num + 1) % 5 == 0:
            print(" " * len(msg), end='\r')
            msg = f'Testing batch[{batch_num + 1}/{total_steps}]'
            print (msg, end='\r' if batch_num < total_steps else "\n", flush=True)
    print(f"\nFinal test accuracy for {total} examples: {100 * correct/total:.5f}")

    
    
def load_from_file(filepath):
    img = (
            Image.open(filepath)   # load image
                .convert('L')      # convert from RGBA to grayscale
                .resize((28, 28), resample=PIL.Image.Resampling.BICUBIC)  # resize to what our network expects
                
          )
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = torch.where((img_tensor <= 120), 0, 255)
    return img_tensor.type(torch.FloatTensor)


def live_test_images(filepaths, model, device):
    batch_sz = len(filepaths)
    batch = torch.FloatTensor(batch_sz, 28, 28) 
    torch.cat([load_from_file(f) for f in filepaths], out=batch)
    batch = batch.reshape(batch_sz, 1, 28, 28).to(device) # Only item in batch, 1 channel, 28 * 28 pixels
    out = model(batch)
    preds = torch.argmax(out, dim=1)
    for i, p in enumerate(preds):
        show_img(filepaths[i], p.item())

def show_img(path, prediction):
    img = load_from_file(path)
    fig = plt.figure(figsize=(1., 1.))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False
    )
    fig.suptitle(f"Prediction for image: {prediction}", y=0)
