import torch
import torchvision
import torch.nn as nn 
import torchvision.transforms as transforms 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from torchvision.datasets import ImageFolder 
import tqdm
import timm #pretrained models


#dataset class
class WasteDataset(Dataset): 
    def __init__(self, data_root, transform=None):
        self.data= ImageFolder(data_root, transform = transform)
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index): 
        return self.data[index]
    
    def classes(self): 
        return self.data.classes
    
if __name__ == '__main__':
    #data paths
    data_root=r'C:\Users\anjan\OneDrive\Documents\dataset' 
    test_root=r'C:\Users\anjan\OneDrive\Documents\testSet' 
    val_root=r'C:\Users\anjan\OneDrive\Documents\validationSet' 

    #mean and standard values of dataset
    mean = [0.7422, 0.7366, 0.7117] 
    std = [0.3060, 0.3009, 0.3302]

    #image transformations and normalizing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])
    
    #datasets for training, validation, and testing. 
    dataset= WasteDataset(data_root, transform)
    valset = WasteDataset(val_root, transform)
    testset = WasteDataset(test_root, transform)

    ''' code used to calculate mean and std values for a dataset
    loader = DataLoader(datasetTransformed, batch_size=len(datasetTransformed), num_workers=1)
    data = next(iter(loader))
    print(torch.mean(data[0], dim=(0,2,3)), torch.std(data[0], dim=(0, 2, 3))) 
    '''
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

#convolutional neural network model 
class TrashClassifier(nn.Module):
    def __init__(self, num_classes=4): 
        super(TrashClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True) #efficientnetB0 base model pretrained on imagenet
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) #remove final classification layer from pretrained model
        self.classifier = nn.Linear(1280, num_classes) #define custom classification layer, 1280 is num of features in efficientnet model

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
#select device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loss function and optimizer
lossFunct = nn.CrossEntropyLoss()
classifier = TrashClassifier().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

#training
train_losses, val_losses = [], []
num_epochs = 10
if __name__ == "__main__":
    for epoch in range(10):
        print(f'epoch: {epoch + 1}')
        classifier.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc="Training loop"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = lossFunct(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(dataloader.dataset)
        train_losses.append(train_loss)

        #validation 
        classifier.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valloader, desc= "Validation loop"):
                outputs = classifier(images)
                loss = lossFunct(outputs, labels)
                running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(valloader.dataset)
        val_losses.append(val_loss)

    #testing
    classifier.eval()
    test_loss = 0.0
    correct= 0
    total= 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Testing loop"):
            outputs = classifier(images)
            loss = lossFunct(outputs, labels)
            test_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()

    accuracy = correct/total
    test_loss /= len(testloader.dataset)
    print(f"Test Loss: {test_loss}, test accuracy: {accuracy}")

PATH = r'C:\Users\anjan\OneDrive\Documents\edge-utra\trash_classifier.pth'
torch.save(classifier.state_dict(),PATH)
