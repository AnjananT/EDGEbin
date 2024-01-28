import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim #defines optimizer
from torch.utils.data import Dataset, DataLoader #
import torchvision.transforms as transforms #to normalize images
from torchvision.datasets import ImageFolder
from tqdm import tqdm

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
    
    #datasets for training, validation, and testing
    dataset= WasteDataset(data_root, transform)
    testset = WasteDataset(test_root, transform)
    valset = WasteDataset(val_root, transform)

    '''code used to calculate  mean and std values, tensor([0.7422, 0.7366, 0.7117]), tensor([0.3060, 0.3009, 0.3302])
    loader = DataLoader(datasetTransformed, batch_size=len(datasetTransformed), num_workers=1)
    data = next(iter(loader))
    print(torch.mean(data[0], dim=(0,2,3)), torch.std(data[0], dim=(0, 2, 3))) #https://pytorch.org/docs/stable/generated/torch.std.html
    '''
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #to batch our samples to increase model training speed
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

#custom convolutional neural network model
class SimpleTrashClassifier(nn.Module):
    def __init__(self): 
        super(SimpleTrashClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*56*56, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64,4)

    def forward(self, x): 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32*56*56)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    

#iselect device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loss function and optimizer
lossFunct = nn.CrossEntropyLoss()
classifier = SimpleTrashClassifier().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.0001)


#Training
train_losses, val_losses = [], []
num_epochs = 10
if __name__ == "__main__":
    for epoch in range(10):
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

        #Validation 
        classifier.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valloader, desc="Validation loop"):
                outputs = classifier(images)
                loss = lossFunct(outputs, labels)
                running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(valloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

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


PATH = r'C:\Users\anjan\OneDrive\Documents\edge-utra\trash_classifier3.pth'
torch.save(classifier.state_dict(),PATH)

       
