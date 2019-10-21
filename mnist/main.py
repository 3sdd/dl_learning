import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #パラメータ
    epochs=30
    batch_size=256
    num_workers=4
    lr=0.01

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])

    #データセット作成
    ds_train=torchvision.datasets.MNIST(root="../Datasets/",train=True,transform=transform,download=True)
    ds_test=torchvision.datasets.MNIST(root="../Datasets",train=False,transform=transform,download=True)
    #データローダつくる
    dl_train=data.DataLoader(ds_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    dl_test=data.DataLoader(ds_train,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    #モデル作成
    model=nn.Sequential(
        nn.Linear(28*28,100),
        nn.ReLU(inplace=True),
        nn.Linear(100,10),
    )
    model=model.to(device)
    print(model)

    #optimizerとloss関数
    optimizer=optim.SGD(model.parameters(),lr=lr)
    criterion=nn.CrossEntropyLoss()

    print("学習開始")
    for epoch in range(epochs):
        print("開始epoch[{0}/{1}]".format(epoch+1,epochs))
        correct=0
        total=0
        for imgs,labels in tqdm(dl_train):
            optimizer.zero_grad()
            imgs,labels=imgs.to(device),labels.to(device)
            #imgs size=(batch_size,1,28,28)
            imgs=imgs.view(-1,28*28)
            outputs=model(imgs)

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            #精度用
            predicted=torch.argmax(outputs,dim=1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
        accuracy=correct/total
        print("精度(training):",accuracy*100,"%")


    print("学習終了")
    torch.save(model.state_dict(),"./mnist_test_model.pth")

    print("テスト")
    correct=0
    total=0
    with torch.no_grad():
        for imgs,labels in tqdm(dl_test):
            imgs,labels=imgs.to(device),labels.to(device)
            imgs=imgs.view(-1,28*28)
            outputs=model(imgs)
            predicted=torch.argmax(outputs,dim=1)

            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)       
    accuracy=correct/total
    print("精度(test):",accuracy*100,"%")
