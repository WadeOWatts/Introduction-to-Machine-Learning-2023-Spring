training_model_v2.ipynb
這次使用到的model
resnet34
resnet50
resnext50
VGG19
VGG16
densenet16
Xception
都來自torchchvision內建函式models
如model = models.vgg16(pretrained=False).cuda()
整體架構皆如training_model_v2.ipynb所示

training_model_v3_forcontinue.ipynb
用來繼續訓練，整體架構如同training_model_v2.ipynb
提供兩種方法來加載model
1. 整個model load
如下r
model.load(torch.load("位置"))
2. 從記錄點load
如下
checkpoint = torch.load("位置")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']
criterion = nn.CrossEntropyLoss().cuda()


