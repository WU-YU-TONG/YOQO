import matplotlib.pyplot as plt
# online = [85.4, 83.2, 81.2, 76.0, 73.8]
# Hotjumpskip = [84.8, 82.3, 79.5, 75, 72.7]
# Data_aug = [76.7, 74.3, 70.07, 68.6, 66.3]
# gap_atk = [70.7, 68.0, 66.8, 65.3, 63.4]

# x = [1500, 2500, 3500, 5000, 7500]

# plt.plot(x, online, c='r', linestyle='--', marker='o', label='Online Atk')
# plt.plot(x, Hotjumpskip, c='g', linestyle='--', marker='^', label='Boundary Atk')
# plt.plot(x, gap_atk, c='b', linestyle='--', marker='x', label='Gap Atk')
# plt.plot(x, Data_aug, c='orange', linestyle='--', marker='*', label='Data Augmentation Atk')
# plt.legend()
# plt.xlabel('Dataset Size')
# plt.ylabel('ASR')
# plt.savefig('./overfit_cons.jpg')
x = ['CNN7', 'VGG16', 'Assembly']
vgg16 = [76.38, 70.0, 71.33]
resnet18 = [78.54, 82.14, 79.10]
cnn7 = [80.67, 75.73, 75.71]
densenet121 = [75.07, 74.27, 74.58]
inceptionv3 = [77.76, 76.70, 73.73]
seresnet18 = [81.32, 81.66, 82.91]
sizes = [60,60,60]
plt.scatter(x, vgg16, c='r', marker='^', label='VGG16', sizes=sizes)
plt.scatter(x, resnet18, c='skyblue', marker='o', label='ResNet18', sizes=sizes)
plt.scatter(x, cnn7, c='pink', marker='v', label='CNN7', sizes=sizes)
plt.scatter(x, densenet121, c='g', marker='D', label='DenseNet121', sizes=sizes)
plt.scatter(x, inceptionv3, c='orange', marker='s', label='Inceptionv3', sizes=sizes)
plt.scatter(x, seresnet18, c='b', marker='p', label='seRenNet18', sizes=sizes)
plt.legend()
plt.grid(linestyle='--')
plt.xlabel('Shadow Model Types')
plt.ylabel('ASR')
plt.savefig('./transferability.jpg')