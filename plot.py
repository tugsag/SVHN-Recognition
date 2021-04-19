import matplotlib.pyplot as plt
import os
import keras


dir = 'report'
#
#
# acc = [.5038, .8884, .9339, .9366, .9553]
# val_acc = [.7928, .9063, .9121, .9195, .9320]
# loss = [1.9909, .3985, .2408, .2330, .1668]
# val_loss = [.7220, .3376, .3470, .3026, .2881]

# acc = [.4459, .5263, .8346, .8513, .9309]
# val_acc = [.4479, .3850, .2925, .8935, .8074]
# loss = [1.9253, 1.5632, .5888, .5781, .2511]
# val_loss = [1.9126, 1.8554, 2.9671, .4143, .8228]
# # vgg_empty
# Test Score:  0.8227839469909668
# Test Accuracy:  0.8074173927307129

plt.plot(acc)
plt.plot(val_acc)
plt.title('VGG16 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.savefig(os.path.join(dir, 'vgg_acc_empty.png'))
plt.clf()

plt.plot(loss)
plt.plot(val_loss)
plt.title('VGG16 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.savefig(os.path.join(dir, 'vgg_loss_empty.png'))
plt.clf()
