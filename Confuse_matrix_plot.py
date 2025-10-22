import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = np.load('4Confuse_matrix.npy')

labels = ['Blink', 'Down', 'Left', 'Right', 'Up']

plt.figure(figsize=(8, 6))

# 绘制混淆矩阵图
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Dataset5')
plt.colorbar()

# 添加坐标轴标签
# tick_marks = np.arange(len(confusion_matrix))
# plt.xticks(tick_marks, tick_marks)
# plt.yticks(tick_marks, tick_marks)

plt.tight_layout()
plt.yticks(range(5), labels)
plt.xticks(range(5), labels,rotation=45)#X轴字体倾斜45°
# 添加文本标签
thresh = confusion_matrix.max() / 2.
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Predicted Label')
plt.xlabel('True Label')

# 显示图表
plt.savefig('Confusion_Dataset5.png', bbox_inches='tight',dpi = 600)
# plt.show()
a=1