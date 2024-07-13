from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from main import f1_score

test_data_dir = Path('dataset/test')

# Tạo generator để tải ảnh
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(224, 224), batch_size=32,
                                                  class_mode='categorical', shuffle=False)

# Tải mô hình đã được huấn luyện
model = load_model('output/best_model.keras', custom_objects={'f1_score': f1_score})

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Lấy nhãn thực tế
y_true = test_generator.classes

# Tạo confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Tạo đối tượng ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())

# Vẽ confusion matrix
disp.plot(cmap='Blues')

# Thêm tiêu đề
plt.title('Confusion Matrix')
plt.savefig("output/conf_matrix.png")

# Hiển thị biểu đồ
# plt.show()
