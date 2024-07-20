import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# 生成数据
file_path = 'dataset.xlsx'
data = pd.read_excel(file_path)
data = data.astype({'x1': 'float', 'x2': 'float', 'x3': 'float', 'y1': 'category', 'y2': 'float'})

# 分割数据为特征和标签
X = data[['x1', 'x2', 'x3']].values
Y_class = data['y1'].values
Y_reg = data['y2'].values
# print one line of data
i= 180
print(X[i], Y_class[i], Y_reg[i])

# onehot encode the class labels
encoder = OneHotEncoder(sparse_output=False)
Y_class = encoder.fit_transform(Y_class.reshape(-1, 1))
# Y_class = Y_class.toarray()


X_train, X_test, Y_class_train, Y_class_test, Y_reg_train, Y_reg_test = train_test_split(
    X, Y_class, Y_reg, test_size=0.2, random_state=42)


input_layer = Input(shape=(3,))  # 输入层，对应3个特征

# 隐藏层
hidden_layer_1 = Dense(1024, activation='elu')(input_layer)
hidden_layer_2 = Dense(512, activation='elu')(hidden_layer_1)
hidden_layer_3 = Dense(286, activation='elu')(hidden_layer_2)

# 输出层
class_output = Dense(8, activation='softmax', name='class_output')(hidden_layer_2)  # 多分类输出层
reg_output = Dense(1, name='reg_output')(hidden_layer_2)  # 回归输出层

# 定义模型
model = Model(inputs=input_layer, outputs=[class_output, reg_output])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'class_output': 'categorical_crossentropy', 'reg_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy', 'reg_output': 'mean_squared_error'})

history = model.fit(X_train, [Y_class_train, Y_reg_train],
                    validation_split=0.3,
                    epochs=100,
                    batch_size=32,
                    verbose=2)

x_new = X[100]
y_class_p, y_reg_p = model.predict(x_new)
i=1
print(y_class_p[0])
print(y_reg_p[0])