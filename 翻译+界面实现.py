import pickle
from keras_transformer import get_model, decode

path = 'middle_data/'
with open('D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\encode_input.pkl' , 'rb') as f:
    encode_input = pickle.load(f)
with open('D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\decode_input.pkl' , 'rb') as f:
    decode_input = pickle.load(f)
with open('D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\decode_output.pkl', 'rb') as f:
    decode_output = pickle.load(f)
with open('D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\source_token_dict.pkl', 'rb') as f:
    source_token_dict = pickle.load(f)
with open(r'D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\target_token_dict.pkl', 'rb') as f:
    target_token_dict = pickle.load(f)
with open('D:\桌面\文件\机器翻译\Github机器翻译下载\MachineTranslation-Transformer-master\MachineTranslation-Transformer-master\middle_data\source_tokens.pkl', 'rb') as f:
    source_tokens = pickle.load(f)
print('Done')


print(len(source_token_dict))
print(len(target_token_dict))
print(len(encode_input))
# 构建模型
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=64,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=256,
    dropout_rate=0.05,
    use_same_embed=False,  # 不同语言需要使用不同的词嵌入
)
model.compile('adam', 'sparse_categorical_crossentropy')
# model.summary()
print('模型构建完成')

#加载模型
model.load_weights('model/W-- 40-0.0563-.h5')
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
print('模型加载完成')

import jieba
def get_input(seq):
    seq = ' '.join(jieba.lcut(seq, cut_all=False))
    # seq = ' '.join(seq)
    seq = seq.split(' ')
    print(seq)
    seq = ['<START>'] + seq + ['<END>']
    seq = seq + ['<PAD>'] * (34 - len(seq))
    print(seq)
    for x in seq:
        try:
            source_token_dict[x]
        except KeyError:
            flag=False
            break
        else:
            flag=True
    if(flag):
        seq = [source_token_dict[x] for x in seq]
    return flag, seq

def get_ans(seq):
    decoded = decode(
    model,
    [seq],
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
    # top_k=10,
    # temperature=1.0,
  )
    # print(' '.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1])))
    return ' '.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1]))

def tr(seq):
    while True:
        #seq = input("请输入需要翻译的中文：",)
        if seq == 'x':
            break
        flag, seq = get_input(seq)
        if(flag):
            get_ans(seq)
            return get_ans(seq)

        else:
            return '抱歉，数据集太小，还没能训练出能你需要翻译的句子的能力'

from PyQt6 import QtWidgets, QtGui
class TranslateApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # 创建单行文本输入框和标签
        input_textbox = QtWidgets.QLineEdit()
        translated_label = QtWidgets.QLabel('')
        # 设置标签不透明
        translated_label.setStyleSheet('background-color: white; opacity: 1;')

        # 创建翻译按钮，并绑定 fun 函数
        translate_button = QtWidgets.QPushButton('翻译')
        translate_button.clicked.connect(lambda: self.fun(input_textbox.text(), translated_label))

        # 设置界面布局
        layout = QtWidgets.QVBoxLayout()

        # 添加背景图片（使用QLabel）
        background_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap('OIP-C (1).jpg')  # 修改文件路径为自己的图片路径
        background_label.setPixmap(pixmap)
        background_label.setScaledContents(True)
        layout.addWidget(background_label)

        layout.addWidget(input_textbox)
        layout.addWidget(translate_button)
        layout.addWidget(translated_label)

        self.setLayout(layout)
        self.show()

    def fun(self, text, label):
        # TODO: 进行翻译操作，将结果设置为 label 的文本内容
        # 示例：将原始文本逆序输出并设置为标签文本
        result = tr(text)
        label.setText(result)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = TranslateApp()
    window.resize(400, 300)
    window.show()

    app.exec()

