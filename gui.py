import sys
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5 import uic
import re

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils.wrapper import predict_class


ui_file = "./Classify.ui"
class MainDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self, None)
        uic.loadUi(ui_file, self)
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.gridLayout.addWidget(self.canvas)
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax1.set_title('Age Probability')
        self.ax2.set_title('Gender Probability')
        self.ax3.set_title('Violence Level')
        self.ax4.set_title('Community Probability')

        self.PushButtonClassify.clicked.connect(self.buttonClicked)

    def predictCommunity(self):
        prob = predict_class(
            self.input_text, emb_dim=512, hid_dim=128,
            class_num=2, bidirectional=False, dropout=0.5,
            weight_file='model/weights.pt', vocab_file='saved_vocab.pt', device='cpu')
        self.df4 = pd.DataFrame(
            {
                'prob': prob,
                'label': ['dcinside', 'ruliweb'],
            }
        )
        self.df4 = self.df4.sort_values(by='prob', ascending=False, ignore_index=True)

    def drawBarGraph(self, ax, df, title='Probability'):
        colors = ['#6699CC', 'silver']
        ax.cla()
        ax.set_title(title)
        ax.bar(df.index, df.prob, color=colors)
        ax.set_ylabel('Probability')
        ax.set_xticks(df.index)
        ax.set_xticklabels(labels=df.label)
        for ii, pp in enumerate(df.prob):
            ax.text(ii, pp+0.01, f'{pp*100:.3f}%', ha='center')
    
    def drawPieChart(self, ax, df, title='Probability'):
        ax.cla()
        ax.set_title(title)
        colors = ['#6699CC', 'silver']
        wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 1}
        ax.pie(df.prob, labels=df.label, autopct='%.3f%%',
                startangle=90, counterclock=False,
                colors=colors, wedgeprops=wedgeprops)

    def clearAll(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax1.set_title('Age Probability')
        self.ax2.set_title('Gender Probability')
        self.ax3.set_title('Violence Level')
        self.ax4.set_title('Community Probability')

    def buttonClicked(self):
        self.input_text = self.textEdit.toPlainText()
        self.input_text = re.sub('(\s|,)+', ' ', self.input_text)
        self.input_text = re.sub(r'[^ 가-힣]', '', self.input_text)
        if len(self.input_text) > 5:
            self.predictCommunity()
            if self.barButton.isChecked():
                self.drawBarGraph(self.ax4, self.df4, "Community Probability")
            elif self.pieButton.isChecked():
                self.drawPieChart(self.ax4, self.df4, "Community Probability")
            self.canvas.draw()
        else:
            self.clearAll()
            msg = self.messageBox()
    

    def messageBox(self):
        msgBox = QMessageBox()
        msgBox.setWindowTitle("오류")
        msgBox.setIcon(QMessageBox.NoIcon)
        msgBox.setText("ERROR!")
        msgBox.setInformativeText("한글이 5글자 이상 포함되어 있어야 합니다.")
        msgBox.setStandardButtons(QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Cancel)

        return msgBox.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_dialog = MainDialog()
    main_dialog.show()
    sys.exit(app.exec_())
        

