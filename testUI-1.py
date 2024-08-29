# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from SFTP_remote import *
import paramiko


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(970, 830)
        MainWindow.setMinimumSize(QtCore.QSize(970, 830))
        MainWindow.setMaximumSize(QtCore.QSize(1440, 900))
        MainWindow.setStyleSheet("background-color: rgb(255, 253, 244);\n"
                                 "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(950, 780))
        self.centralwidget.setMaximumSize(QtCore.QSize(1440, 858))
        self.centralwidget.setStyleSheet("background-color: rgb(255, 253, 244);\n"
                                         "")
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, -10, 1041, 881))

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(901, 761))
        self.frame.setMaximumSize(QtCore.QSize(1440, 900))
        self.frame.setSizeIncrement(QtCore.QSize(31, 27))
        font = QtGui.QFont()
        font.setFamily("Futura")
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.frame.setFont(font)
        self.frame.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.frame.setStyleSheet("background-color: rgb(255, 253, 244);\n"
                                 "font: 75 24pt \"Futura\";")
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(10, 20, 1021, 841))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Title = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.Title.setMinimumSize(QtCore.QSize(0, 70))
        self.Title.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setFamily("Futura")
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.Title.setFont(font)
        self.Title.setTextFormat(QtCore.Qt.AutoText)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.verticalLayout.addWidget(self.Title)
        self.frame_2 = QtWidgets.QFrame(self.verticalLayoutWidget_3)
        self.frame_2.setMinimumSize(QtCore.QSize(889, 500))
        font = QtGui.QFont()
        font.setFamily("Futura")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(7)
        self.frame_2.setFont(font)
        self.frame_2.setAutoFillBackground(False)
        self.frame_2.setStyleSheet("font: 57 14pt \"Futura\";\n"
                                   "background-color: rgb(255, 251, 232);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setLineWidth(2)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1001, 741))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Left_verticalLayout = QtWidgets.QVBoxLayout()
        self.Left_verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.Left_verticalLayout.setObjectName("Left_verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem)
        self.Model = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.Model.setMinimumSize(QtCore.QSize(400, 50))
        self.Model.setMaximumSize(QtCore.QSize(400, 50))
        self.Model.setAutoFillBackground(False)
        self.Model.setStyleSheet("border-color: rgb(181, 180, 244);")
        self.Model.setFlat(False)
        self.Model.setCheckable(False)
        self.Model.setObjectName("Model")
        self.AtUNet_label = QtWidgets.QLabel(self.Model)
        self.AtUNet_label.setGeometry(QtCore.QRect(160, 20, 71, 21))
        self.AtUNet_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.AtUNet_label.setAlignment(QtCore.Qt.AlignCenter)
        self.AtUNet_label.setObjectName("AtUNet_label")
        self.Left_verticalLayout.addWidget(self.Model)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem1)
        self.Testfile_type = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.Testfile_type.setMinimumSize(QtCore.QSize(400, 80))
        self.Testfile_type.setMaximumSize(QtCore.QSize(16777215, 80))
        self.Testfile_type.setAutoFillBackground(False)
        self.Testfile_type.setStyleSheet("border-color: rgb(181, 180, 244);")
        self.Testfile_type.setFlat(False)
        self.Testfile_type.setCheckable(False)
        self.Testfile_type.setObjectName("Testfile_type")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.Testfile_type)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 30, 381, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.Type_choices = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.Type_choices.setContentsMargins(10, 5, 0, 5)
        self.Type_choices.setSpacing(0)
        self.Type_choices.setObjectName("Type_choices")
        self.only_btn = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.only_btn.setMaximumSize(QtCore.QSize(170, 16777215))
        self.only_btn.setStyleSheet("font: 480 13pt \"Avenir\";")
        self.only_btn.setChecked(False)
        self.only_btn.setObjectName("only_btn")
        self.Type_choices.addWidget(self.only_btn)
        self.pair_btn = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.pair_btn.setMaximumSize(QtCore.QSize(200, 16777215))
        self.pair_btn.setStyleSheet("font: 480 13pt \"Avenir\";")
        self.pair_btn.setChecked(True)
        self.pair_btn.setObjectName("pair_btn")
        self.Type_choices.addWidget(self.pair_btn)
        self.Left_verticalLayout.addWidget(self.Testfile_type)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem2)
        self.Modes = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.Modes.setMinimumSize(QtCore.QSize(390, 80))
        self.Modes.setMaximumSize(QtCore.QSize(16777215, 80))
        self.Modes.setObjectName("Modes")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.Modes)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 30, 381, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.Mode_choices = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.Mode_choices.setContentsMargins(10, 5, 0, 5)
        self.Mode_choices.setSpacing(0)
        self.Mode_choices.setObjectName("Mode_choices")
        self.denoise_btn = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.denoise_btn.setStyleSheet("font: 500 14pt \"Avenir\";")
        self.denoise_btn.setChecked(True)
        self.denoise_btn.setObjectName("denoise_btn")
        self.Mode_choices.addWidget(self.denoise_btn)
        self.derain_btn = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.derain_btn.setStyleSheet("font: 500 14pt \"Avenir\";")
        self.derain_btn.setObjectName("derain_btn")
        self.Mode_choices.addWidget(self.derain_btn)
        self.deblur_btn = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.deblur_btn.setStyleSheet("font: 500 14pt \"Avenir\";")
        self.deblur_btn.setObjectName("deblur_btn")
        self.Mode_choices.addWidget(self.deblur_btn)
        self.Left_verticalLayout.addWidget(self.Modes)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem3)
        self.test_file = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.test_file.setMaximumSize(QtCore.QSize(16777215, 88))
        self.test_file.setObjectName("test_file")
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.test_file)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(10, 20, 381, 56))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.test_file_text = QtWidgets.QLineEdit(self.horizontalLayoutWidget_7)
        self.test_file_text.setStyleSheet("background-color: rgb(228, 244, 242);\n"
                                          "font: 57 13pt \"Avenir\";")
        self.test_file_text.setObjectName("test_file_text")
        self.horizontalLayout_4.addWidget(self.test_file_text)
        self.test_file_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.test_file_btn.setAutoFillBackground(False)
        self.test_file_btn.setStyleSheet("font: 57 13pt \"Avenir\";\n"
                                         "background-color: rgb(161, 224, 251);")
        self.test_file_btn.setObjectName("test_file_btn")
        self.horizontalLayout_4.addWidget(self.test_file_btn)
        self.Left_verticalLayout.addWidget(self.test_file)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem4)
        self.res_path = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.res_path.setMaximumSize(QtCore.QSize(16777215, 88))
        self.res_path.setObjectName("res_path")
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.res_path)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(10, 20, 381, 64))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(10)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.res_path_text = QtWidgets.QLineEdit(self.horizontalLayoutWidget_8)
        self.res_path_text.setStyleSheet("background-color: rgb(228, 244, 242);\n"
                                         "font: 57 13pt \"Avenir\";\n"
                                         "")
        self.res_path_text.setObjectName("res_path_text")
        self.horizontalLayout_6.addWidget(self.res_path_text)
        self.res_path_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        self.res_path_btn.setStyleSheet("font: 57 13pt \"Avenir\";\n"
                                        "background-color: rgb(161, 224, 251);")
        self.res_path_btn.setObjectName("res_path_btn")
        self.horizontalLayout_6.addWidget(self.res_path_btn)
        self.Left_verticalLayout.addWidget(self.res_path)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem5)
        self.Btn_set = QtWidgets.QHBoxLayout()
        self.Btn_set.setObjectName("Btn_set")
        self.apply_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.apply_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.apply_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.apply_btn.setStyleSheet("background-color: rgb(255, 151, 137);")
        self.apply_btn.setObjectName("apply_btn")
        self.Btn_set.addWidget(self.apply_btn)
        self.reset_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.reset_btn.setMinimumSize(QtCore.QSize(100, 29))
        self.reset_btn.setMaximumSize(QtCore.QSize(100, 29))
        self.reset_btn.setStyleSheet("background-color: rgb(255, 215, 119);")
        self.reset_btn.setObjectName("reset_btn")
        self.Btn_set.addWidget(self.reset_btn)
        self.Left_verticalLayout.addLayout(self.Btn_set)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.Left_verticalLayout.addItem(spacerItem6)
        self.horizontalLayout.addLayout(self.Left_verticalLayout)
        self.line_2 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.Right_verticalLayout = QtWidgets.QVBoxLayout()
        self.Right_verticalLayout.setContentsMargins(5, -1, 8, -1)
        self.Right_verticalLayout.setObjectName("Right_verticalLayout")
        self.Degraded_img_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Degraded_img_label.setEnabled(True)
        self.Degraded_img_label.setMinimumSize(QtCore.QSize(430, 19))
        self.Degraded_img_label.setMaximumSize(QtCore.QSize(420, 19))
        self.Degraded_img_label.setObjectName("Degraded_img_label")
        self.Right_verticalLayout.addWidget(self.Degraded_img_label)
        self.img_in_area = QtWidgets.QHBoxLayout()
        self.img_in_area.setObjectName("img_in_area")
        self.input_img = QtWidgets.QGraphicsView(self.horizontalLayoutWidget)
        self.input_img.setMinimumSize(QtCore.QSize(420, 159))
        self.input_img.setStyleSheet("background-color: rgba(255, 255, 255, 97);")
        self.input_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.input_img.setObjectName("input_img")
        self.img_in_area.addWidget(self.input_img)
        self.Right_verticalLayout.addLayout(self.img_in_area)
        self.Restored_img_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Restored_img_label.setMinimumSize(QtCore.QSize(420, 19))
        self.Restored_img_label.setMaximumSize(QtCore.QSize(420, 19))
        self.Restored_img_label.setObjectName("Restored_img_label")
        self.Right_verticalLayout.addWidget(self.Restored_img_label)
        self.img_out_area = QtWidgets.QHBoxLayout()
        self.img_out_area.setObjectName("img_out_area")
        self.out_img = QtWidgets.QGraphicsView(self.horizontalLayoutWidget)
        self.out_img.setMinimumSize(QtCore.QSize(420, 171))
        self.out_img.setStyleSheet("background-color: rgba(255, 255, 255, 97);")
        self.out_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.out_img.setObjectName("out_img")
        self.img_out_area.addWidget(self.out_img)
        self.Right_verticalLayout.addLayout(self.img_out_area)
        self.line = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.Right_verticalLayout.addWidget(self.line)
        self.Result_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Result_label.setMaximumSize(QtCore.QSize(420, 19))
        self.Result_label.setObjectName("Result_label")
        self.Right_verticalLayout.addWidget(self.Result_label)
        self.test_output = QtWidgets.QTextBrowser(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Futura")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(7)
        self.test_output.setFont(font)
        self.test_output.setStyleSheet("font: 23 11pt \"Monaco\";\n"
                                       "background-color: rgba(255, 255, 255, 97);")
        self.test_output.setFrameShape(QtWidgets.QFrame.Box)
        self.test_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.test_output.setReadOnly(True)
        self.test_output.setObjectName("test_output")
        self.Right_verticalLayout.addWidget(self.test_output)
        self.horizontalLayout.addLayout(self.Right_verticalLayout)
        self.verticalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 970, 21))
        self.menubar.setObjectName("menubar")
        self.menutestUI = QtWidgets.QMenu(self.menubar)
        self.menutestUI.setObjectName("menutestUI")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menutestUI.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ####### btn click ######
        self.test_file_btn.clicked.connect(self.select_test_file)
        self.res_path_btn.clicked.connect(self.select_result_path)
        # self.denoise_btn.stateChanged.connect(self.changecb2)
        # self.derain_btn.stateChanged.connect(self.changecb2)
        # self.deblur_btn.stateChanged.connect(self.changecb2)
        self.apply_btn.clicked.connect(self.apply)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title.setText(_translate("MainWindow", "IMAGE RESTORATION TEST"))
        self.Model.setTitle(_translate("MainWindow", "Model"))
        self.AtUNet_label.setText(_translate("MainWindow", "AtUNet"))
        self.Testfile_type.setTitle(_translate("MainWindow", "Type of test file"))
        self.only_btn.setText(_translate("MainWindow", "degraded image only"))
        self.pair_btn.setText(_translate("MainWindow", r"degarded & groundtruth pairs"))
        self.Modes.setTitle(_translate("MainWindow", "Mode"))
        self.denoise_btn.setText(_translate("MainWindow", "denoise"))
        self.derain_btn.setText(_translate("MainWindow", "derain"))
        self.deblur_btn.setText(_translate("MainWindow", "deblur"))
        self.test_file.setToolTip(_translate("MainWindow",
                                             "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                             "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                             "p, li { white-space: pre-wrap; }\n"
                                             "</style></head><body style=\" font-family:\'Times New Roman\'; font-size:11pt; font-weight:25; font-style:normal;\">\n"
                                             "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-style:italic;\">If U are using pairs of test files, we suppose to store each pair in one folder, and please upload the directory of these folders, THKs.</span></p></body></html>"))
        self.test_file.setTitle(_translate("MainWindow", "Choose test file(s):"))
        self.test_file_btn.setText(_translate("MainWindow", "Browse"))
        self.res_path.setTitle(_translate("MainWindow", "Choose result path:"))
        self.res_path_btn.setText(_translate("MainWindow", "Browse"))
        self.apply_btn.setText(_translate("MainWindow", "Apply"))
        self.reset_btn.setText(_translate("MainWindow", "Reset"))
        self.Degraded_img_label.setText(_translate("MainWindow", "Degraded image example"))
        self.Restored_img_label.setText(_translate("MainWindow", "Restored image example"))
        self.Result_label.setText(_translate("MainWindow", "Result"))
        self.menutestUI.setTitle(_translate("MainWindow", "testUI"))

    ###### self-defined funcs ######
    def printf(self, msg):
        self.test_output.append(msg)  # 在指定的区域显示提示信息
        QApplication.processEvents()
        self.cursot = self.test_output.textCursor()
        self.test_output.moveCursor(self.cursot.End)

    def select_test_file(self):
        foldername = QFileDialog.getExistingDirectory(None, "Select Directory", "../")
        self.test_file_text.setText(foldername)

    def select_result_path(self):
        foldername = QFileDialog.getExistingDirectory(None, "Select Directory", "../")
        self.res_path_text.setText(foldername)

    def apply(self):
        self.useCmd()


    def useCmd(self):
        mySSH = SSHConnection(remote_dict)
        mySSH.connect()
        # data_remote_path = '/root/autodl-tmp/FYP-Yin/dataset'
        res_remote_path = '/root/autodl-tmp/FYP-Yin/results'
        res_local_path = '/Volumes/Work Space/Codes/Python/FYP-Yin/results'

        if self.only_btn.isChecked():
            test_type = 'only'
        if self.pair_btn.isChecked():
            test_type_type = 'pair'

        # mode = 'denoise'
        if self.denoise_btn.isChecked():
            mode = 'denoise'
        if self.derain_btn.isChecked():
            mode = 'derain'
        if self.deblur_btn.isChecked():
            mode = 'deblur'

        data_local_path = self.test_file_text.text()
        res_local_path = self.res_path_text.text()

        foldername = os.path.split(data_local_path)[-1]
        data_remote_path = os.path.join('/root/autodl-tmp/FYP-Yin/dataset', foldername)

        mySSH.run_cmd('source /etc/profile \nsource ~/.bashrc \ncd /root/autodl-tmp/FYP-Yin/source')

        mySSH.sftp_upload_dir(data_local_path, data_remote_path)
        # print(data_remote_path)

        cmd = "python3 test5.py --arch AtUNet" + " --mode " + mode + " --gpu 0 --input_dir " + data_remote_path + " --result_dir " + res_remote_path
        print(cmd)

        ssh = paramiko.SSHClient()
        ssh._transport = mySSH.transport
        try:
            stdin, stdout, stderr = ssh.exec_command('source /etc/profile \nsource ~/.bashrc \ncd /root/autodl-tmp/FYP-Yin/source \n' + cmd)
            for line in stdout:
                self.printf(line)
                QApplication.processEvents()
                # time.sleep(1)
                print(line)
            for error in stderr:
                self.printf(error)
                QApplication.processEvents()
                # time.sleep(1)
                print(error)
        except Exception as e:
            print("Execute command %s erorr, error msg: %s" % (cmd, e))
            return ""

        # output = mySSH.run_cmd(cmd)
        # mySSH.sftp_download_dir(res_remote_path, res_local_path)
        # print(output['res'].strip('\n'))

        mySSH.close()


if __name__ == '__main__':
    mySSH = SSHConnection(remote_dict)
    mySSH.connect()

    app = QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainwindow)
    mainwindow.show()

    sys.exit(app.exec_())
    mySSH.close()
