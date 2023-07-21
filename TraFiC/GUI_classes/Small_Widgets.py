# Version 0.83 - 2022, September 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PyQt5.QtWidgets import (QMainWindow, QWidget, QFrame, QCheckBox, \
                             QLabel, QHBoxLayout, QVBoxLayout, \
                             QPushButton, QButtonGroup)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont, QCloseEvent
#========================================================================
class QHLine(QFrame):
    """Horizontal Line"""
    def __init__(self):
        QFrame.__init__(self)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
#========================================================================
class QVLine(QFrame):
    """Vertical Line"""
    def __init__(self):
        QFrame.__init__(self)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
#========================================================================
class CheckBoxGroup(QButtonGroup):
    """Group of exclusive check boxes."""
    stateChanged = pyqtSignal(object,int)
    #--------------------------------------------------------------------
    def __init__(self, check_boxes, parent=None, verbose=False) :
        QButtonGroup.__init__(self, parent)
        self.__boxes = check_boxes
        self.__verb = verbose
        first_box = self.__boxes[0]
        self.addButton(first_box)
        first_box.setChecked(True)
        self.__prev_box = first_box
        for box in self.__boxes[1:] :
            self.addButton(box)
            box.setChecked(False)
        self.buttonReleased.connect(self.__update)
    #--------------------------------------------------------------------
    def __update(self, clicked_box=None) :
        if clicked_box is None :
            no = -2
        else :
            no = self.__boxes.index(clicked_box)
        if self.__verb :
            print(f"CheckBoxGroup.__update(box no.{no})")
        if clicked_box is self.__prev_box : # Uncheck all
            self.setExclusive(False)
            clicked_box.setChecked(False)
            self.setExclusive(True)
            self.__prev_box = None
            self.stateChanged.emit(self, 0)
            no = -1
        else : 
            self.__prev_box = clicked_box
        self.stateChanged.emit(self, no)
#========================================================================
class Selector(QFrame):
    """Selector with check boxes."""
    released = pyqtSignal()
    stateChanged = pyqtSignal()
    #--------------------------------------------------------------------
    def __init__(self, label, items, parent=None, exclusive_items=[], \
                 verbose=False):
        QFrame.__init__(self, parent)
        self.__labels = tuple( QLabel("  "+txt) for txt in items )
        self.__check_boxes = tuple(QCheckBox(parent) for _ in items)
        self.__dict = dict( zip(items,self.__check_boxes) )
        # Group(s) of exclusive check boxes
        self.__bgrp = []
        excl_boxes = []
        err_msg = "Selector constructor error:\n\t" + \
                      "an item cannot belong to more than one group"
        for grp_it in exclusive_items :
            grp = []
            for itm in grp_it :
                box = self.__dict[itm]
                if box not in excl_boxes :
                    excl_boxes.append(box)
                    grp.append(box)
                else :
                    raise ValueError(err_msg)
            box_grp = CheckBoxGroup(grp, self, verbose)
            self.__bgrp.append(box_grp)
            box_grp.stateChanged.connect(self.__update)
        for box in self.__check_boxes :
            if box not in excl_boxes :
                box.setChecked(True)
                box.stateChanged.connect(self.__update)                        
        # Button
        but = QPushButton(label, parent)
        but.released.connect(self.__change)
        self.__verb = verbose
        self.__initUI(but)
    #--------------------------------------------------------------------
    def __initUI(self, but):
        self.setEnabled(True) # Activated       
        self.setFrameStyle(QFrame.Box|QFrame.Plain)
        self.setLineWidth(2)
        layout = QHBoxLayout()
        layout.addWidget(but)
        for lbl,chk in zip(self.__labels,self.__check_boxes) :
            layout.addWidget(lbl)
            layout.addWidget(chk)
        self.setLayout(layout)
    #--------------------------------------------------------------------
    @property
    def items(self) :
        LI = list(self.__dict.keys())
        LI.sort()
        return tuple(LI)
    #--------------------------------------------------------------------
    def setEnabled(self, is_true) : 
        self.__selected = is_true  
        for lbl,chk in zip(self.__labels,self.__check_boxes) :
            lbl.setEnabled(is_true)
            chk.setEnabled(is_true)
    #--------------------------------------------------------------------
    def isChecked(self, item) :
        return self.__dict[item].isChecked() 
    #--------------------------------------------------------------------
    def __change(self, arg1=None, arg2=None) :
        if self.__verb :
            print(f"Selector.__change({arg1},{arg2})")
        if self.__selected : # Unselected
            self.setEnabled(False)
        else : # Selected
            self.setEnabled(True)
            all_to_check = True
            for chk in self.__check_boxes :
                if chk.isChecked() :
                    all_to_check = False
                    break
            if all_to_check :
                for chk in self.__check_boxes :
                    chk.setChecked(True)
        self.released.emit()
    #--------------------------------------------------------------------
    def __update(self, arg1=None, arg2=None) :
        if self.__verb :
            print(f"Selector.__update({arg1},{arg2})")   
        self.stateChanged.emit()
#========================================================================
class sensitive_button(QPushButton) :
    def __init__(self, *args, **kwargs) :
        QPushButton.__init__(self, *args, **kwargs)
    def enterEvent(self, event) :        
        stsh = "color: rgb(0,100,0); background-color: rgb(200,255,200);"
        self.setStyleSheet(stsh)
    def leaveEvent(self, event) :
        stsh = "color: rgb(0,0,0); background-color: rgb(255,255,255);"
        self.setStyleSheet(stsh)    
#========================================================================
class PopUp_Select_Item(QWidget) :
    selected_item = pyqtSignal(int)
    __CL = 13 # Coefficient for button width
    __w_min = 20
    #--------------------------------------------------------------------
    def __init__(self, head, list_of_items, calling_window=None) :
        self.__call = calling_window
        self.__no_item = None
        QWidget.__init__(self, None)
        self.setWindowTitle("Select item...")
        nw = max( [ len(m) for m in [head]+list_of_items ] ) + 2
        ww = max( self.__w_min, round(self.__CL*nw) )
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        font.setFamily("Courier New")
        v_lay = QVBoxLayout()
        head_but = QPushButton(head, self)
        head_but.setFixedWidth(ww)
        stsh = "color: rgb(0,0,200); background-color: rgb(200,255,255);"
        head_but.setStyleSheet(stsh)
        head_but.setFont(font)
        v_lay.addWidget(head_but)
        stsh = "color: rgb(0,0,0); background-color: rgb(255,255,255);"
        for no,itm in enumerate(list_of_items, 1) :
            itm_but = sensitive_button(itm, self)
            itm_but.setFixedWidth(ww)
            itm_but.setFont(font)
            itm_but.setStyleSheet(stsh)   
            itm_but.released.connect(lambda i=no : self.__close(i))
            v_lay.addWidget(itm_but)
        v_lay.addStretch()
        self.setLayout(v_lay)
        self.setFixedWidth(ww+20)
        self.setFixedHeight( (no+1)*50 )
        if self.__call is None : self.show()
    #--------------------------------------------------------------------
    def __close(self, no) :
        print(f"PopUp_Select_Item.selected_item.emit({no})")
        self.selected_item.emit(no)
        self.close() 
#========================================================================
class PopUp_Interrupt(QMainWindow) :
    stop = pyqtSignal()
    __CL = 15 # Coefficient for button width
    __w_min = 20
    __H = 50
    #--------------------------------------------------------------------
    def __init__(self, label="Interrupt", calling_window=None) :
        self.__call = calling_window
        self.__stop = False
        QMainWindow.__init__(self)
        self.setWindowTitle("Interrupt")
        ww = max( self.__w_min, round(self.__CL*(len(label)+5)) )
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        font.setFamily("Courier New")
        v_lay = QVBoxLayout()
        print(f"PopUp_Interrupt: label = '{label}'")
        inter_but = sensitive_button(label, self)
        inter_but.released.connect(self.__interrupt)
        inter_but.setFixedWidth(ww)
        inter_but.setFixedHeight(self.__H)
        stsh = "color: rgb(0,0,0); background-color: rgb(255,255,255);"
        inter_but.setStyleSheet(stsh)
        inter_but.setFont(font)
        v_lay.addWidget(inter_but)
        v_lay.addStretch()
        cw = QWidget()
        self.setCentralWidget(cw)
        cw.setLayout(v_lay)
        self.setFixedWidth(ww+20)
        self.setFixedHeight(2*self.__H)
        #if self.__call is None : self.show()
        self.show()
    #--------------------------------------------------------------------
    def __interrupt(self) :
        print("PopUp_Interrupt.stop.emit()")
        self.stop.emit()
        self.close()
#========================================================================
if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication,QWidget
    import sys
    #--------------------------------------------------------------------
    class Test_Selector(QWidget) :
        def __init__(self, items, exclusive_items) :
            QWidget.__init__(self)
            self.__sel = Selector("Test", items, parent=self, \
                                  exclusive_items=exclusive_items, \
                                  verbose=True)
            self.setFixedSize(500,100)
            self.show()
    #--------------------------------------------------------------------
    app = QApplication(sys.argv)
#    my_app = Test_Selector(["a","b","c","d","e","f"], \
#                           [["b","c"],["d","e","f"]])
    my_app = PopUp_Select_Item("en-tête",["choix 1", "choix 2"])
#    my_app = PopUp_Interrupt()
    app.exec_()
        
    
