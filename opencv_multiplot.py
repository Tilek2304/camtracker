import cv2
import numpy as np

class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 0, 0), (0, 128, 0), (0, 0, 250),
                           (148, 0, 62), (0, 255, 250), (250, 0, 250),
                           (0, 255, 250), (250, 250, 0), (100, 200, 200),
                           (200, 100, 200)]
        self.color = []
        self.val = []
        self.keys = []
        self.plot = np.ones((self.height, self.width, 3)) * 255

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])
            
    def setValName(self, keys):
        self.keys = keys
        
    def multiplot(self, val, label="plot"):
        self.val.append(val)
        if len(self.val) > self.width:
            self.val.pop(0)
        self.show_plot(label)

    def show_plot(self, label):
        self.plot = np.ones((self.height, self.width, 3)) * 255
        
        for i in range(len(self.keys)):
            cv2.putText(self.plot, f"# {self.keys[i]}", 
                        (self.width - 125, self.height - 10 - (i * 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_list[i], 1, cv2.LINE_AA)

        for i in range(len(self.val) - 1):
            for j in range(len(self.val[0])):
                # Нормализация для отображения
                # Предположим, что значения находятся в диапазоне высоты кадра (0-240)
                # Вы можете изменить `self.height/2` на другой масштаб, если нужно
                y1 = int(self.height / 2) - int(self.val[i][j] - self.val[i][j+1 if j % 2 == 0 else j-1])
                y2 = int(self.height / 2) - int(self.val[i+1][j] - self.val[i+1][j+1 if j % 2 == 0 else j-1])
                
                # Ограничение значений, чтобы не выходить за рамки
                y1 = max(0, min(y1, self.height-1))
                y2 = max(0, min(y2, self.height-1))

                # Рисуем линию только если она относится к значению, а не к сетпоинту
                if j % 2 == 0:
                     cv2.line(self.plot, (i, y1), (i + 1, y2), self.color[j], 1)
                else: # Рисуем линию сетпоинта
                     cv2.line(self.plot, (i, self.height // 2), (i + 1, self.height // 2), self.color[j], 1)

        cv2.imshow(label, self.plot)
        cv2.waitKey(1)