import time

class Stepper28BYJ_PCA:
    # Последовательность для полушагового режима (8 шагов)
    __halfStepping = [
        [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
        [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]
    ]

    def __init__(self, pca_driver, control_pins_list):
        """
        Инициализация двигателя.
        pca_driver: активный объект Adafruit_PCA9685.
        control_pins_list: список из 4 номеров каналов на PCA9685 (например, [0, 1, 2, 3]).
        """
        self.pca = pca_driver
        self.controlPins = control_pins_list
        self.step_sequence = self.__halfStepping
        self.step_count = len(self.step_sequence)
    
    def _set_pin(self, pin_channel, value):
        """Включает или выключает канал на PCA9685."""
        duty_cycle = 4095 if value == 1 else 0
        self.pca.set_pwm(pin_channel, 0, duty_cycle)

    def _perform_step(self, step_idx):
        """Выполняет один шаг в последовательности."""
        for pin_idx in range(4):
            pin_channel = self.controlPins[pin_idx]
            pin_value = self.step_sequence[step_idx][pin_idx]
            self._set_pin(pin_channel, pin_value)
        time.sleep(0.001)

    def cwStepping(self, steps):
        """Вращение по часовой стрелке."""
        for _ in range(steps):
            for step in range(self.step_count):
                self._perform_step(step)

    def ccwStepping(self, steps):
        """Вращение против часовой стрелки."""
        for _ in range(steps):
            for step in range(self.step_count - 1, -1, -1):
                self._perform_step(step)

    def cleanup(self):
        """Отключает все пины двигателя."""
        for pin_channel in self.controlPins:
            self._set_pin(pin_channel, 0)