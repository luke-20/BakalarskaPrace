import matplotlib.pyplot as plt

from src.scaler import Scaler
from src.matlab_handler import MatlabData


class DataVisualizer:
    def __init__(self, file_path):
        self.matlab_data = MatlabData(file_path=file_path)
        self.scaler = Scaler()

    def plot_napeti(self):
        plt.figure(1)
        plt.plot(self.matlab_data.dq_napeti_filtrovane_4)

        plt.title(f"dq napeti filtrovane 4 {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("U [V]")
        plt.show()

    def plot_napeti_scaled(self):
        plt.figure(1)
        plt.plot(self.scaler.scale_dataset(self.matlab_data.dq_napeti_filtrovane_4))

        plt.title(f"dq napeti filtrovane 4 - kvantizovana data {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("U [V]")
        plt.show()

    def plot_napeti_scaled_per_column(self):
        plt.figure(1)
        plt.plot(self.scaler.scale_2d_array_per_column(self.matlab_data.dq_napeti_filtrovane_4))

        plt.title(f"dq napeti filtrovane 4 - kvantizovana data podle kazde veliciny zvlast {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("U [V]")
        plt.show()

    def plot_proudy(self):
        plt.figure(2)
        plt.plot(self.matlab_data.dq_proudy_filtrovane_4)

        plt.title(f"dq proudy filtrovane 4 {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("I [A]")
        plt.show()

    def plot_proudy_scaled(self):
        plt.figure(2)
        plt.plot(self.scaler.scale_dataset(self.matlab_data.dq_proudy_filtrovane_4))

        plt.title(f"dq proudy filtrovane 4 - kvantizovana data {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("I [A]")
        plt.show()

    def plot_proudy_scaled_per_column(self):
        plt.figure(2)
        plt.plot(self.scaler.scale_2d_array_per_column(self.matlab_data.dq_proudy_filtrovane_4))

        plt.title(f"dq proudy filtrovane 4 - kvantizovana data podle kazde veliciny zvlast {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("I [A]")
        plt.show()

    def plot_proudy_240(self):
        plt.figure(3)
        plt.plot(self.matlab_data.ab_proudy_filtrovane_240)

        plt.title(f"ab proudy filtrovane 240 {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("I [A]")
        plt.show()

    def plot_oscilace(self):
        plt.plot(self.matlab_data.dq_oscilace_filtrovane_16)

        plt.title(f"dq oscilace filtrovane 16 {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("Amplituda [-]")
        plt.show()

    def plot_oscilace_scaled(self):
        plt.plot(self.scaler.scale_dataset(self.matlab_data.dq_oscilace_filtrovane_16))

        plt.title(f"dq oscilace filtrovane 16 - kvantizovana data {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("Amplituda [-]")
        plt.show()

    def plot_oscilace_scaled_per_column(self):
        plt.plot(self.scaler.scale_2d_array_per_column(self.matlab_data.dq_oscilace_filtrovane_16))

        plt.title(f"dq oscilace filtrovane 16 - kvantizovana data podle kazde veliciny zvlast {self.matlab_data.category_text}")
        plt.xlabel("samples [-]")
        plt.ylabel("Amplituda [-]")
        plt.show()
