import scipy.io as sio
import numpy as np


class MatlabData:
    FAULT: bool = False

    def __init__(self, file_path):
        self.file_path: str = file_path
        self.FAULT = self.select_category()
        self.dq_oscilace_filtrovane_16, self.dq_napeti_filtrovane_4, self.dq_proudy_filtrovane_4, self.ab_proudy_filtrovane_240, self.speed = self.preprocess_mat()

        self.category_text = "- fault" if self.FAULT else "- normal"

    def load_mat_file(self):
        return sio.loadmat(self.file_path)

    def select_category(self) -> bool:
        return "fault" in self.file_path.lower()

    def preprocess_mat(self):
        unpreprocessed_mat_file = self.load_mat_file()
        # del unpreprocessed_mat_file['AB_Prudy_Filtrovanie240']
        ab_proudy_filtrovane_240 = np.swapaxes(unpreprocessed_mat_file['AB_Prudy_Filtrovanie240'], 0, 1)
        dq_oscilace_filtrovane_16 = np.swapaxes(unpreprocessed_mat_file['DQ_oscilacie_Filtrovanie16'], 0, 1)

        dq_napeti_filtrovane_4 = np.swapaxes(unpreprocessed_mat_file['DQ_napatia_Filtrovanie4'], 0, 1)
        dq_proudy_filtrovane_4 = np.swapaxes(unpreprocessed_mat_file['DQ_prudy_Filtrovanie4'], 0, 1)
        speed = np.swapaxes(unpreprocessed_mat_file['Speed'], 0, 1)

        return dq_oscilace_filtrovane_16, dq_napeti_filtrovane_4, dq_proudy_filtrovane_4, ab_proudy_filtrovane_240, speed



