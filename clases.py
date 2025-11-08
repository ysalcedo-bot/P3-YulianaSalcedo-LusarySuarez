import cv2
import numpy as np
import dicom2nifti
import pydicom
import os
import  pickle

class EstudioImaginologico:
    def __init__(self, carpeta):
        self.carpeta =carpeta
        self.slices = [pydicom.dcmread(os.path.join(carpeta, s))for s in os.listdir(carpeta)]
        self.slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        self.volumen= np.stack([s.pixel_array for s in self.slices])

        #atributos
        self.StudyDate = self.slices[0].StudyDate
        self.StudyTime = self.slices[0].StudyTime
        self.StudyModality = self.slices [0].StudyModality
        self.StudyDescription =getattr(self.slices[0],"StudyDescription","Sin descripcion")
        self.SeriesTime = self.slices[0].SeriesTime
        self.Duracion = float(self.SeriesTime)-float(self.StudyTime)
        self.volumen = np.stack([s.pixel_array for s in self.slices])
        self.Forma= self.volumen.shape

 
