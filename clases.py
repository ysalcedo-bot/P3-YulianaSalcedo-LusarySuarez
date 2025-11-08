import cv2
import numpy as np
import dicom2nifti
import pydicom
import os
import  pandas as pd


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
        self.Volumen = np.stack([s.pixel_array for s in self.slices])
        self.Forma= self.volumen.shape

    def conversion_NIFTI (self,carpeta_salida="nifti_output"): # se encarga de convertir la cerpeta DICOM a NIFTI
        os.makedirs(carpeta_salida,exist_ok=True)
        dicom2nifti.convert_directory(self.carpeta,carpeta_salida)
        print(f"Se ha convertido a NIFTI el DICOM en: {carpeta_salida}")

#ALMACENAR OBJETOS
class SistemaEstudioImaginologico:
    def __init__(self):
        self.estudios = []
    def anexar_estudio(self,estudio):
        self.estudios.append(estudio)
        print (f"Se agrego estudio: {estudio.StudyDate}-{estudio.Modality}")
    def guardar_estudio(self,carpeta_salida="resultados_estudios"):
        os.makedirs(carpeta_salida,exit_ok=True)
        registros=[]
        for i, estudio in enumerate(self.estudios):#guaradar en lista
            registro = {"Id":i+1,"StudyDate": estudio.StudyDate,"StudyTime": estudio.StudyTime, "StudyModality":estudio.StudyModality,  "StudyDescription":estudio.StudyDescription,"SeriesTime":estudio.SeriesTime, "Duracion":estudio.Duracion, "Volumen":estudio.Volumen, "Forma":str(estudio.Forma) }
            registros.append(registro)

            #guardar imagen
            corte_medio= estudio.ImagenAsociada[estudio.ImagenAsociada.shape[0]//2,:,:]
            corte_normal = cv2.normalize(corte_medio,None,0,255,cv2.NORM_MINIMAX).astype(np.uint8)
            img_path = os.path.join(carpeta_salida,f"estudio-{i+1}.png")
            cv2.imwrite(img_path,corte_normal)
        # Guaradar en csv
        df= pd.Dataframe(registros)
        df.to_csv(os.path.join(carpeta_salida,"estudios.csv"),index=False,encoding="utf-8")

        print(f"{len(self.estudios)}estudios guaradados en la carpeta'{carpeta_salida}'")
    def cargar_csv(self,archivo_csv):
        if not os.path.exists(archivo_csv):
            print("No se encontro el archivo")
            return
        df=pd.read_csv(archivo_csv)
        print("Metadatos caragdos corectamente")
        print(df)
