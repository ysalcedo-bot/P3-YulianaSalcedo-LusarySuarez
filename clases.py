import cv2
import numpy as np
import dicom2nifti
import pydicom
import os
import  pandas as pd
import matplotlib.pyplot as plt

class EstudioImaginologico:
    def __init__(self, carpeta):
        self.carpeta =carpeta
        self.ruta=carpeta

        slices=[]
        for root,dirs,files in os.walk (carpeta):
            for f in files:
                if f.endswith('.dcm'):
                    path = os.path.join(root,f)
                    ds= pydicom.dcmread (path)
                    slices.append(ds)
        try:
            slices.sort(key=lambda x: float (x.ImagePositionPatient[2]))
        except:
            slices.sort(key=lambda x: x.IntanceNumber)

        self.slices = slices

        pixel_spacing = self.slices[0].PixelSpacing
        slice_thickness = self.slices[0].SliceThickness
        self.espaciado = (float(slice_thickness),float(pixel_spacing[0]), float(pixel_spacing[1]))

        self.volumen= np.stack([s.pixel_array for s in self.slices])

        #atributos
        self.StudyDate = self.slices[0].StudyDate
        self.StudyTime = self.slices[0].StudyTime
        self.StudyModality = getattr(self.slices [0], "Modality","Desconocido")
        self.StudyDescription =getattr(self.slices[0],"StudyDescription","Sin descripcion")
        self.SeriesTime = self.slices[0].SeriesTime
        self.Duracion = float(self.SeriesTime)-float(self.StudyTime)
        self.Volumen = np.stack([s.pixel_array for s in self.slices])
        self.ImagenAsociada = self.volumen
        self.Forma= self.volumen.shape

    def conversion_NIFTI (self,carpeta_salida="nifti_output"): # se encarga de convertir la cerpeta DICOM a NIFTI
        os.makedirs(carpeta_salida,exist_ok=True)
        dicom2nifti.convert_directory(self.carpeta,carpeta_salida)
        print(f"Se ha convertido a NIFTI el DICOM en: {carpeta_salida}")

###########################################################################################################################
    def zoom(self, corte_index=None):
        """Permite seleccionar manualmente un recorte (zoom) con el mouse y guardarlo."""
        if self.volumen is None:
            print("Primero carga una carpeta DICOM con .cargar_carpeta()")
            return

        if corte_index is None:
            corte_index = self.volumen.shape[0] // 2

        # Seleccionamos un corte axial
        img = self.volumen[corte_index, :, :].astype(np.float32)
        img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        # Mostrar imagen y dejar que el usuario seleccione la ROI
        #from google.colab.patches import cv2_imshow  # para mostrar en Colab
        print("Selecciona con el mouse la región que quieres recortar.")
        print("Cuando termines, presiona ENTER o ESPACIO.")
        r = cv2.selectROI("Selecciona región", img_bgr, showCrosshair=True)
        cv2.destroyAllWindows()

        if r == (0, 0, 0, 0):
            print("No se seleccionó ninguna región.")
            return

        x, y, w, h = map(int, r)
        z, py, px = self.espaciado
        ancho_mm = w * px
        alto_mm = h * py
        texto_dim = f"{ancho_mm:.1f} x {alto_mm:.1f} mm"

        # Dibujar el cuadro
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_bgr, texto_dim, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Recorte y redimensionamiento
        recorte = img_norm[y:y + h, x:x + w]
        recorte_resize = cv2.resize(recorte, (img_norm.shape[1], img_norm.shape[0]))

        # Pedir nombre de archivo
        nombre_salida = input("Ingrese el nombre con el que desea guardar la imagen: ")

        # Mostrar resultados
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Imagen original con recuadro')
        axs[1].imshow(recorte_resize, cmap='gray')
        axs[1].set_title('Región recortada y redimensionada')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Guardar
        if not nombre_salida.endswith(".png"):
            nombre_salida += ".png"
        cv2.imwrite(nombre_salida, recorte_resize)
        print(f"Imagen guardada como '{nombre_salida}'")

#ALMACENAR OBJETOS
class SistemaEstudioImaginologico:
    def __init__(self):
        self.estudios = []
    def anexar_estudio(self,estudio):
        self.estudios.append(estudio)
        print (f"Se agrego estudio: {estudio.StudyDate}-{estudio.StudyModality}")
    def guardar_estudio(self,carpeta_salida="resultados_estudios"):
        os.makedirs(carpeta_salida,exist_ok=True)
        registros=[]
        for i, estudio in enumerate(self.estudios):#guaradar en lista
            registro = {"Id":i+1,"StudyDate": estudio.StudyDate,"StudyTime": estudio.StudyTime, "StudyModality":estudio.StudyModality,  "StudyDescription":estudio.StudyDescription,"SeriesTime":estudio.SeriesTime, "Duracion":estudio.Duracion, "Volumen":estudio.Volumen, "Forma":str(estudio.Forma) }
            registros.append(registro)

            #guardar imagen
            corte_medio= estudio.ImagenAsociada[estudio.ImagenAsociada.shape[0]//2,:,:]
            corte_normal = cv2.normalize(corte_medio,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            img_path = os.path.join(carpeta_salida,f"estudio-{i+1}.png")
            cv2.imwrite(img_path,corte_normal)
        # Guaradar en csv
        df= pd.DataFrame(registros)
        df.to_csv(os.path.join(carpeta_salida,"estudios.csv"),index=False,encoding="utf-8")

        print(f"{len(self.estudios)}estudios guaradados en la carpeta'{carpeta_salida}'")
    def cargar_csv(self,archivo_csv):
        if not os.path.exists(archivo_csv):
            print("No se encontro el archivo")
            return
        df=pd.read_csv(archivo_csv)
        print("Metadatos caragdos corectamente")
        print(df)


class GestorDICOM:
    def __init__(self):
        self.volumen = None
        self.ruta = None

    def cargar_carpeta(self, carpeta):
        self.ruta = carpeta #OJO PIOJO
        slices = []

        for root, dirs, files in os.walk(carpeta):
            for f in files:
                if f.endswith('.dcm'):
                    path = os.path.join(root, f)
                    ds = pydicom.dcmread(path)
                    slices.append(ds)

    # Ordenar por posición o número
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            slices.sort(key=lambda x: x.InstanceNumber)

        # Leer espaciado
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = slices[0].SliceThickness
        self.espaciado = (float(slice_thickness),
                              float(pixel_spacing[0]),
                              float(pixel_spacing[1]))

        # Crear volumen
        self.volumen = np.stack([s.pixel_array for s in slices], axis=0)
        print(f"Volumen con {len(slices)} cortes.")


    def mostrar_cortes(self):
        corte_axial = self.volumen[self.volumen.shape[0] // 2, :, :]
        corte_coronal = self.volumen[:, self.volumen.shape[1] // 2, :]
        corte_sagital = self.volumen[:, :, self.volumen.shape[2] // 2]

        z, y, x = self.espaciado

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(corte_axial, cmap='gray')
        axs[0].set_title('Corte Axial')
        axs[1].imshow(corte_coronal, cmap='gray')
        axs[1].set_title('Corte Coronal')
        axs[2].imshow(corte_sagital, cmap='gray')
        axs[2].set_title('Corte Sagital')

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

     def zoom(self, corte_index=None):
        # Si no se especifica corte, se usa el central
        if corte_index is None:
            corte_index = self.volumen.shape[0] // 2

        # Seleccionar el corte y normalizar a [0,255]
        img = self.volumen[corte_index, :, :].astype(np.float32)
        img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

        # Convertir a BGR para dibujar a color
        img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        # Mostrar imagen con ejes como referencia
        plt.figure(figsize=(6, 6))
        plt.imshow(img_norm, cmap='gray')
        plt.xlabel("x (horizontal)")
        plt.ylabel("y (vertical)")
        plt.show()

        try:
            x1 = int(input("Ingresa coordenada X inicial: "))
            x2 = int(input("Ingresa coordenada X final: "))
            y1 = int(input("Ingresa coordenada Y inicial: "))
            y2 = int(input("Ingresa coordenada Y final: "))
        except:
            print("Coordenadas no válidas.")
            return

        # Calcular dimensiones en mm
        z, py, px = self.espaciado
        ancho_mm = (x2 - x1) * px
        alto_mm = (y2 - y1) * py
        texto_dim = f"{ancho_mm:.1f} x {alto_mm:.1f} mm"

        # Dibujar cuadro con medidas
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, texto_dim, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Recorte como tal
        recorte = img_norm[y1:y2, x1:x2]
        recorte_resize = cv2.resize(recorte, (img_norm.shape[1], img_norm.shape[0]))

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Imagen original")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(recorte_resize, cmap='gray')
        plt.title("Región recortada")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


        nombre = input("Ingresa el nombre del archivo para guardar: ").strip()
        if not nombre.lower().endswith(".png"):
            nombre += ".png"

        cv2.imwrite(nombre, recorte_resize)
        print(f"Imagen guardada como '{nombre}'")



