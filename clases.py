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
        print(f"Volumen con {len(slices)} cortes.Forma: {self.volumen.shape}")


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


    def segmentar(self, corte_index=None):
        img_vol = self.volumen

        # Si el volumen tiene dimensiones extra, se colapsa
        while img_vol is not None and img_vol.ndim > 3:
            img_vol = img_vol[0]

        self.volumen = img_vol

        if corte_index is None:
            corte_index = self.volumen.shape[0] // 2

        # extraer el corte 2D
        img = self.volumen[corte_index, :, :]
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        # Selección tipo de umbral
        print("Tipos de binarización disponibles: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv', 'otsu', 'triangle'")
        tipo = input("Ingresa el tipo de binarización: ").strip().lower()

        if tipo == 'otsu':
            # Umbral automático
            ret, mask = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            if tipo == 'binary':
                flag = cv2.THRESH_BINARY
            elif tipo == 'binary_inv':
                flag = cv2.THRESH_BINARY_INV
            elif tipo == 'trunc':
                flag = cv2.THRESH_TRUNC
            elif tipo == 'tozero':
                flag = cv2.THRESH_TOZERO
            elif tipo == 'tozero_inv':
                flag = cv2.THRESH_TOZERO_INV
            elif tipo == 'triangle':
                flag = cv2.THRESH_TRIANGLE
            else:
                print("Intente de nuevo.")
                return

            # Pide el umbral solo si no es Otsu
            try:
                umbral = int(input("Ingresa el valor del umbral (0-255): "))
            except:
                print("Valor de umbral inválido.")
                return

            ret, mask = cv2.threshold(img_norm, umbral, 255, flag)

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img_norm, cmap='gray')
        plt.title(f"Corte {corte_index} (original)")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Segmentación ({tipo})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        nombre = input("Nombre para guardar la imagen modificada: ").strip()
        if not nombre.lower().endswith(".png"):
            nombre += ".png"
        cv2.imwrite(nombre, mask)
        print(f"Imagen guardada como '{nombre}'")


    def morfologia(self, corte_index=None):
        vol = self.volumen
        while vol is not None and vol.ndim > 3:
            vol = vol[0]
        self.volumen = vol

        if corte_index is None:
            corte_index = self.volumen.shape[0] // 2

        try:
            img = self.volumen[corte_index, :, :].astype(np.float32)
        except Exception as e:
            print("Error al extraer el corte:", e)
            print("Forma del volumen:", self.volumen.shape)
            return

        # Normalizar a uint8
        img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)


        plt.figure(figsize=(5,5))
        plt.imshow(img_norm, cmap='gray')
        plt.axis('off')
        plt.show()

        print("Transformaciones morfolóficas disponibles: 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'")
        op = input("Ingresa la transformación morfológica: ").strip().lower()
        try:
            k = int(input("Ingresa el tamaño del kernel (3, 5, 7, etc): "))
        except:
            print("Tamaño inválido. Se usará 3")
            k = 3
        if k <= 0: k = 3

        # Crear kernel y aplicar filtro
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))



        if op == 'erode':
            res = cv2.erode(img_norm, kernel, iterations=1)
        elif op == 'dilate':
            res = cv2.dilate(img_norm, kernel, iterations=1)
        elif op == 'open':
            res = cv2.morphologyEx(img_norm, cv2.MORPH_OPEN, kernel)
        elif op == 'close':
            res = cv2.morphologyEx(img_norm, cv2.MORPH_CLOSE, kernel)
        elif op == 'gradient':
            res = cv2.morphologyEx(img_norm, cv2.MORPH_GRADIENT, kernel)
        elif op == 'tophat':
            res = cv2.morphologyEx(img_norm, cv2.MORPH_TOPHAT, kernel)
        elif op == 'blackhat':
            res = cv2.morphologyEx(img_norm, cv2.MORPH_BLACKHAT, kernel)
        else:
            print("Intente de nuevo")
            return


        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img_norm, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(res, cmap='gray')
        plt.title(f"Resultado ({op}, k={k})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


        nombre = input("Ingresa el nombre de la imagen a guardar: ").strip()


        if not nombre.lower().endswith(".png"):
            nombre += ".png"
        cv2.imwrite(nombre,res)
        print(f"Imagen guardada como '{nombre}'")



