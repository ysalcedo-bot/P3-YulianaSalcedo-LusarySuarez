#menu
from clases import EstudioImaginologico,SistemaEstudioImaginologico
import os

def menu():
    sistema=  SistemaEstudioImaginologico()

    while True:
        print("\n***SISTEMA DE GESTION DE ESTUDIO IMAGENOLOGICOS***\n")
        print("1- Crear nuevo estudio (Objeto)")
        print("2- Cargar carpeta DICOM y mostrar cortes 3D")
        print("3- Operaciones del estudio actual")
        print("4- Guardar estudios en archivo CSV")
        print("5 -Cargar estudios desde archivos CSV")
        print("6- Salir del programa")

        opc= input("Selecione una opción:")

        if opc == "1":
            carpeta = input ("ingresar la ruta de la carpeta DICOM: ").strip()
            est = EstudioImaginologico(carpeta)
            sistema.anexar_estudio(est)
            print("Estudio creado y anexado correctamente")

            #C:\Users\salce\OneDrive\Desktop\P3 YulianaSalcedo LusarySuarez\P3-YulianaSalcedo-LusarySuarez\T2
            #C:\Users\salce\OneDrive\Desktop\P3 YulianaSalcedo LusarySuarez\P3-YulianaSalcedo-LusarySuarez\Sarcoma
            #C:\Users\salce\OneDrive\Desktop\P3 YulianaSalcedo LusarySuarez\P3-YulianaSalcedo-LusarySuarez\PPMI

        elif opc=="2":
             if not os.path.exists(carpeta):
                print("La ruta ingresada no existe. Intente nuevamente.")
            else:
                gestor = GestorDICOM()
                gestor.cargar_carpeta(carpeta)
                gestor.mostrar_cortes()
            
        elif opc=="3":
            while True:
                print("\n*Operaciones disponibles sobre el estudio*\n")
                print("1- Ver atributos del estudio")
                print("2- Aplicar Zoom a una región")
                print("3- Realizar segmentación")
                print("4- Aplicar tranformación morfologica")
                print("5- Convertir a formato NIFTI")
                print("6- Volver al menu principal")

                op= input("Selecione una opcion:")

                if op=="1":
                    if len(sistema.estudios)==0:
                        print("No se ha creado estudios")
                    else:
                        estudio= sistema.estudios[-1]
                        print("\n-ATRIBUTOS DEL ESTUDIO ACTUAL-")
                        print(f"Study Date: {estudio.StudyDate}")
                        print(f"Study Time: {estudio.StudyTime}")
                        print(f"Study Modality: {estudio.StudyModality}")
                        print(f"Study Description: {estudio.StudyDescription}")
                        print(f"Series Time: {estudio.SeriesTime}")
                        print(f"Duracion del estudio: {estudio.Duracion}")
                        print(f"Forma del volumen recontruido: {estudio.Forma}")

                elif op=="2":
                    gestor.zoom()

                elif op=="3":
                    gestor.segmentar()
                    
                elif op=="4":
                    gestor.morfologia()

                elif op=="5":
                    if len(sistema.estudios)==0:
                        print("No se ha creado estudios")
                    else:
                        estudio= sistema.estudios[-1]
                        carpeta_salida= input("Ingresar carpeta donde desea guardar el NIFTI o Enter para 'nifti_output'): ").strip() or "nifti_output"
                        estudio.conversion_NIFTI(carpeta_salida)
                elif op=="6":
                    break
                else:
                    print("Opcion invalida, intete de nuevo...")

        elif opc=="4":
            if len(sistema.estudios) ==0:
                        print("No hay estudios cargados")
            else:
                carpeta_salida = input("Ingresar carpeta donde desea guardar los estudios o Enter para 'resultados_estudios").strip() or "resultados_estudios"
                sistema.guardar_estudio(carpeta_salida)
        elif opc=="5":
            archivo_csv = input("Ingresar ruta del archivo csv que desea guardar o Enter para 'resultados_estudios/estudios.csv").strip() or "resultados_estudios/estudios.csv"
            if not os.path.exists(archivo_csv):
                print(f"No se encontró el archivo enn la ruta:{archivo_csv}")
            else:
                sistema.cargar_csv(archivo_csv)
        elif opc=="6":
            print("A salido del programa...")
            break
        else:
            print("Opcion invalida intetara de nuevo")

if __name__ =="__main__":
    menu()
