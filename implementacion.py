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
            pass
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
                    pass

                elif op=="2":
                    if len(sistema.estudios) ==0:
                        print("No hay estudios cargados")
                    else:
                        estudio=sistema.estudios[-1]# se toma el ultimo estudio
                        estudio.zoom()

                elif op=="3":
                    pass
                elif op=="4":
                    pass
                elif op=="5":
                    pass
                elif op=="6":
                    break
                else:
                    print("Opcion invalida, intete de nuevo...")

        elif opc=="4":
            pass
        elif opc=="5":
            pass
        elif opc=="6":
            print("A salido del programa...")
            break
        else:
            print("Opcion invalida intetara de nuevo")

if __name__ =="__main__":
    menu()
