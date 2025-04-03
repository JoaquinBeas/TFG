import sys
import os
import os
import sys
from src.utils.data_loader import get_mnist_dataloaders
from scripts import train_teacher, generate_synthetic, label_synthetic_data, train_student

def main(model_type):
    
    

    # # 1. Entrenar el modelo teacher
    print("Iniciando entrenamiento del Teacher...")
    train_loader, test_loader = get_mnist_dataloaders()
    train_teacher.train_model(train_loader)
    
    # # 2. Generar imágenes sintéticas con el teacher entrenado
    print("Generando imágenes sintéticas...")
    generate_synthetic.generate_synthetic_data()
    
    # 3. Etiquetar las imágenes sintéticas
    print("Etiquetando imágenes sintéticas...")
    label_synthetic_data.label_synthetic_data() 
    
    # 4. Entrenar el modelo student
    # Para forzar el uso del modelo MNIST_STUDENT_GUIDED (por ejemplo) llamar al main con el
    print("Entrenando el modelo Student...")
    train_student.train_student(model_type)

if __name__ == "__main__":
    model_type_student = train_student.StudentModelType.MNIST_STUDENT_COPY
    main(model_type_student)
