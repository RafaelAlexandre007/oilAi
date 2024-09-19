from ultralytics import YOLO
import torch

# Verifica se há GPU disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo YOLOv8 (você pode usar diferentes variantes: 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8n.pt').to(device)  # 'n' é para o modelo nano, mais leve

# Iniciar o treinamento
# Iniciar o treinamento
model.train(
    data='/Users/rafael/Documents/Projetos/oilAi/data.yaml', 
    epochs=5, 
    imgsz=640, 
    batch=3,
    patience=5,  # Interrompe o treinamento se não houver melhoria na métrica de validação após n épocas
    device=device
)
