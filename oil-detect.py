from ultralytics import YOLO
import cv2

# Função para verificar câmeras disponíveis
def find_available_cameras(max_cameras=5):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Listar câmeras disponíveis
available_cameras = find_available_cameras()

if not available_cameras:
    print("Nenhuma câmera disponível.")
else:
    print("Câmeras disponíveis:")
    for idx, cam in enumerate(available_cameras):
        print(f"{idx}: Camera {cam}")

    # Solicitar ao usuário que selecione uma câmera
    cam_idx = int(input(f"Escolha a câmera (0 a {len(available_cameras)-1}): "))
    selected_camera = available_cameras[cam_idx]

    # Carregar o modelo YOLOv8 com pesos da pasta weights
    model_path = '/Users/rafael/Documents/Projetos/oilAi/runs/detect/train/weights/best.pt'  # Caminho para o arquivo de pesos
    model = YOLO(model_path)
    # Inicializar a câmera escolhidas
    cap = cv2.VideoCapture(selected_camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fazer predição usando YOLOv8
        results = model(frame)

        # Desenhar as predições no frame
        annotated_frame = results[0].plot()

        # Exibir o frame com anotações
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()
