from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11l.pt")

    model.train(data="data.yaml", imgsz=640, device=0,
                batch=8, epochs=100, workers=2)
