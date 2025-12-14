from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=r"D:\test\Find-PaperBalls-1\Find-PaperBalls-2\data.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=8, 
        workers=2
    )

if __name__ == '__main__':
    main()