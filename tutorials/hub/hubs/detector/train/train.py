if __name__ == "__main__":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt", task="detect")
        model.train(
            data="/home/lhj/ws/release/waffle/docs/tutorials/hub/datasets/sample_dataset/exports/YOLO/data.yaml",
            epochs=50,
            batch=64,
            imgsz=[640, 640],
            lr0=0.01,
            lrf=0.01,
            rect=True,
            device="0",
            workers=2,
            seed=0,
            verbose=True,
            project="hubs/detector",
            name="artifacts",
        )