from ultralytics import YOLO

def main():
    #Load a model
    model = YOLO("yolo11m.yaml")  #build a new model from YAML
    model = YOLO("yolo11m.pt")  #load a pretrained model (recommended for training)
    model = YOLO("yolo11m.yaml").load("yolo11m.pt")  #build from YAML and transfer weights

    #Train the model
    results = model.train(data="coco8.yaml", epochs=1)
    #Uncomment the above if one wants to train the model. (WARNING: Expensive operation!)
    #Ultralytics recommends 100 epochs. This is 1 because I already trained it when running this multiple times.
    #If first time running, I think you should run 100. I think...
    """At least one epoch is required to run, or a properly configured YAML file is required,
    otherwise one might not get proper labels on the image."""

    #Test the model
    results = model("bus.jpg")  #predict on an image
    #results = model("car.jpg")

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        #The above is unused, but the code was provided by Ultralytics. Could possibly work with matplotlib?
        result.show()
        result.save("results.jpg")

if __name__ == "__main__":
    main()