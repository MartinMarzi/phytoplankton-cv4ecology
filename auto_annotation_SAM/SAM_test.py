from ultralytics import SAM

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
model('/mnt/ssd-cluster/martin/dataset/Bounding Box 1/Pn_calliantha_20230523_0DB2_1.jpg', bboxes=[1281.79,
                957.45,
                1244.28,
                239.35])

# # Run inference with points prompt
# model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])