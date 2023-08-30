# Open the file for reading
source = "/home/martin/cv4e/pp_segmentation/runs/segment/pred_resol=1920_epochs=70_iou=0.22/labels/Pn_calliantha_20230523_0DB2_1.txt"

with open(source, "r") as file:
    # Read all lines from the file
    lines = file.readlines()

    # Process each line
    data = []
    for line in lines:
        # Split the lines into individual values using space as the seperator
        values = line.strip().split()
        
        # Convert values 
        processed_values = [values[0]] + [int(len(values[1:])/2)] + [float(v) for v in values[1:]]

        # Append the processed values to the main data list
        data.append(processed_values)
        