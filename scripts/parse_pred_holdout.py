import os 
import pandas as pd

# Open the file for reading
dir_path = "/Users/mmarzi/MLprojects/cv4e/evaluation/evaluation_data/pred_resol=1920_epochs=70_iou=0.2_holdout/labels"
file_names = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

df_holdout = pd.DataFrame(columns=["class", "surface", "confidence", "image_name", "img_inst_id"])

for f_name in file_names:

    with open(f_name, "r") as file:
        # Read all lines from the file
        lines = file.readlines()
        
        # Process each line
        line_data = []

        for count, inst in enumerate(lines):
            # Split the lines into individual values using space as the seperator
            instance = inst.strip().split()
            
            img_class = instance[0]
            surface = len(instance[1:-1])/2
            conf = instance[-1]
            img_name = os.path.basename(f_name)
            
            # Add to df
            df_predicted.loc[len(df_predicted)] = [img_class, surface, conf, img_name, count]

