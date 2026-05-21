import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

metadata_path = os.path.join(script_dir, '../data/external_val/retina_age_40_70.csv')
output_path = os.path.join(script_dir, '../data/external_val/retina_age_processed_crop.csv')
image_dir = os.path.join(script_dir, '../data/external_val/processed_crop')
image_dir = os.path.abspath(image_dir)

metadata_df = pd.read_csv(metadata_path)
metadata_df = metadata_df.rename(columns={'patient_age': 'Age'})
metadata_df['Path'] = metadata_df['image_id'].apply(lambda x: os.path.join(image_dir, x + '.png'))
metadata_df.to_csv(output_path, index=False)

print(f"Data updated in {output_path} (added 'Path' column).")