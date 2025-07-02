import os
import json
import argparse

def convert_to_jsonl(input_folder, output_folder, suffix):
    os.makedirs(output_folder, exist_ok=True)
    
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(subdir, file)
                relative_subdir = os.path.relpath(subdir, input_folder)
                output_subdir = os.path.join(output_folder, relative_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                # Modify the filename: AAA.txt -> AAA_train.jsonl
                base_filename = os.path.splitext(file)[0]
                output_filename = f"{base_filename}_{suffix}.jsonl"
                output_path = os.path.join(output_subdir, output_filename)

                print(f"Processing {input_path} -> {output_path}")
                
                with open(input_path, 'r', encoding='utf-8') as infile, \
                     open(output_path, 'w', encoding='utf-8') as outfile:
                    
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            json.dump({"text": line}, outfile, ensure_ascii=False)
                            outfile.write('\n')

if __name__ == "__main__":
    # usage example:
    #    python /home/s2678328/mission-impossible_fork/data/convert_to_jsonl.py  --input_folder /home/s2678328/BabyLM_dataset/translated_dev_file/Russain   --output_folder /home/s2678328/100M  --suffix validation
    #    Processing 

    parser = argparse.ArgumentParser(description="Convert .txt files to .jsonl format with suffix.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing .txt files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path where .jsonl files will be saved")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix to add to the output filenames")

    args = parser.parse_args()

    convert_to_jsonl(args.input_folder, args.output_folder, args.suffix)