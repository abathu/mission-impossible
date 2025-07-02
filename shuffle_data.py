import os
import json # Import the json library to handle JSONL files

from numpy.random import default_rng

def __perturb_shuffle_nondeterministic(sent_text, rng):
    # Split the sentence into a list of words
    words = sent_text.split()
    
    # Shuffle the list of words in-place
    rng.shuffle(words)
    
    # Join the shuffled words back into a sentence
    shuffled_sent = " ".join(words)
    
    return shuffled_sent


def shuffle_data(input_file, output_file):
    """
    Reads a JSONL file line by line, shuffles the 'text' content of each line,
    and saves the shuffled data to a new JSONL file.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSONL file.
    """
    rng = default_rng() # Initialize the random number generator
    shuffled_lines = []

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    if "text" in data:
                        data["text"] = __perturb_shuffle_nondeterministic(data["text"], rng)
                    shuffled_lines.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} in file: {input_file}, line: {line.strip()}")
                    continue
    except IOError as e:
        print(f"Error opening or reading input file {input_file}: {e}")
        return # Exit the function if the input file can't be read
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for shuffled_line in shuffled_lines:
                outfile.write(shuffled_line + '\n')
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")
        return

    print(f"Shuffled data saved to: {output_file}")


if __name__ == "__main__":
    file_path = "/home/s2678328/100M"
    output_dir = "/home/s2678328/shuffled_100M" # Define a separate output directory
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(file_path)

    file_prefix = "shuffled_" # Changed file_predix to file_prefix for clarity
    
    # Filter for JSONL files
    jsonl_files = [f for f in files if f.endswith('.jsonl')] # Assuming .jsonl extension

    if not jsonl_files:
        print(f"No .jsonl files found in {file_path}")
    else:
        for file in jsonl_files: # Iterate over filtered files
            print("========================================")
            print(f"Processing {file}...")
            
            input_file = os.path.join(file_path, file)
            
            # Construct the output file path in the dedicated output directory
            output_file = os.path.join(output_dir, file_prefix + file)
            
            shuffle_data(input_file, output_file)
            
            print(f"Done processing {file}.")
            print(f"Output file: {output_file}")
            print("========================================")
            print("\n")