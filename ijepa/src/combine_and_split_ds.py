import json
import glob
import random

def combine_and_split_jsonl_files(output_train_file, output_test_file, output_combined_file, test_ratio=0.1):
    # Find all JSONL files in the current directory
    jsonl_files = glob.glob("generated_completions_*.jsonl")
    combined_entries = []

    # Combine entries from all JSONL files and add model name
    for file_path in jsonl_files:
        model_name = file_path.split('_')[2].replace(".jsonl", "").replace("_", ":")
        with open(file_path, 'r') as infile:
            for line in infile:
                entry = json.loads(line.strip())
                entry['model_name'] = model_name
                combined_entries.append(entry)

    # Shuffle the combined entries
    random.shuffle(combined_entries)

    # Split the combined entries into train and test sets
    split_index = int(len(combined_entries) * (1 - test_ratio))
    train_entries = combined_entries[:split_index]
    test_entries = combined_entries[split_index:]

    # Write combined entries to the combined output file
    with open(output_combined_file, 'w') as combined_file:
        for entry in combined_entries:
            json.dump(entry, combined_file)
            combined_file.write('\n')

    # Write train entries to the train output file
    with open(output_train_file, 'w') as train_file:
        for entry in train_entries:
            json.dump(entry, train_file)
            train_file.write('\n')

    # Write test entries to the test output file
    with open(output_test_file, 'w') as test_file:
        for entry in test_entries:
            json.dump(entry, test_file)
            test_file.write('\n')

    print(f"Combined JSONL file created at {output_combined_file}")
    print(f"Training JSONL file created at {output_train_file}")
    print(f"Testing JSONL file created at {output_test_file}")

# Call the function to combine and split JSONL files
combine_and_split_jsonl_files('train_completions.jsonl', 'test_completions.jsonl', 'combined_completions.jsonl')