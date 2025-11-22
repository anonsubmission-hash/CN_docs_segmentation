import json
import os
import sys
import re # Import the regular expression module

def reconstruct_files_with_tags():
    """
    Processes a merged segmentation results JSON file to reconstruct original 
    text files. It first removes pre-existing "<line #>" tags from the source 
    files and then wraps the cleaned content with the correct segmentation labels.
    """
    # --- Configuration ---
    json_input_file = 'sample_data/itemize_logs/merged_segmentation_results.json'
    original_txt_dir = 'sample_data/txt_files'
    output_dir = 'sample_data/txt_files_tagged'

    # --- Pre-run Checks ---
    if not os.path.exists(json_input_file):
        print(f"Error: JSON input file not found at the specified path: {json_input_file}")
        sys.exit(1)
        
    if not os.path.isdir(original_txt_dir):
        print(f"Error: Directory with original text files not found: {original_txt_dir}")
        sys.exit(1)

    # --- Main Execution ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created or already exists: {output_dir}")
    except OSError as e:
        print(f"Fatal Error: Could not create output directory {output_dir}: {e}")
        sys.exit(1)

    processed_count = 0
    
    try:
        with open(json_input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_input_file}': {e}")
        sys.exit(1)
    
    results = data.get("results", {})
    if not results:
        print("Warning: JSON file does not contain a 'results' key or it is empty.")
        return

    # Iterate over each file's data in the results dictionary
    for filename, file_data in results.items():
        try:
            segments = file_data.get('segments', [])
            if not segments:
                print(f"Warning: No segments found for '{filename}'. Skipping.")
                continue

            original_file_path = os.path.join(original_txt_dir, filename)
            if not os.path.exists(original_file_path):
                print(f"Warning: Original file not found for '{filename}'. Skipping.")
                continue

            with open(original_file_path, 'r', encoding='utf-8') as f_orig:
                original_lines = f_orig.readlines()

            line_tags = {}
            for segment in segments:
                label = segment.get('label')
                line_numbers = segment.get('line_numbers', [])
                if not label or not line_numbers:
                    continue
                
                if len(line_numbers) == 1:
                    line_tags[line_numbers[0]] = {'label': label, 'pos': 'single'}
                elif len(line_numbers) > 1:
                    line_tags[line_numbers[0]] = {'label': label, 'pos': 'start'}
                    line_tags[line_numbers[-1]] = {'label': label, 'pos': 'end'}

            reconstructed_lines = []
            for i, current_line in enumerate(original_lines):
                line_num = i + 1
                tag_info = line_tags.get(line_num)
                
                line_content = current_line.rstrip()
                
                # **MODIFIED LINE**: Remove the <line X> tags from the beginning of the line
                # This regex finds a pattern at the start of the line (^) matching "<line #>"
                # and any following space, then replaces it with an empty string.
                processed_line = re.sub(r'^<line \d+>\s*', '', line_content)

                if tag_info:
                    label = tag_info['label']
                    pos = tag_info['pos']
                    if pos == 'single':
                        reconstructed_lines.append(f"<{label}>{processed_line}</{label}>")
                    elif pos == 'start':
                        reconstructed_lines.append(f"<{label}>{processed_line}")
                    elif pos == 'end':
                        reconstructed_lines.append(f"{processed_line}</{label}>")
                else:
                    reconstructed_lines.append(processed_line)
            
            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                f_out.write('\n'.join(reconstructed_lines))
            
            processed_count += 1
            print(f"Successfully processed and tagged: {filename}")

        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}")
    
    print(f"\nProcessing complete. Total files reconstructed: {processed_count}")


if __name__ == '__main__':
    reconstruct_files_with_tags()