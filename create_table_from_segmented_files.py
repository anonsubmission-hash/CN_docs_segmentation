import os
import csv
import re

def create_segment_csv(input_folder, output_csv_file):
    """
    Processes segmented .txt files from a folder into a structured CSV file.

    Args:
        input_folder (str): The path to the folder containing the .txt files.
        output_csv_file (str): The path where the output CSV file will be saved.
    """
    # Initialize counters for unique IDs
    global_segment_id = 1
    document_id = 1

    # Define the pattern for segments to be omitted
    # Matches strings that *only* contain this pattern, ignoring leading/trailing whitespace.
    omission_pattern = re.compile(r'^Minimal Segmentation Layer Determined: L\d+$')

    try:
        # Get a sorted list of .txt files to ensure consistent processing order
        txt_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])

        if not txt_files:
            print(f"No .txt files found in '{input_folder}'.")
            return

        # Open the CSV file for writing
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header row, including the new 'Paragraph_content' column
            csv_writer.writerow(['ID', 'DocumentID', 'order', 'Paragraph_content', 'R_1', 'R_2'])

            # Process each text file
            for filename in txt_files:
                file_path = os.path.join(input_folder, filename)
                print(f"Processing file: {filename} (DocumentID: {document_id})")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Segments are separated by one or more empty lines
                segments = re.split(r'\n\s*\n', content)
                segment_order_in_doc = 1

                # Process each segment
                for segment in segments:
                    # Clean up the segment by removing leading/trailing whitespace
                    cleaned_segment = segment.strip()

                    # Skip empty segments that might result from splitting
                    if not cleaned_segment:
                        continue
                    
                    # Check if the segment matches the omission pattern
                    if omission_pattern.fullmatch(cleaned_segment):
                        print(f"  - Omitting segment: '{cleaned_segment}'")
                        continue

                    # Write the data to the CSV row
                    # Add the actual text content to the 'Paragraph_content' column
                    csv_writer.writerow([
                        global_segment_id,
                        document_id,
                        segment_order_in_doc,
                        cleaned_segment, # Paragraph_content
                        '',              # R_1
                        ''               # R_2
                    ])

                    # Increment counters
                    global_segment_id += 1
                    segment_order_in_doc += 1

                # Move to the next DocumentID for the next file
                document_id += 1
        
        print(f"\nProcessing complete. CSV file saved to '{output_csv_file}'")

    except FileNotFoundError:
        print(f"Error: The folder '{input_folder}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':

    input_directory = 'sample_data/segmentation_results'  # The folder with your .txt files
    output_csv_path = 'sample_data/data.csv' # The name of the resulting CSV file

    # Run the main function
    create_segment_csv(input_directory, output_csv_path)
