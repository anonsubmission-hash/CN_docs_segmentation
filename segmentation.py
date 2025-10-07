import os
import re
import glob
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm  # Import the tqdm library for the progress bar

# --- TUNABLE PARAMETERS ---
# Adjust these thresholds to change the "fineness" of the segmentation.
CONFIG = {
    # --- Chinese Action Keywords ---
    "CHINESE_ACTION_WORDS": {
        "必须", "应", "须", "可以", "可", "建议", "宜",
        "不得", "禁止", "严禁", "立即", "紧急"
    },
    
    # --- Scoring Weights ---
    # These weights control the heuristic for deciding the best segmentation layer.
    
    # WHAT IT MEANS: Controls the importance of finding unique "action words" in a block.
    # A higher value gives a higher score to blocks containing instructions.
    # INCREASE: Favors more granular, action-oriented segments.
    # DECREASE: Makes action words less important, relying more on length and structure.
    "COLOR_DIVERSITY_WEIGHT": 3.0,
    
    # WHAT IT MEANS: Controls the importance of a block's character count.
    # INCREASE: Favors longer, more substantial blocks of text, leading to broader segments.
    # DECREASE: Makes block length less important, allowing shorter, more focused blocks to score well.
    "LENGTH_WEIGHT": 0.5,
    
    # WHAT IT MEANS: Applies a penalty to deeper structural layers (L3, L4, etc.). This creates a bias
    # for segmenting at shallower layers (L1, L2).
    # INCREASE: Makes it much harder for deeper layers to be chosen, forcing broader, high-level segmentation.
    # DECREASE: Makes it easier for deeper, more granular layers (like L3) to be chosen if their content is good.
    "LAYER_PENALTY_FACTOR": 1.5,
    
    # --- Thresholds ---
    # These values set the minimum requirements for scoring and selection.
    
    # WHAT IT MEANS: The minimum number of characters a block must have to even be considered for scoring.
    # Any block shorter than this is immediately ignored.
    # INCREASE: Ignores more (and longer) short blocks.
    # DECREASE: Allows very short blocks to be part of the analysis.
    "MIN_LENGTH_THRESHOLD": 15,
    
    # WHAT IT MEANS: The "passing grade" a layer's blocks must achieve on average to be chosen
    # as the document's main segmentation layer.
    # INCREASE: Makes it harder for any layer to be chosen, causing the script to fall back to the DEFAULT_LAYER more often.
    # DECREASE: Makes it easier for layers to be chosen, leading to more content-driven segmentation.
    "MINIMAL_LAYER_SCORE_THRESHOLD": 0.4,
    
    # WHAT IT MEANS: If no layer meets the score threshold, this layer is used for the informational
    # header in the output files. A value of 1 means the broadest possible fallback.
    "DEFAULT_LAYER": 1
}
# --- END OF TUNABLE PARAMETERS ---

def calculate_block_score(text, layer_num, config):
    """Calculates a "quality" score for a single block of text."""
    length = len(text)
    if length < config["MIN_LENGTH_THRESHOLD"]:
        return 0.0
    action_word_count = sum(1 for word in config["CHINESE_ACTION_WORDS"] if word in text)
    length_score = min(length / 50.0, 5.0)
    layer_penalty = (layer_num - 1) * config["LAYER_PENALTY_FACTOR"]
    score = ((action_word_count * config["COLOR_DIVERSITY_WEIGHT"]) +
             (length_score * config["LENGTH_WEIGHT"]) -
             layer_penalty)
    return max(0, score)

def determine_minimal_layer(soup, config):
    """
    Analyzes the document to find the most appropriate segmentation layer.
    This is now primarily for metadata purposes.
    """
    all_l_tags = soup.find_all(re.compile(r'^L(\d+)$', re.I))
    if not all_l_tags:
        return config["DEFAULT_LAYER"]

    max_layer = max(int(re.search(r'(\d+)', tag.name).group(1)) for tag in all_l_tags)

    for level in range(max_layer, 0, -1): # Changed range to include L1
        layer_tags = soup.find_all(f'L{level}', recursive=True)
        if not layer_tags:
            continue
        scores = [calculate_block_score(tag.get_text(strip=True), level, config) for tag in layer_tags]
        avg_score = sum(scores) / len(scores) if scores else 0
        if avg_score >= config["MINIMAL_LAYER_SCORE_THRESHOLD"]:
            return level

    return config["DEFAULT_LAYER"]

def insert_segment_breaks(soup):
    """
    Inserts segment break markers based on hierarchical and punctuation rules.
    """
    SEGMENT_BREAK_MARKER = "_||_SEGMENT_BREAK_||_"
    
    structural_tags = soup.find_all(
        lambda tag: hasattr(tag, 'name') and \
                    (tag.name.upper().startswith('L') or tag.name.upper() == 'NEUTRAL_CONTENT')
    )

    if len(structural_tags) < 2:
        return None

    for i in range(1, len(structural_tags)):
        current_tag = structural_tags[i]
        previous_tag = structural_tags[i-1]
        
        should_insert_break = False

        current_name = current_tag.name.upper()
        previous_name = previous_tag.name.upper()

        if current_name == 'NEUTRAL_CONTENT':
            if previous_name != 'NEUTRAL_CONTENT':
                should_insert_break = True
        
        else: # Current tag is an L-tag
            if previous_name == 'NEUTRAL_CONTENT':
                should_insert_break = True
            else: # Both are L-tags, compare them
                current_layer_match = re.match(r'^L(\d+)$', current_name, re.I)
                previous_layer_match = re.match(r'^L(\d+)$', previous_name, re.I)
                
                if current_layer_match and previous_layer_match:
                    current_layer = int(current_layer_match.group(1))
                    previous_layer = int(previous_layer_match.group(1))
                    
                    previous_text = previous_tag.get_text(strip=True)
                    
                    if previous_text.endswith(('，', '；', ',', ';')) and current_layer == previous_layer:
                        should_insert_break = False
                    elif current_layer <= previous_layer:
                        should_insert_break = True
        
        if should_insert_break:
            current_tag.insert_before(SEGMENT_BREAK_MARKER)
            
    return SEGMENT_BREAK_MARKER


def process_file(filepath, output_dir, config):
    """Main processing function for a single file."""
    basename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(basename)[0]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        content_to_parse = f"<root>{original_content}</root>" if not original_content.strip().startswith('<root>') else original_content
        soup = BeautifulSoup(content_to_parse, 'lxml-xml')

        minimal_layer = determine_minimal_layer(soup, config)
        header_line = f"Minimal Segmentation Layer Determined: L{minimal_layer}\n\n"
        #if to save the original for comparison
        '''
        original_output_path = os.path.join(output_dir, f"{filename_no_ext}_original.txt")
        with open(original_output_path, 'w', encoding='utf-8') as f:
            f.write(header_line)
            f.write(original_content)
        '''

        break_marker = insert_segment_breaks(soup)
        
        if break_marker:
            full_text = soup.get_text()
            raw_segments = [s.strip() for s in full_text.split(break_marker) if s.strip()]
            
            cleaned_segments = []
            for segment in raw_segments:
                cleaned_segment = re.sub(r'(\s*\n\s*){2,}', '\n', segment)
                cleaned_segments.append(cleaned_segment)
                
            final_output_string = "\n\n".join(cleaned_segments)
        else:
            full_text = soup.get_text(strip=True)
            final_output_string = re.sub(r'(\s*\n\s*){2,}', '\n', full_text)

        segmented_output_path = os.path.join(output_dir, f"{filename_no_ext}_segmented.txt")
        with open(segmented_output_path, 'w', encoding='utf-8') as f:
            f.write(header_line) 
            if not final_output_string:
                pass
            else:
                f.write(final_output_string)

    except Exception as e:
        # Errors are suppressed to keep the output clean
        pass


if __name__ == "__main__":
    input_directory = "sample_data/txt_files_tagged"
    output_directory = "sample_data/segmentation_results"

    os.makedirs(output_directory, exist_ok=True)

    if not os.path.isdir(input_directory):
        # Error messages are suppressed
        pass
    else:
        files_to_process = glob.glob(os.path.join(input_directory, '*.txt'))
        if not files_to_process:
            # Warning messages are suppressed
            pass
        else:
            # Wrap the file loop with tqdm to show a progress bar
            for file in tqdm(files_to_process, desc="Processing files", unit="file"):
                # --- NEW LOGIC TO SKIP PROCESSED FILES ---
                # Construct the expected output filenames for the current input file
                basename = os.path.basename(file)
                filename_no_ext = os.path.splitext(basename)[0]
                original_output_path = os.path.join(output_directory, f"{filename_no_ext}_original.txt")
                segmented_output_path = os.path.join(output_directory, f"{filename_no_ext}_segmented.txt")

                # Check if both output files already exist in the output directory
                if os.path.exists(original_output_path) and os.path.exists(segmented_output_path):
                    continue  # Skip to the next file in the loop

                # If either file is missing, proceed with processing
                process_file(file, output_directory, CONFIG)
