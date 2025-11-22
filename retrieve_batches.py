import os
import json
import time
from openai import OpenAI

# --- SCRIPT CONFIGURATION ---
# It's recommended to use environment variables for API keys in production.
# For example: api_key=os.environ.get("OPENAI_API_KEY")
API_KEY = "" #put your openai api key here

# Directory to store logs, state, and the final merged output
LOG_DIR = "sample_data/itemize_logs"

# The state file from the submission script, which contains the batch IDs
SUBMISSION_STATE_FILE = "sample_data/itemize_state.json"

# The final merged output file
MERGED_OUTPUT_FILE = os.path.join(LOG_DIR, "merged_segmentation_results.json")

# State file to keep track of batches that have already been processed
PROCESSED_BATCHES_STATE_FILE = os.path.join(LOG_DIR, "processed_batches.json")


class BatchResultManager:
    """
    Manages downloading, processing, and merging results from completed OpenAI batch jobs.
    """

    def __init__(self):
        """Initializes the manager, OpenAI client, and loads the state."""
        self.client = OpenAI(api_key=API_KEY)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.processed_batch_ids = self._load_state()
        self._log("--- Initialized Batch Result Manager ---")

    def _log(self, message):
        """Prints a log message with a timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _load_state(self):
        """Loads the set of processed batch IDs from the state file."""
        if os.path.exists(PROCESSED_BATCHES_STATE_FILE):
            try:
                with open(PROCESSED_BATCHES_STATE_FILE, 'r', encoding='utf-8') as f:
                    self._log(f"Loading state from '{PROCESSED_BATCHES_STATE_FILE}'.")
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                self._log(f"Warning: Could not load state file. Starting fresh. Error: {e}")
        return set()

    def _save_state(self):
        """Saves the current set of processed batch IDs to the state file."""
        with open(PROCESSED_BATCHES_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed_batch_ids), f, indent=4)
        self._log(f"State saved. Processed {len(self.processed_batch_ids)} batches so far.")

    def get_new_completed_batches(self):
        """
        Reads the submission state file to find relevant batch IDs, retrieves their status,
        and returns the ones that are 'completed' and have not been processed yet.
        """
        self._log(f"Reading submission state from '{SUBMISSION_STATE_FILE}' to find relevant batches...")

        if not os.path.exists(SUBMISSION_STATE_FILE):
            self._log(f"ERROR: Submission state file not found at '{SUBMISSION_STATE_FILE}'. Cannot determine which batches to retrieve.")
            return []

        try:
            with open(SUBMISSION_STATE_FILE, 'r', encoding='utf-8') as f:
                submission_state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log(f"ERROR: Could not read or parse submission state file '{SUBMISSION_STATE_FILE}'. Error: {e}")
            return []

        # Get batch IDs from the state file. The submission script stores them as keys in 'active_batches'.
        batch_ids_to_check = list(submission_state.get("active_batches", {}).keys())

        if not batch_ids_to_check:
            self._log("No active batches found in the submission state file.")
            self._log("Note: If you expect to see completed batches, they may have been cleared from the state file by the submission script upon its completion.")
            return []

        self._log(f"Found {len(batch_ids_to_check)} batch ID(s) in state file. Checking their status...")

        completed_batches = []
        for batch_id in batch_ids_to_check:
            if batch_id in self.processed_batch_ids:
                continue

            try:
                self._log(f"  -> Retrieving status for batch {batch_id}...")
                batch = self.client.batches.retrieve(batch_id)
                if batch.status == 'completed':
                    self._log(f"    -> Status is 'completed'. Adding to processing queue.")
                    completed_batches.append(batch)
                else:
                    self._log(f"    -> Status is '{batch.status}'. Will check again later.")

            except Exception as e:
                self._log(f"  -> ERROR: Could not retrieve batch {batch_id}. Details: {e}")
        
        self._log(f"Found {len(completed_batches)} new completed batches to process.")
        return completed_batches

    def process_batch_result(self, batch):
        """
        Downloads and parses the result file for a single completed batch.
        """
        results = {}
        if not batch.output_file_id:
            self._log(f"Warning: Batch {batch.id} is completed but has no output file. Skipping.")
            return results

        self._log(f"Downloading results for batch {batch.id} (File ID: {batch.output_file_id})...")
        try:
            # The API returns a response object; access its .text property for the decoded content
            response = self.client.files.content(batch.output_file_id)
            content_str = response.text
            
            # The result file is in JSONL format (one JSON object per line)
            result_lines = content_str.strip().split('\n')
            self._log(f"  -> Successfully downloaded and found {len(result_lines)} results in the batch.")

            for line in result_lines:
                try:
                    data = json.loads(line)
                    custom_id = data.get("custom_id")
                    response_body = data.get("response", {}).get("body", {})
                    
                    if not custom_id or not response_body:
                        self._log(f"  -> Warning: Skipping malformed result line: {line}")
                        continue
                    
                    # The actual model output is nested deep inside the response
                    content_str = response_body['choices'][0]['message']['content']
                    
                    # The content itself is a JSON string, so we parse it again
                    final_data = json.loads(content_str)
                    results[custom_id] = final_data

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    self._log(f"  -> ERROR: Failed to parse a result line. Error: {e}. Line: {line}")
            
            return results

        except Exception as e:
            self._log(f"ERROR: Failed to download or process file content for batch {batch.id}. Details: {e}")
            return {}

    def run(self):
        """
        Main execution function to find, download, and merge batch results.
        """
        new_batches = self.get_new_completed_batches()
        if not new_batches:
            self._log("No new results to download. Exiting.")
            return

        all_new_results = {}
        for batch in new_batches:
            batch_results = self.process_batch_result(batch)
            all_new_results.update(batch_results)
            # Mark this batch as processed so we don't download it again
            self.processed_batch_ids.add(batch.id)
            self._save_state()

        if not all_new_results:
            self._log("Finished processing, but no valid new data was found.")
            return
            
        # Merge with existing results if the file already exists
        final_results = {}
        if os.path.exists(MERGED_OUTPUT_FILE):
             self._log(f"Found existing merged file. Loading to update it.")
             with open(MERGED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                 final_results = json.load(f)
        
        # Update the 'results' dictionary and metadata
        if 'results' not in final_results:
            final_results['results'] = {}
            
        final_results['results'].update(all_new_results)
        final_results['last_updated_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        final_results['total_files_processed'] = len(final_results['results'])
        
        # Save the final consolidated file
        with open(MERGED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        
        self._log(f"--- Success! ---")
        self._log(f"Processed {len(new_batches)} batches containing {len(all_new_results)} new file results.")
        self._log(f"All results have been saved to '{MERGED_OUTPUT_FILE}'.")


if __name__ == "__main__":
    retriever = BatchResultManager()
    retriever.run()

