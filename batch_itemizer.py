import os
import time
import sys
import json
import random
import tiktoken
from tqdm import tqdm
from openai import OpenAI
import glob

# --- SCRIPT CONFIGURATION ---

API_KEY = "" #put your openai api key here

INPUT_FOLDER = r'sample_data/txt_files'
SEGMENTATION_RESULTS_FOLDER = r'sample_data/itemized_result'

MODEL_NAME = "gpt-4.1-mini"
QUEUE_LIMIT_TOKENS = 900_000_000 # adjust based on your usage tier
MAX_FILES_PER_BATCH = 5000 # adjust based on your usage tier
LOOP_INTERVAL_SECONDS =5 # asjust ad need, can manuall
N_FILES_TO_PROCESS = 0  
TAKE_RANDOM_SAMPLE = False

STATE_FILE = "sample_data/itemize_state.json"
LOG_DIR = r"sample_data/itemize_logs"
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error_log.txt")

SYSTEM_PROMPT = """
您是专门用于对中文法律文件进行结构化分析的AI助理。您的任务是识别文本中的特定段落，并根据其内容和结构进行分类。
输入文本的每一行都以`<line #>`作为前缀。

核心目标：
核心目标是：隔离层级列表项 (L# 标签) 和无意义的样板文本 (NEUTRAL_CONTENT 标签)，同时保留有意义的散文段落不进行标记，以供后续分析。

输出要求：
您的输出必须是一个JSON对象。此对象包含一个名为 "segments" 的键，其值为一个数组。数组中的每个对象都应包含：
1. "label": (string) 必须是 "L1", "L2", "L3", ..., 或 "NEUTRAL_CONTENT"。
2. "line_numbers": (array of integers) 与该标签对应的原始行号列表。

标记规则：
1. 层级标签 (L#):
   - 使用 L1、L2 等标签来标记结构化的、有编号或有字母的列表项。
   - 层级顺序: 缩进更深的列表项是缩进较浅的列表项的子项，并应使用更高的层级编号（例如，L3 是 L2 的子项）。
   - 文章条款 (第#条): 以 第#条 开头的列表项（例如 第一条、第二条）是最高层级，必须标记为 L1。
   - 确保一个层级标签包括该蹭几下的所有内容，即使他们不在一行内。

2. 中性内容 (NEUTRAL_CONTENT):
   - 仅用于标记客观的、程序性的、或不具有任何分析意义或情感色彩的样板文本。
   - 包括但不限于：文件标题、文首、目录、人名、引用编号、称呼、结尾落款和日期、简单的项目列举（例如人名或职称列表）。
   - 如果连续多行被识别为中性内容，必须将它们的行号组合在一个 "line_numbers" 数组中。
   - 如果一段文本含有指令，指示，建议等有意义内容，则不该被标记为中性内容。

3. 不标记的内容：
   - 提供解释、推理、条件或背景信息且不属于列表结构的散文段落，被视为“有意义的”。这些段落必须保持不标记，以用于情感和情绪分析。

示例：
输入文本:
<line 1> 　　以下名单为签署者：北京市（70人） 卢军民   李玉梅 王  波   王春来 朱海群 左新文 郝国柱 刘晓谕 刘海泽 周全涛 张瑞彬
<line 2>
<line 3> 　第一章　总则
<line 4> 　第一条　为了实施《中华人民共和国海关稽查条例》（以下简称《稽查条例》），制定本办法。
<line 5> 　第二条　本办法中所称“被稽查人”是指《稽查条例》第三条所列企业、单位。其进出口活动包括：
<line 6> 　（一）进出口申报；
<line 7>   包括跟及相关条例申报所有货物
<line 8> 　（二）进出口关税和其他税、费的缴纳；
<line 9> 　（三）进出口许可证、件的交验；
<line 10> 　第三条　海关在下列期限内对被稽查人的会计帐簿、会计凭证、报关单证以及其他有关资料和有关进出口货物进行稽查          
<line 11>  
<line 12> 　　中国海关管理委员会
<line 13> 　二○一○年十一月二十二日

预期输出 (必须是可解析的JSON，不要包含换行号):
{"segments":[{"label":"NEUTRAL_CONTENT","line_numbers":[1]},{"label":"L1","line_numbers":[3]},{"label":"L2","line_numbers":[4]},{"label":"L2","line_numbers":[5]},{"label":"L3","line_numbers":[6,7]},{"label":"L3","line_numbers":[8]},{"label":"L3","line_numbers":[9]},{"label":"L2","line_numbers":[10]},{"label":"NEUTRAL_CONTENT","line_numbers":[12,13]}]}
最终指令：
严格按照上述规则分析提供的文本，并仅返回一个JSON对象。不要包含任何额外的解释或评论。
"""

class SegmentationBatchManager:
    """Manages submitting and monitoring document segmentation jobs via OpenAI Batch API."""

    def __init__(self, source_directory):
        self.source_directory = source_directory
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", API_KEY))
        self.state = self._load_state()
        self._initialize_source_files()
        
        os.makedirs(LOG_DIR, exist_ok=True)
        
        try:
            self.encoder = tiktoken.encoding_for_model(MODEL_NAME)
        except Exception as e:
            self._log_error(f"Failed to initialize tiktoken encoder: {e}")
            sys.exit(1)

    def _log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _log_error(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
        self._log(f"ERROR: {message}. Check {ERROR_LOG_FILE} for details.")

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            self._log(f"Found existing state file. Resuming from '{STATE_FILE}'.")
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_file_submitted_index": -1,
            "active_batches": {},
            "source_files": [],
            "all_files_submitted": False
        }

    def _save_state(self):
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=4)
        self._log(f"State saved to '{STATE_FILE}'.")

    def _initialize_source_files(self):
        """Scans the source directory for .txt files and stores them in the state."""
        # Always refresh the source_files list to avoid mistaken zero-file state
        self._log(f"Scanning '{self.source_directory}' for .txt files...")
        
        # --- Filtering logic ---
        result_files = glob.glob(os.path.join(SEGMENTATION_RESULTS_FOLDER, "*_original.txt"))
        skip_basenames = set()
        for rf in result_files:
            base = os.path.basename(rf)
            if base.endswith("_original.txt"):
                skip_basenames.add(base[:-len("_original.txt")])

        all_files = glob.glob(os.path.join(self.source_directory, "*.txt"))

        files_to_process = []
        skipped_files = []
        for f in all_files:
            base = os.path.basename(f)
            name_no_ext = base[:-4] if base.endswith('.txt') else base
            if name_no_ext in skip_basenames:
                skipped_files.append(f)
            else:
                files_to_process.append(f)

        self._log(f"Skipped {len(skipped_files)} source files (because they match a segmentation result).")
        self._log(f"Selected {len(files_to_process)} candidate files to process.")

        # Always randomize before trimming, if enabled
        if TAKE_RANDOM_SAMPLE and len(files_to_process) > 1:
            self._log("Random sampling enabled: shuffling file list.")
            random.shuffle(files_to_process)

        # Only trim if N_FILES_TO_PROCESS > 0 (if 0, process ALL)
        if N_FILES_TO_PROCESS > 0:
            if TAKE_RANDOM_SAMPLE:
                self._log(f"Selecting random {N_FILES_TO_PROCESS} files sample from candidates.")
            else:
                self._log(f"Limiting file list to first {N_FILES_TO_PROCESS} files found per config.")
            files_to_process = files_to_process[:N_FILES_TO_PROCESS]
            self._log(f"Final list: {len(files_to_process)} files selected for batch processing.")
        else:
            self._log(f"Configured to process ALL {len(files_to_process)} files after skipping and optional randomization.")

        self.state["source_files"] = files_to_process
        self.state["last_file_submitted_index"] = -1  # Always reset index if re-initialized
        self.state["all_files_submitted"] = False     # Always allow full run after source_files refresh
        self._save_state()

    def get_current_queue_size(self):
        """Calculates the total tokens of all active batches and updates their statuses."""
        total_tokens = 0
        if not self.state["active_batches"]:
            return 0

        self._log("Checking status of active batches...")
        for batch_id in list(self.state["active_batches"].keys()):
            try:
                batch_obj = self.client.batches.retrieve(batch_id)
                status = batch_obj.status
                self.state["active_batches"][batch_id]['status'] = status

                if status in ["validating", "in_progress", "finalizing"]:
                    total_tokens += self.state["active_batches"][batch_id]["total_tokens"]
                elif status in ["completed", "failed", "cancelled", "expired"]:
                    self._log(f"Batch {batch_id} has finished with status '{status}'. It no longer counts towards queue limit.")
                    del self.state["active_batches"][batch_id]
            except Exception as e:
                self._log_error(f"Could not retrieve status for batch {batch_id}. Error: {e}")
                total_tokens += self.state["active_batches"][batch_id].get("total_tokens", 0)

        self._save_state()
        return total_tokens

    def submit_new_batch(self, available_token_capacity):
        """Prepares and submits a new batch of files if there is capacity."""
        if self.state.get("all_files_submitted"):
            return

        self._log("Preparing a new batch from unprocessed files...")
        files_for_batch = []
        batch_total_tokens = 0
        system_prompt_tokens = len(self.encoder.encode(SYSTEM_PROMPT))
        last_file_index_in_batch = self.state["last_file_submitted_index"]

        source_files = self.state.get("source_files", [])
        start_index = self.state["last_file_submitted_index"] + 1

        with tqdm(initial=start_index, total=len(source_files), desc="Scanning Files", unit="file") as pbar:
            for i in range(start_index, len(source_files)):
                filepath = source_files[i]
                filename = os.path.basename(filepath)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    prefixed_content = "".join([f"<line {idx+1}> {line.strip()}\n" for idx, line in enumerate(lines)])
                    request_tokens = system_prompt_tokens + len(self.encoder.encode(prefixed_content))

                    if batch_total_tokens + request_tokens > available_token_capacity or len(files_for_batch) >= MAX_FILES_PER_BATCH:
                        break

                    files_for_batch.append({"filename": filename, "content": prefixed_content})
                    batch_total_tokens += request_tokens
                    last_file_index_in_batch = i
                    pbar.update(1)
                except Exception as e:
                    self._log_error(f"Could not read or process file '{filepath}'. Skipping. Error: {e}")
                    last_file_index_in_batch = i
            else:
                self.state["all_files_submitted"] = True

        self.state["last_file_submitted_index"] = last_file_index_in_batch

        if not files_for_batch:
            self._log("No new files to process that fit within the available queue capacity.")
            self._save_state()
            return

        self._log(f"Prepared {len(files_for_batch)} new files for a batch. Total tokens: {batch_total_tokens:,}")

        batch_input_file = os.path.join(LOG_DIR, f"batch_input_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        with open(batch_input_file, "w", encoding="utf-8") as f:
            for file_data in files_for_batch:
                request = {
                    "custom_id": file_data["filename"],
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": file_data["content"]}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')

        try:
            self._log(f"Uploading batch file: {batch_input_file}...")
            uploaded_file = self.client.files.create(file=open(batch_input_file, "rb"), purpose="batch")

            self._log("Submitting batch job...")
            batch_obj = self.client.batches.create(
                input_file_id=uploaded_file.id, endpoint="/v1/chat/completions", completion_window="24h"
            )

            self.state["active_batches"][batch_obj.id] = {
                "status": batch_obj.status, "submitted_at": time.time(),
                "input_file_id": uploaded_file.id, "total_tokens": batch_total_tokens,
                "file_count": len(files_for_batch)
            }
            self._log(f"--- Batch Job Submitted Successfully! Batch ID: {batch_obj.id} ---")
            self._save_state()
        except Exception as e:
            self._log_error(f"An error occurred during batch submission: {e}")

    def run(self):
        """The main execution loop for submitting and monitoring batches."""
        self._log("--- Starting Document Batch Submission Manager ---")
        while True:
            current_queue_size = self.get_current_queue_size()
            self._log(f"Current active queue size: {current_queue_size:,} / {QUEUE_LIMIT_TOKENS:,} tokens.")
            available_capacity = QUEUE_LIMIT_TOKENS - current_queue_size

            if available_capacity > 0:
                self.submit_new_batch(available_capacity)
            else:
                self._log("Queue is full. Will check again later.")

            if self.state.get("all_files_submitted") and not self.state.get("active_batches"):
                self._log("--- All files have been submitted and all batches are complete. Exiting. ---")
                break

            self._log(f"Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
            time.sleep(LOOP_INTERVAL_SECONDS)

if __name__ == "__main__":
    manager = SegmentationBatchManager(source_directory=INPUT_FOLDER)
    manager.run()