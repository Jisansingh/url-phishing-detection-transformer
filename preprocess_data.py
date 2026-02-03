import csv
import os

INPUT_FILE = "final_urls_10k.txt"
OUTPUT_FILE = "final_urls_10k_cleaned.txt"

def preprocess_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Processing {INPUT_FILE}...")
    
    unique_lines = set()
    cleaned_rows = []
    total_lines = 0
    duplicates = 0
    invalid = 0

    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        # Check if the file has a header, though inspection showed it likely doesn't
        # simpler to just read line by line and validate structure
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            # Split by last comma to separate URL and label
            parts = line.rsplit(',', 1)
            
            if len(parts) != 2:
                # Malformed line?
                # Based on inspection, some URLs might contain commas, but the format is url,label
                # The label is likely an integer (1 or 0 usually for phishing datasets)
                # Let's be strict: Needs to end with ,<something>
                print(f"Skipping malformed line {total_lines}: {line[:50]}...")
                invalid += 1
                continue
            
            url, label = parts[0].strip(), parts[1].strip()
            
            # Basic validation
            if not url or not label:
                invalid += 1
                continue
                
            # Reconstruct to ensure consistent spacing/formatting
            clean_line = f"{url},{label}"
            
            if clean_line in unique_lines:
                duplicates += 1
                continue
            
            unique_lines.add(clean_line)
            cleaned_rows.append(clean_line)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for row in cleaned_rows:
            f.write(row + "\n")

    print("-" * 30)
    print(f"Original lines: {total_lines}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Invalid/Malformed removed: {invalid}")
    print(f"Final valid lines: {len(cleaned_rows)}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
