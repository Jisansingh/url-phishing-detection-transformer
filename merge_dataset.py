import csv
import random

PHISHING_TARGET = 5000
LEGIT_TARGET = 5000

#  LOAD PHISHING URLS 
phishing_urls = set()

for file in ["phishing-links-ACTIVE.txt", "openphish_urls.txt"]:
    try:
        with open(file, "r", errors="ignore") as f:
            for line in f:
                url = line.strip()
                if url:
                    phishing_urls.add(url)
    except FileNotFoundError:
        pass

phishing_urls = list(phishing_urls)
print("Total phishing URLs available:", len(phishing_urls))

if len(phishing_urls) < PHISHING_TARGET:
    raise ValueError("Not enough phishing URLs to sample 5000")

phishing_sample = random.sample(phishing_urls, PHISHING_TARGET)


# LOAD LEGITIMATE URLS 
legit_urls = []

with open("top-1m.csv", newline="", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    for row in reader:
        domain = row[1]
        legit_urls.append("https://www." + domain)

print("Total legitimate URLs available:", len(legit_urls))

legit_sample = random.sample(legit_urls, LEGIT_TARGET)


# ---------- WRITE FINAL DATASET ----------
with open("final_urls_10k.txt", "w", encoding="utf-8") as f:
    for url in phishing_sample:
        f.write(url + ",1\n")   # phishing label

    for url in legit_sample:
        f.write(url + ",0\n")   # legitimate label

print("✅ final_urls_10k.txt created")
print("Total samples:", PHISHING_TARGET + LEGIT_TARGET)
