count = 0

with open("phishing-links-ACTIVE.txt", "r", errors="ignore") as f:
    for line in f:
        if line.strip():
            count += 1

print("Total URLs in phishing-links-ACTIVE.txt:", count)