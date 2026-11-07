import csv
from pathlib import Path
from jiosaavn_client import JioSaavnClient

queries = [
    "Shiva bhajan", "Shiv aarti", "Bholenath", "Mahadev", "Shankar", "Shiv stuti",
    "Vishnu bhajan", "Narayan", "Krishna bhajan", "Radha Krishna", "Govinda", "Jagannath", "Krishna aarti",
    "Rama bhajan", "Ram bhajan", "Ramayan", "Sita Ram",
    "Hanuman chalisa", "Bajrangbali", "Anjaneya", "Hanuman bhajan",
    "Durga bhajan", "Kali bhajan", "Devi bhajan", "Mata rani", "Lakshmi aarti", "Saraswati bhajan",
    "Ganesha bhajan", "Ganpati aarti", "Ganesh atharvashirsha",
    "Sai bhajan", "Sai baba",
    "Guru Nanak", "Waheguru", "Sikh kirtan", "Gurbani",
    "Bhajan", "Aarti", "Chalisa", "Kirtan", "Mantra",
    "Om Namah Shivaya", "Jai Shri Ram", "Jai Hanuman",
    "Parvati", "Durga", "Kali", "Lakshmi", "Saraswati", "Ganga",
    "Rudra", "Mahakal", "Bhairav", "Nataraj",
    "Vishnu sahasranam", "Lalita sahasranam",
    "Bhagavad Gita", "Ramcharitmanas",
    "Tulsi", "Meera", "Kabir", "Sur", "Tulsidas",
    "Bhakti", "Devotional", "Spiritual",
]

client = JioSaavnClient()
rows = []
seen = set()
for q in queries:
    for item in client.search(q, limit=100):
        key = item.get("song_id") or item.get("normalized_title") or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(item)

out = Path("bhakti_jiosaavn.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {out}")