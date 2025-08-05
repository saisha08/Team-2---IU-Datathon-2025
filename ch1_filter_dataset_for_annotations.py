# Load Libraries
import pandas as pd
import json

# Load your Bright Data output file
df = pd.read_csv("/content/sample_data/brightdata.csv")

# Define your search terms (case-insensitive)
keywords = ["jew", "netanyahu", "#gaza", "Zionist", "Israel", "settlers"]  # 👈 Edit this list as needed

# Storage for filtered results
parsed_rows = []
text_id = 1

# Iterate through each row
for _, row in df.iterrows():
    try:
        username = row.get("id", "")
        posts_raw = row.get("posts", "")

        if not posts_raw or pd.isna(posts_raw):
            continue

        # Safely parse JSON from the 'posts' column
        try:
            posts = json.loads(posts_raw)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content preview: {posts_raw[:300]}\n")
            continue

        for post in posts:
            description = post.get("description", "")
            tweet_id = post.get("post_id", "")
            date_posted = post.get("date_posted", "")

            if description and tweet_id:
                if any(kw.lower() in description.lower() for kw in keywords):
                    parsed_rows.append({
                        "text_id": text_id,
                        "Text": description,
                        "tweet_id": tweet_id,
                        "Username": username,
                        "date_posted": date_posted
                    })
                    text_id += 1

    except Exception as e:
        print(f"Unexpected error: {e}")

# Save the filtered output
df_out = pd.DataFrame(parsed_rows)
df_out.to_csv("/content/sample_data/parsed_tweets_filtered.csv", index=False, encoding="utf-8")

# Shows you how much Tweets your collection already contains
print(f"Parsing complete. {len(df_out)} relevant tweets saved.")