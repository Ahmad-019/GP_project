import pandas as pd
import re
import time
from googleapiclient.discovery import build

# --- CONFIGURATION AND PARAMETERS ---
# Enter your YouTube Data API v3 Key below
API_KEY = 'AIzaSyB0ygBDFFwwdnNdoyJnaBYuLrwE-dCn5R0' 

# Target Video: Joe Rogan Experience #2219 - Donald Trump
VIDEO_ID = 'hBMoPUAeLnY' 

# Extraction limits as per project requirements
TARGET_SCAN_LIMIT = 50000 

def fetch_youtube_research_data(api_key, video_id, target_limit):
    """
    Connects to YouTube API, performs paginated extraction of comments,
    extracts all embedded timestamps, and collects metadata (Likes, Replies, Author).
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    dataset = []
    next_page_token = None
    scanned_count = 0
    
    # Regular Expression for time-series extraction (HH:MM:SS or MM:SS)
    timestamp_pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b'

    print(f"SYSTEM: Commencing data extraction for Video ID: {video_id}")
    print(f"SYSTEM: Scan target set to {target_limit} records.")

    while scanned_count < target_limit:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                
                # Metadata collection
                raw_comment = snippet['textDisplay']
                author = snippet['authorDisplayName']
                likes = snippet['likeCount']
                replies = item['snippet']['totalReplyCount']
                
                # Data normalization (removing unnecessary whitespaces/newlines)
                normalized_text = " ".join(raw_comment.split())
                
                # Comprehensive timestamp extraction (capturing multiple signals per comment)
                timestamps = re.findall(timestamp_pattern, normalized_text)
                
                if timestamps:
                    for ts in timestamps:
                        dataset.append({
                            'Author': author,
                            'Timestamp': ts,
                            'Likes': likes,
                            'Total_Replies': replies,
                            'Comment_Content': normalized_text
                        })
            
            scanned_count += len(response['items'])
            print(f"STATUS: {scanned_count} comments processed. Current Data Points: {len(dataset)}")

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                print("STATUS: End of comment threads reached.")
                break
                
            # Latency to prevent API rate limiting
            time.sleep(0.05) 

        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
            break
            
    return dataset

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    try:
        # Initialize extraction process
        extracted_data = fetch_youtube_research_data(API_KEY, VIDEO_ID, TARGET_SCAN_LIMIT)
        
        # Load into Pandas DataFrame for cleaning
        df = pd.DataFrame(extracted_data)
        
        if not df.empty:
            # Removing redundant records where Timestamp and Content are identical
            df = df.drop_duplicates()
            
            # Export to CSV for Business Intelligence analysis
            filename = 'GP_Final_Research_Data.csv'
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print("\n" + "="*30)
            print("EXTRACTION SUMMARY")
            print("="*30)
            print(f"Total Unique Signals: {len(df)}")
            print(f"Export Status: Successful")
            print(f"File Path: {filename}")
            print("="*30)
        else:
            print("ALERT: No relevant data points identified.")

    except Exception as main_error:
        print(f"PROCESS FAILED: {main_error}")