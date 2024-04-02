import praw
import csv
import os

LIMIT = 250
# Init instance
reddit = praw.Reddit(
        client_id="",
        client_secret="",
        password="",
        user_agent="",
        username="",
    )

#### Subreddits of interest ####
subreddits = ['Anxiety', 'AnxietyDepression', 'depression', 'depression_help',
                'mentalhealth', 'MentalHealthSupport', 'mentalillness', 'ptsd', 
                'selfhelp', 'selfimprovement', 'socialanxiety']



header = ['id', 'title', 'author', 'author_id', 'author_karma', 'author_flair',
          'upvotes', 'score', 'upvote_ratio', 'post_ts', 'distinguished',
          'edited', 'original', 'self', 'locked', 'name', 'comments', 'nsfw',
          'self_text', 'spoiler', 'stickied', 'subreddit', 'subreddit_id',
          'subreddit_subs', 'url', 'category']

name = '/scratch/ssd004/scratch/amorim/src/CSCI6405/Reddit_MentalHealth_250.csv'

WRITE_HEADER = True
if os.path.isfile(os.path.join(name)):
    WRITE_HEADER = False

def get_post_properties(submission):
    id = submission.id
    title = submission.title
    try:
        author_name = submission.author.name
    except:
        author_name = ''
    try:
        author_id = submission.author.id
    except:
        author_id = ''
    try:
        author_karma = submission.author.comment_karma
    except:
        author_karma = float('-inf')
    upvotes = submission.ups
    score = submission.score
    upvote_ratio = submission.upvote_ratio
    author_flair = submission.author_flair_text
    time_post = submission.created_utc
    distinguished = submission.distinguished
    edited = submission.edited
    OG_content = submission.is_original_content
    is_self = submission.is_self
    locked = submission.locked
    name = submission.name
    num_comments = submission.num_comments
    nsfw = submission.over_18
    self_text = submission.selftext
    spoiler = submission.spoiler
    stickied = submission.stickied
    subreddit = submission.subreddit
    subreddit_id = submission.subreddit.id
    subreddit_subs = submission.subreddit.subscribers
    url = submission.url

    row = [id, title, author_name, author_id, author_karma, author_flair,
            upvotes, score, upvote_ratio, time_post, distinguished,
            edited, OG_content, is_self, locked, name, num_comments,
            nsfw, self_text, spoiler, stickied, subreddit, subreddit_id,
            subreddit_subs, url]
    return row

def save_rows(rows):
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if WRITE_HEADER:
            writer.writerow(header)
        writer.writerows(rows)

def get_info(posts, current_num_rows, category):
    rows = []
    
    for post in posts:
        ok = False
        while not ok: 
            try:   
                row = get_post_properties(post)
                row.append(category)
                rows.append(row)
                current_num_rows += 1
                ok = True
            except:
                print("------> EXCEPTION in get_info!! ", flush=True)
    save_rows(rows)

    return current_num_rows


for subreddit in subreddits:
    print("==================== " + subreddit + " ====================", flush=True)
    current_num_rows = 0
    sr = reddit.subreddit(subreddit)

    print("--------------- CONTROVERSIAL ---------------", flush=True)    
    category = 'Controversial'
    ok = False
    while not ok:
        try:
            posts = sr.controversial(limit=LIMIT)
            ok = True
        except:
            print("------> EXCEPTION in post retrieval!! ", flush=True)

    current_num_rows = get_info(posts, current_num_rows, category)
    print("NUM ROWS: ", current_num_rows, flush=True)

    print("--------------- HOT ---------------", flush=True)   
    rows = []
    ok = False
    category = 'Hot'
    while not ok:
        try:
            posts = sr.hot(limit=LIMIT)
            ok = True
        except:
            print("------> EXCEPTION in post retrieval!! ", flush=True)

    current_num_rows = get_info(posts, current_num_rows, category)
    print("NUM ROWS: ", current_num_rows, flush=True)

    print("--------------- NEW ---------------", flush=True)
    rows = []
    ok = False
    category = 'New'
    while not ok:
        try:
            posts = sr.new(limit=LIMIT)
            ok = True
        except:
            print("------> EXCEPTION in post retrieval!! ", flush=True)

    current_num_rows = get_info(posts, current_num_rows, category)
    print("NUM ROWS: ", current_num_rows, flush=True)

    print("--------------- TOP ---------------", flush=True)
    rows = []
    ok = False
    category = 'Top'
    while not ok:
        try:
            posts = sr.top(limit=LIMIT)
            ok = True
        except:
            print("------> EXCEPTION in post retrieval!! ", flush=True)

    current_num_rows = get_info(posts, current_num_rows, category)
    print("NUM ROWS: ", current_num_rows, flush=True)

print("DONE!!!", flush=True)

