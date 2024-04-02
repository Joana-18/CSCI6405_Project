import os
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# File to save best hyperparams
name = '/scratch/ssd004/scratch/amorim/src/CSCI6405/best_hyperparams.csv'
header = ['model', 'params', 
          'accuracy_mean_up', 'accuracy_std_up', 
          'f1_score_false_mean_up', 'f1_score_false_std_up', 
          'recall_false_mean_up', 'recall_false_std_up', 
          'precision_false_mean_up','precision_false_std_up', 
          'f1_score_true_mean_up','f1_score_true_std_up',
          'recall_true_mean_up', 'recall_true_std_up',  
          'precision_true_mean_up', 'precision_true_std_up',

          'accuracy_mean_down', 'accuracy_std_down', 
          'f1_score_false_mean_down', 'f1_score_false_std_down', 
          'recall_false_mean_down', 'recall_false_std_down', 
          'precision_false_mean_down','precision_false_std_down', 
          'f1_score_true_mean_down','f1_score_true_std_down',
          'recall_true_mean_down', 'recall_true_std_down',  
          'precision_true_mean_down', 'precision_true_std_down',

          'accuracy_mean_com', 'accuracy_std_com', 
          'f1_score_false_mean_com', 'f1_score_false_std_com', 
          'recall_false_mean_com', 'recall_false_std_com', 
          'precision_false_mean_com','precision_false_std_com', 
          'f1_score_true_mean_com','f1_score_true_std_com',
          'recall_true_mean_com', 'recall_true_std_com',  
          'precision_true_mean_com', 'precision_true_std_com']

WRITE_HEADER = True
if os.path.isfile(os.path.join(name)):
    WRITE_HEADER = False

# Source https://gist.github.com/sebleier/554280?permalink_comment_id=3126707#gistcomment-3126707
custom_stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]


# Load sets
def data_loader():
    ## Training
    train_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/train.csv")
    train_data['title_sentiment'] = pd.Categorical(train_data.title_sentiment)
    train_data['text_sentiment'] = pd.Categorical(train_data.text_sentiment)
    train_data['subreddit'] = pd.Categorical(train_data.subreddit)
    train_data['category'] = pd.Categorical(train_data.category)
    train_data['karma_interval'] = pd.Categorical(train_data.karma_interval)
    train_data['upvote_popular'] = train_data.upvote_popular.astype(bool)
    train_data['downvote_popular'] = train_data.downvote_popular.astype(bool)
    train_data['comments_popular'] = train_data.comments_popular.astype(bool)

    ## Validation
    val_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/val.csv")
    val_data['title_sentiment'] = pd.Categorical(val_data.title_sentiment)
    val_data['text_sentiment'] = pd.Categorical(val_data.text_sentiment)
    val_data['subreddit'] = pd.Categorical(val_data.subreddit)
    val_data['category'] = pd.Categorical(val_data.category)
    val_data['karma_interval'] = pd.Categorical(val_data.karma_interval)
    val_data['upvote_popular'] = val_data.upvote_popular.astype(bool)
    val_data['downvote_popular'] = val_data.downvote_popular.astype(bool)
    val_data['comments_popular'] = val_data.comments_popular.astype(bool)

    ## Test
    test_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/test.csv")
    test_data['title_sentiment'] = pd.Categorical(test_data.title_sentiment)
    test_data['text_sentiment'] = pd.Categorical(test_data.text_sentiment)
    test_data['subreddit'] = pd.Categorical(test_data.subreddit)
    test_data['category'] = pd.Categorical(test_data.category)
    test_data['karma_interval'] = pd.Categorical(test_data.karma_interval)
    test_data['upvote_popular'] = test_data.upvote_popular.astype(bool)
    test_data['downvote_popular'] = test_data.downvote_popular.astype(bool)
    test_data['comments_popular'] = test_data.comments_popular.astype(bool)

    return train_data, val_data, test_data

# One-hot-encoding of categorical features
def one_hot_encoding(train_data, val_data, test_data):
    cols_to_encode = ['title_sentiment', 'text_sentiment', 'karma_interval']
    # Replace NaNs with mode

    print(train_data['text_sentiment'].isna().sum(), flush=True)
    print(train_data['karma_interval'].isna().sum(), flush=True)

    ENCODER = OneHotEncoder(handle_unknown='ignore')
    ENCODER.fit(train_data[cols_to_encode])

    ## Training
    hot_encoded_train = pd.DataFrame(
        ENCODER.transform(train_data[cols_to_encode]).toarray())
    train_data_he = train_data.join(hot_encoded_train)
    train_data_he = train_data_he.drop(columns = cols_to_encode)
    train_data_he.columns = train_data_he.columns.astype(str)

    ## Validation
    hot_encoded_val = pd.DataFrame(
        ENCODER.transform(val_data[cols_to_encode]).toarray())
    val_data_he = val_data.join(hot_encoded_val)
    val_data_he = val_data_he.drop(columns = cols_to_encode)
    val_data_he.columns = val_data_he.columns.astype(str)

    ## Test
    hot_encoded_test = pd.DataFrame(
        ENCODER.transform(test_data[cols_to_encode]).toarray())
    test_data_he = test_data.join(hot_encoded_test)
    test_data_he = test_data_he.drop(columns = cols_to_encode)
    test_data_he.columns = test_data_he.columns.astype(str)

    return train_data_he, val_data_he, test_data_he

# Split into features and labels
def data_split(train_data_he, val_data_he, test_data_he):
    ## Training
    train_X = train_data_he.loc[:, ~train_data_he.columns.isin(['upvote_popular', 
                                                'downvote_popular', 
                                                'comments_popular', 
                                                'upvotes', 'downvotes',
                                                'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    train_y = train_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(train_X.columns, flush=True)

    ## Validation
    val_X = val_data_he.loc[:, ~val_data_he.columns.isin(['upvote_popular', 
                                        'downvote_popular', 
                                        'comments_popular', 
                                        'upvotes', 'downvotes',
                                        'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    val_y = val_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(val_X.columns, flush=True)

    ## Test
    test_X = test_data_he.loc[:, ~test_data_he.columns.isin(['upvote_popular', 
                                            'downvote_popular', 
                                            'comments_popular',  
                                            'upvotes', 'downvotes',
                                            'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    test_y = test_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(test_X.columns, flush=True)

    return train_X, train_y, val_X, val_y, test_X, test_y

# Text simplification
def preprocess_text(text):
    # Remove emoticons, symbols, etc.
	# source https://medium.com/codex/making-wordcloud-of-tweets-using-python-ca114b7a4ef4
	regex_pattern = re.compile(pattern = "["
			u"\U0001F600-\U0001F64F"  # emoticons
			u"\U0001F300-\U0001F5FF"  # symbols & pictographs
			u"\U0001F680-\U0001F6FF"  # transport & map symbols
			u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
							"]+", flags = re.UNICODE)
	text = re.sub(regex_pattern, '', text)

	text = text.encode('ascii','ignore') # remove zero-width-space x200b
	text = text.decode('utf-8')	

	# Lowercase
	text = text.lower()
	# Remove punctuation
	text = re.sub(r'[^\w\s]', '', text)
	# Tokenize words
	tokens = nltk.word_tokenize(text)
	# Remove stopwords (default and custom)
	stopwords_set = set(stopwords.words('english'))
	stopwords_set.update(custom_stopwords)
    # Filter tokens
	new_tokens = [token for token in tokens if token not in stopwords_set]
	# Stemming
	porter_stemmer = PorterStemmer()
	tokens_after_stemming = [porter_stemmer.stem(token) for token in new_tokens]

	return ' '.join(tokens_after_stemming)

# General function to preprocess different batches of text
def preprocess_all(data):
    preproc_text = []
    for sample in data:
        preproc_text.append(preprocess_text(sample))
    return preproc_text

# TF-IDF vectorization
def tfidf_vectorization(train_X, val_X, test_X, max_features_text, 
                        max_features_title, is_test = False):
    # Vectorizing text body 
    ## NaN replacement
    train_X.loc[:, 'self_text'] = train_X.loc[:, 'self_text'].apply(
        lambda x: x if isinstance(x, str) else '')
    val_X.loc[:, 'self_text'] = val_X.loc[:, 'self_text'].apply(
        lambda x: x if isinstance(x, str) else '')

    ## Preprocessing
    train_text = preprocess_all(train_X['self_text'])
    val_text = preprocess_all(val_X['self_text'])

    ## Vectorization
    vectorizer_text = TfidfVectorizer(max_features = max_features_text)
    vectorizer_text.fit(train_text)

    ### Training
    train_text_tfidf = pd.DataFrame(
        vectorizer_text.transform(train_text).toarray())
    train_text_tfidf = train_text_tfidf.add_prefix('text_')
    train_X = pd.concat([train_X, train_text_tfidf], axis=1)
    train_X = train_X.drop(columns = ['self_text'])

    ### Validation
    val_text_tfidf = pd.DataFrame(
        vectorizer_text.transform(val_text).toarray())
    val_text_tfidf = val_text_tfidf.add_prefix('text_')
    val_X = pd.concat([val_X, val_text_tfidf], axis=1)
    val_X = val_X.drop(columns = ['self_text'])

    ### Test
    if is_test:
        test_X.loc[:, 'self_text'] = test_X.loc[:, 'self_text'].apply(
            lambda x: x if isinstance(x, str) else '')
        test_text = preprocess_all(test_X['self_text'])

        test_text_tfidf = pd.DataFrame(
            vectorizer_text.transform(test_text).toarray())
        test_text_tfidf = test_text_tfidf.add_prefix('text_')
        test_X = pd.concat([test_X, test_text_tfidf], axis=1)
        test_X = test_X.drop(columns = ['self_text'])

    # Vectorizing title
    ## NaN replacement
    train_X.loc[:, 'title'] = train_X.loc[:, 'title'].apply(
        lambda x: x if isinstance(x, str) else '')
    val_X.loc[:, 'title'] = val_X.loc[:, 'title'].apply(
        lambda x: x if isinstance(x, str) else '')

    ## Preprocessing
    train_title = preprocess_all(train_X['title'])
    val_title = preprocess_all(val_X['title'])

    ## Vectorization
    vectorizer_title = TfidfVectorizer(max_features = max_features_title)
    vectorizer_title.fit(train_title)

    ### Training
    train_title_tfidf = pd.DataFrame(
        vectorizer_title.transform(train_title).toarray())
    train_title_tfidf = train_title_tfidf.add_prefix('title_')
    train_X = pd.concat([train_X, train_title_tfidf], axis=1)
    train_X = train_X.drop(columns = ['title'])

    ### Validation
    val_title_tfidf = pd.DataFrame(
        vectorizer_title.transform(val_title).toarray())
    val_title_tfidf = val_title_tfidf.add_prefix('title_')
    val_X = pd.concat([val_X, val_title_tfidf], axis=1)
    val_X = val_X.drop(columns = ['title'])

    ### Test
    if is_test:
        test_X.loc[:, 'title'] = test_X.loc[:, 'title'].apply(
            lambda x: x if isinstance(x, str) else '')
        test_title = preprocess_all(test_X['title'])

        test_title_tfidf = pd.DataFrame(
            vectorizer_title.transform(test_title).toarray())
        test_title_tfidf = test_title_tfidf.add_prefix('title_')
        test_X = pd.concat([test_X, test_title_tfidf], axis=1)
        test_X = test_X.drop(columns = ['title'])
        test_X.columns = test_X.columns.astype(str)
    
    train_X.columns = train_X.columns.astype(str)
    val_X.columns = val_X.columns.astype(str)

    return train_X, val_X, test_X

# Function to save best results per model
def save_results(result):
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if WRITE_HEADER:
            writer.writerow(header)
        writer.writerow(result)

# Auxiliary function to keep track of metrics for all 10 runs w/ best parameters
def append_metrics(class_metrics_dict, metric, result):
    # Update if metric exists
    if metric in class_metrics_dict:
        class_metrics_dict[metric].append(result)
    # Add if doesn't exist
    else:
        class_metrics_dict[metric] = [result]
    return class_metrics_dict

# Prediction function
def predict(best_params, result, train_X, train_y, val_X, val_y,
            test_X, test_y, rf = False, mlp = False, xgboost = False):
    # Run TF-IDF w/ best number of tokens
    train_X, val_X, test_X = tfidf_vectorization(train_X, val_X, test_X, 
                                                 best_params['max_features_text'], 
                                                 best_params['max_features_title'],
                                                 is_test = True)
    # Best classifier parameters
    classifier_params = {key: best_params[key] 
                         for key in best_params 
                         if key not in {'max_features_text', 
                                        'max_features_title'}}
                                        
    metrics_val = {}
    metrics_test = {}
    # 10 runs w/ best parameters w/ different random states
    for i in range(0, 10):
        rs = random.randint(0, 100)
        if rf:
            classifier = RandomForestClassifier(random_state = rs)
        elif mlp:
            classifier = MLPClassifier(random_state = rs)
        else:
            classifier = XGBClassifier(random_state = rs)
        classifier.set_params(**classifier_params)
        classifier.fit(train_X, train_y)

        # Validation results
        y_pred_val = classifier.predict(val_X)
        for i, class_name in enumerate(['upvote_popular', 'downvote_popular', 
                                        'comments_popular']):
            # Validation report
            output_dict = classification_report(val_y[class_name], 
                                                y_pred_val[:, i], 
                                                output_dict = True)  
            # Save run metrics per class
            if class_name not in metrics_val:
                metrics_val[class_name] = {}
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'accuracy', 
                output_dict['accuracy']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'f1_score_false', 
                output_dict['False']['f1-score']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'recall_false', 
                output_dict['False']['recall']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'precision_false', 
                output_dict['False']['precision']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'f1_score_true', 
                output_dict['True']['f1-score']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'recall_true',
                output_dict['True']['recall']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'precision_true', 
                output_dict['True']['precision']
            )

        # Test results
        y_pred_test = classifier.predict(test_X)
        for i, class_name in enumerate(['upvote_popular', 'downvote_popular', 
                                        'comments_popular']):
            # Test report
            output_dict = classification_report(test_y[class_name], 
                                                y_pred_test[:, i], 
                                                output_dict = True)                                              
            # Save run metrics per class
            if class_name not in metrics_test:
                metrics_test[class_name] = {}  
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'accuracy', 
                output_dict['accuracy']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'f1_score_false', 
                output_dict['False']['f1-score']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'recall_false', 
                output_dict['False']['recall']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'precision_false', 
                output_dict['False']['precision']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'f1_score_true', 
                output_dict['True']['f1-score']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'recall_true',
                output_dict['True']['recall']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'precision_true', 
                output_dict['True']['precision']
            )

    # Average Validation results
    for class_name in ['upvote_popular', 'downvote_popular', 'comments_popular']:
        items = metrics_val[class_name].items()
        print(f"\nValidation {class_name}", flush=True)

        for key, value in items:
            mean = np.round(np.mean(metrics_val[class_name][key]), 5)
            std = np.round(np.std(metrics_val[class_name][key]), 5)
            print(f"{key}: {mean} +/- {std}", flush=True)

            
    # Average Test results
    for class_name in ['upvote_popular', 'downvote_popular', 'comments_popular']:
        items = metrics_test[class_name].items()
        print(f"\nTest {class_name}", flush=True)

        for key, value in items:
            mean = np.round(np.mean(metrics_test[class_name][key]), 5)
            std = np.round(np.std(metrics_test[class_name][key]), 5)
            print(f"{key}: {mean} +/- {std}", flush=True)

            result.append(mean)
            result.append(std)
            
    # Save test metrics to csv
    save_results(result)

# Training function
def train(classifier, train_X, train_y, val_X, val_y):
    classifier.fit(train_X, train_y)
    
    # Evaluate on validation set
    y_pred = classifier.predict(val_X)
    f1 = f1_score(val_y, y_pred, average = 'weighted')
    return f1


# Random forest's objective function
def RF_objective_function(trial):
    # Params
    max_features_text = trial.suggest_int('max_features_text', 10, 10000)
    max_features_title = trial.suggest_int('max_features_title', 10, 10000)
    params = {
        'criterion' : trial.suggest_categorical('criterion', 
                                            ['gini', 'entropy', 'log_loss']),
        'max_depth' : trial.suggest_int('max_depth', 5, 10),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)        
    }
    train_X_2, val_X_2, _ = tfidf_vectorization(train_X, val_X, test_X, 
                                                 max_features_text, 
                                                 max_features_title)
    print(params, flush=True)
    if W_MOC:
        clf = MultiOutputClassifier(
            RandomForestClassifier(**params, random_state = 42, n_jobs = -1, 
                                   class_weight='balanced'))
    else:
        clf = RandomForestClassifier(**params, random_state = 42, n_jobs = -1,
                                     class_weight='balanced')
    f1 = train(clf, train_X_2, train_y, val_X_2, val_y)
    
    return f1
    

# MLP's objective function
def MLP_objective_function(trial):
    # Params
    max_features_text = trial.suggest_int('max_features_text', 10, 50)
    max_features_title = trial.suggest_int('max_features_title', 10, 50)
    # source: https://gist.github.com/toshihikoyanase/7e75b054395ec0a6dbb144a300862f60
    layers = []
    for i in range(1, 3):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))
    params = {
        'hidden_layer_sizes' : tuple(layers),
        'activation' : trial.suggest_categorical('activation', 
                                            ['relu', 'tanh']),
        'alpha' : trial.suggest_float('alpha', 1e-3, 1e-1, log=True),
        'solver' : trial.suggest_categorical('solver', ['sgd', 'adam']),
        'learning_rate' : trial.suggest_categorical('learning_rate', 
                                                ['constant', 'invscaling', 
                                                 'adaptive']),
    }
    print(params, flush=True)
    
    train_X_2, val_X_2, _ = tfidf_vectorization(train_X, val_X, test_X, 
                                                 max_features_text, 
                                                 max_features_title)
    if W_MOC:
        clf = MultiOutputClassifier(MLPClassifier(**params, max_iter=100000, 
                                                random_state = 42))
    else:
        clf = MLPClassifier(**params, max_iter=100000, 
                                              random_state = 42)
    f1 = train(clf, train_X_2, train_y, val_X_2, val_y)
    return f1


# XGBoost's objective function
def XGBoost_objective_function(trial):
    # Params
    max_features_text = trial.suggest_int('max_features_text', 10, 50)
    max_features_title = trial.suggest_int('max_features_title', 10, 50)
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 1e-1, log=True),
    }
    
    train_X_2, val_X_2, _ = tfidf_vectorization(train_X, val_X, test_X, 
                                                 max_features_text, 
                                                 max_features_title)
    print(params, flush=True)
    if W_MOC:
        clf = MultiOutputClassifier(XGBClassifier(**params, random_state = 42))
    else:
        clf = XGBClassifier(**params, random_state = 42)
    f1 = train(clf, train_X_2, train_y, val_X_2, val_y)
    return f1

# Random forest's prediction
def RF_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    if W_MOC:
        result = ['RF - TFIDF - MOC']
        study_name = 'rf-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TFIDF_RF_MOC.db'
    else:
        result = ['RF - No TFIDF - w/o MOC']
        study_name = 'rf-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TFIDF_RF_NOMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(RF_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    result.append(best_params)

    print("============ RF VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ RF TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, rf = True)



# MLP's prediction
def MLP_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    if W_MOC:
        result = ['MLP - TFIDF - MOC']
        study_name = 'mlp-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TFIDF_MLP_MOC.db'
    else:
        result = ['MLP - No TFIDF - w/o MOC']
        study_name = 'mlp-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TFIDF_MLP_noMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(MLP_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    best_params['hidden_layer_sizes'] = tuple(
        (best_params['n_units_1'], best_params['n_units_2'])
    )
    best_params.pop('n_units_1', None)
    best_params.pop('n_units_2', None)
    result.append(best_params)

    print("============ MLP VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ MLP TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, mlp = True)


# XGBoost's prediction
def XGBoost_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    result = ['XGBoost - TFIDF - No MOC']
    if W_MOC:
        result = ['XGBoost - TFIDF - MOC']
        study_name = 'xgb-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TDIDF_XGB_MOC.db'
    else:
        result = ['XGBoost - No TFIDF - w/o MOC']
        study_name = 'xgb-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_TDIDF_XGB_NOMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(XGBoost_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    result.append(best_params)

    print("============ XGBoost VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ XGBoost TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, xgboost = True)


W_MOC = False
# W_MOC = True

train_data, val_data, test_data = data_loader()
train_data_he, val_data_he, test_data_he = one_hot_encoding(train_data, 
                                                            val_data, test_data)
train_X, train_y, val_X, val_y, test_X, test_y = data_split(train_data_he, 
                                                            val_data_he, 
                                                            test_data_he)
# RF_prediction(train_X, train_y, val_X, val_y, test_X, test_y)
# MLP_prediction(train_X, train_y, val_X, val_y, test_X, test_y)
XGBoost_prediction(train_X, train_y, val_X, val_y, test_X, test_y)