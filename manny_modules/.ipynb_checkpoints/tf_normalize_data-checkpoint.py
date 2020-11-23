import string
import tensorflow as tf
import regex as re

@tf.keras.utils.register_keras_serializable(package='Custom', name='normlize_data')
def normlize_data(text):
    """
    This is the pre-processing callback function used by TextVectorization.
    
    @param: text (str) [this is the text to normalize prior to input to the model]
    @return: result (str) [ this the is the text returned after being normalized]
    """
    
    # define regex to use for replacements
    html_regex = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});' # remove all html tags
    url_regex = 'https?:\/\/\S+\b|www\.(\w+\.)+\S*'
    twitter_username_regex = "@\w+" # twitter username
    html_amp_start_regex = "&\w+"  # remove html entities starting with a &
    smiley_regex = "[8:=;]['`\-]?[)dD]+|[)dD]+['`\-]?[8:=;]"
    lol_face_regex = "[8:=;]['`\-]?p+"
    sad_face_regex = "[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]"
    neutral_face_regex = "[8:=;]['`\-]?[\/|l*]"
    forward_slash_regex = "/" # add space around forward slash
    heart_regex = "<3" # heart emoji
    numbers_regex = "[-+]?[.\d]*[\d]+[:,.\d]*" # remove numbers with signs e.g. -2, +3 etc..
    hash_tag_regex = "#\w+"
    repeated_punctuation_regex = "([!?.]){2,}" # e.g. replace ?????? to ?
    remove_numbers_regex = '[0-9]+' # remove all numbers
 
    # remove punctuation
    punctuation = string.punctuation
    punctuation = punctuation.translate({ord(i):None for i in "'"}) # keep the apostrophe, but remove all other punctuation
    remove_punctuation = f'[{re.escape(punctuation)}]'
    
    result = tf.strings.lower(text)
    result = tf.strings.strip(result) # remove leading and trailing spaces
     
    # make sure to remove html tags second
    result = tf.strings.regex_replace(result, html_regex, '')
     
    # remove any URLs
    result = tf.strings.regex_replace(result, url_regex, ' ') # url 
    result = tf.strings.regex_replace(result, twitter_username_regex, ' ') # twitetr user names 
    result = tf.strings.regex_replace(result, html_amp_start_regex, ' ') # any html entity that starts with an &
    result = tf.strings.regex_replace(result, smiley_regex, ' ') # remove any smilies/emojis in the text
    result = tf.strings.regex_replace(result, lol_face_regex, ' ') # lolface emoji
    result = tf.strings.regex_replace(result, sad_face_regex, ' ') # sad face emoji
    result = tf.strings.regex_replace(result, neutral_face_regex, ' ') # face emoji
    result = tf.strings.regex_replace(result, forward_slash_regex, r' / ') # add space around forward slash
    result = tf.strings.regex_replace(result, heart_regex, ' ') # remove heart emoji
    result = tf.strings.regex_replace(result, numbers_regex, ' ') # remove numbers e.g. -3, 2, 8, +8 etc..
    result = tf.strings.regex_replace(result, hash_tag_regex, ' ') # remove hashtags
    result = tf.strings.regex_replace(result, repeated_punctuation_regex, r'\1 ') # replace any repeated puctuation with single occurance
    result = tf.strings.regex_replace(result, remove_numbers_regex, ' ') # finally remove all numbers
    
    # then remove punctuation at the very end
    result = tf.strings.regex_replace(result, remove_punctuation, '')

    return result
