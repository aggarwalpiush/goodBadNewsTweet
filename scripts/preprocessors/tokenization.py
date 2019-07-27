# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    try:
      output.append(vocab[item])
    except KeyError:
      output.append(-1)
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
    self.ark_tokenizer = Arc_preprocessor()

  def tokenize(self, text):

    # using ark tokeinization
    return self.ark_tokenizer.tokenizeRawTweetText(text)
    # using basic tokenizer original code of the file
    # split_tokens = []
    # for token in self.basic_tokenizer.tokenize(text):
    #   for sub_token in self.wordpiece_tokenizer.tokenize(token):
    #     split_tokens.append(sub_token)

    # return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


# adding ark tokenizer class 

import operator
import sys
import codecs

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

try:
    import html
except ImportError:
    pass


def regex_or(self,*items):
    return '(?:' + '|'.join(items) + ')'


Contractions = re.compile(u"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

punctChars = r"['\"“”‘’.?!…,:;]"
#punctSeq   = punctChars+"+"  #'anthem'. => ' anthem '.
punctSeq   = r"['\"“”‘’]+|[.?!,…]+|[:;]+" #'anthem'. => ' anthem ' .
entity     = r"&(?:amp|lt|gt|quot);"
#  URLs


# BTO 2012-06: everyone thinks the daringfireball regex should be better, but they're wrong.
# If you actually empirically test it the results are bad.
# Please see https://github.com/brendano/ark-tweet-nlp/pull/9

urlStart1  = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
ccTLDs   = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)" #TODO: remove obscure country domains?
urlStart2  = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
urlBody    = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd     = r"(?:\.\.+|[<>]|\s|$)"
url        = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"


# Numeric
timeLike   = r"\d+(?::\d+){1,2}"
#numNum     = r"\d+\.\d+"
numberWithCommas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"
numComb  = u"[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0-\u20b9]?\\d+(?:\\.\\d+)+%?"

# Abbreviations
boundaryNotDot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1  = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
aa2  = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)
separators  = "(?:--+|―|—|~|–|=)"
decorations = u"(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)"
thingsThatSplitWords = r"[^\s\.,?\"]"
embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"

#  Emoticons
# myleott: in Python the (?iu) flags affect the whole expression
#normalEyes = "(?iu)[:=]" # 8 and x are eyes but cause problems
normalEyes = "[:=]" # 8 and x are eyes but cause problems
wink = "[;]"
noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
happyMouths = r"[D\)\]\}]+"
sadMouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned

# mouth repetition examples:
# @aliciakeys Put it in a love song :-))
# @hellocalyclops =))=))=)) Oh well

# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
#bfLeft = u"(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
bfLeft = u"(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bfCenter = r"(?:[\.]|[_-]+)"
bfRight = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
# myleott: in Python the (?i) flag affects the whole expression
#basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5
basicface = "(?:" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5

eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
eeRight= u"[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight

oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"


emoticon = regex_or(
        # Standard version  :) :( :] :D :P
        "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea,"[Oo]") + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths+r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),

        # reversed version (: D:  use positive lookbehind to remove "(word):"
        # because eyes on the right side is more ambiguous with the standard usage of : ;
        regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths,happyMouths,otherMouths) + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",

        #inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
        eastEmote.replace("2", "1", 1), basicface,
        # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]
        # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

        # myleott: o.O and O.o are two of the biggest sources of differences
        #          between this and the Java version. One little hack won't hurt...
        oOEmote
)

Hearts = "(?:<+/?3+)+" #the other hearts are in decorations

Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", u"[\u2190-\u21ff]+")

# BTO 2011-06: restored Hashtag, AtMention protection (dropped in original scala port) because it fixes
# "hello (#hashtag)" ==> "hello (#hashtag )"  WRONG
# "hello (#hashtag)" ==> "hello ( #hashtag )"  RIGHT
# "hello (@person)" ==> "hello (@person )"  WRONG
# "hello (@person)" ==> "hello ( @person )"  RIGHT
# ... Some sort of weird interaction with edgepunct I guess, because edgepunct
# has poor content-symbol detection.

# This also gets #1 #40 which probably aren't hashtags .. but good as tokens.
# If you want good hashtag identification, use a different regex.
Hashtag = "#[a-zA-Z0-9_]+"  #optional: lookbehind for \b
#optional: lookbehind for \b, max length 15
AtMention = "[@＠][a-zA-Z0-9_]+"

# I was worried this would conflict with at-mentions
# but seems ok in sample of 5800: 7 changes all email fixes
# http://www.regular-expressions.info/email.html
Bound = r"(?:\W|^|$)"
Email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +Bound+")"

# We will be tokenizing using these regexps as delimiters
# Additionally, these things are "protected", meaning they shouldn't be further split themselves.
Protected  = re.compile(
    regex_or(
        Hearts,
        url,
        Email,
        timeLike,
        #numNum,
        numberWithCommas,
        numComb,
        emoticon,
        Arrows,
        entity,
        punctSeq,
        arbitraryAbbrev,
        separators,
        decorations,
        embeddedApostrophe,
        Hashtag,
        AtMention), re.UNICODE)

# Edge punctuation
# Want: 'foo' => ' foo '
# While also:   don't => don't
# the first is considered "edge punctuation".
# the second is word-internal punctuation -- don't want to mess with it.
# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.
# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.

# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)
#edgePunctChars    = r"'\"“”‘’«»{}\(\)\[\]\*&" #add \\p{So}? (symbols)
edgePunctChars    = u"'\"“”‘’«»{}\\(\\)\\[\\]\\*&" #add \\p{So}? (symbols)
edgePunct    = "[" + edgePunctChars + "]"
notEdgePunct = "[a-zA-Z0-9]" # content characters
offEdge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
EdgePunctLeft  = re.compile(offEdge + "("+edgePunct+"+)("+notEdgePunct+")", re.UNICODE)
EdgePunctRight = re.compile("("+notEdgePunct+")("+edgePunct+"+)" + offEdge, re.UNICODE)

class Arc_preprocessor(object):

    def splitEdgePunct(self,input):
        input = EdgePunctLeft.sub(r"\1\2 \3", input)
        input = EdgePunctRight.sub(r"\1 \2\3", input)
        return input

    # The main work of tokenizing a tweet.
    def simpleTokenize(self, text):

        # Do the no-brainers first
        splitPunctText = self.splitEdgePunct(text)

        textLength = len(splitPunctText)

        # BTO: the logic here got quite convoluted via the Scala porting detour
        # It would be good to switch back to a nice simple procedural style like in the Python version
        # ... Scala is such a pain.  Never again.

        # Find the matches for subsequences that should be protected,
        # e.g. URLs, 1.0, U.N.K.L.E., 12:53
        bads = []
        badSpans = []
        for match in Protected.finditer(splitPunctText):
            # The spans of the "bads" should not be split.
            if (match.start() != match.end()): #unnecessary?
                bads.append( [splitPunctText[match.start():match.end()]] )
                badSpans.append( (match.start(), match.end()) )

        # Create a list of indices to create the "goods", which can be
        # split. We are taking "bad" spans like
        #     List((2,5), (8,10))
        # to create
        #     List(0, 2, 5, 8, 10, 12)
        # where, e.g., "12" here would be the textLength
        # has an even length and no indices are the same
        indices = [0]
        for (first, second) in badSpans:
            indices.append(first)
            indices.append(second)
        indices.append(textLength)

        # Group the indices and map them to their respective portion of the string
        splitGoods = []
        for i in range(0, len(indices), 2):
            goodstr = splitPunctText[indices[i]:indices[i+1]]
            splitstr = goodstr.strip().split(" ")
            splitGoods.append(splitstr)

        #  Reinterpolate the 'good' and 'bad' Lists, ensuring that
        #  additonal tokens from last good item get included
        zippedStr = []
        for i in range(len(bads)):
            zippedStr = self.addAllnonempty(zippedStr, splitGoods[i])
            zippedStr = self.addAllnonempty(zippedStr, bads[i])
        zippedStr = self.addAllnonempty(zippedStr, splitGoods[len(bads)])

        # BTO: our POS tagger wants "ur" and "you're" to both be one token.
        # Uncomment to get "you 're"
        #splitStr = []
        #for tok in zippedStr:
        #    splitStr.extend(splitToken(tok))
        #zippedStr = splitStr

        return zippedStr

    def addAllnonempty(self, master, smaller):
        for s in smaller:
            strim = s.strip()
            if (len(strim) > 0):
                master.append(strim)
        return master

    # "foo   bar " => "foo bar"
    def squeezeWhitespace(self, input):
        return Whitespace.sub(" ", input).strip()

    # Final pass tokenization based on special patterns
    def splitToken(self, token):
        m = Contractions.search(token)
        if m:
            return [m.group(1), m.group(2)]
        return [token]

    # Assume 'text' has no HTML escaping.
    def tokenize(self, text):
        return self.simpleTokenize(self.squeezeWhitespace(text))


    # Twitter text comes HTML-escaped, so unescape it.
    # We also first unescape &amp;'s, in case the text has been buggily double-escaped.
    def normalizeTextForTagger(self, text):
        assert sys.version_info[0] >= 3 and sys.version_info[1] > 3, 'Python version >3.3 required'
        text = text.replace("&amp;", "&")
        text = html.unescape(text).lower()
        return text

    # This is intended for raw tweet text -- we do some HTML entity unescaping before running the tagger.
    #
    # This function normalizes the input text BEFORE calling the tokenizer.
    # So the tokens you get back may not exactly correspond to
    # substrings of the original text.
    def tokenizeRawTweetText(self, text):
        tokens = self.tokenize(self.normalizeTextForTagger(text))
        return tokens











class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False