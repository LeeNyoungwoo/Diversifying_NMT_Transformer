import random
from normalizer import Normalizer
import tokenizer
'''
Align TED English and Korean xml corpuses:
https://wit3.fbk.eu/mono.php?release=XML_releases&tinfo=cleanedhtml_ted
'''
# character threshold
# if number of characters is less than this, don't output it
ko_len_threshold = 10
en_len_threshold = 20
class TEDVideo(object):
    def __init__(self):
        self.url = None
        self.transcriptions = []
'''
Parse an array of TEDVideo objects from an xml file
'''
def parse_ted_file(fn):
    all_videos = []
    contents = None
    with open(fn, 'r', encoding='utf-8') as fd:
        contents = fd.read()
    for tmp in contents.split('<file')[1:]:
        ted_video = TEDVideo()
        ted_video.url = tmp.split('<url>')[1].split('</url>')[0].lower().strip()
        transcription_part = tmp.split('<transcription>')[1].split('</transcription')[0]
        invalid_file = False
        for ln in transcription_part.split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            #print(ln)
            if '" />' in ln:
                invalid_file = True
                continue # erroneous line <seekvideo id="551600" />
            seekvideo_time = int(ln.split(' id="')[1].split('"')[0])
            seekvideo_text = ln.split('">')[1].split('</seekvideo>')[0].strip()
            # seekvideo_text_normalized = Normalizer.normalize(seekvideo_text, NFC)
            # tk = tokenizer.Tokenizer.tokenize(seekvideo_text, keep_punctuation=True, keep_symbols=True)
            #print((seekvideo_time, seekvideo_text))
            ted_video.transcriptions.append((seekvideo_time, seekvideo_text))
        if not invalid_file:
            all_videos.append(ted_video)
    return all_videos
parsed_en_videos = parse_ted_file('ted_en-20160408.xml')
parsed_ko_videos = parse_ted_file('ted_ko-20160408.xml')
url_set_en = set([v.url for v in parsed_en_videos])
url_set_ko = set([v.url for v in parsed_ko_videos])
monoling_set = url_set_en | url_set_ko
biling_set = url_set_en & url_set_ko
url_to_tedvideo_en = {}
for v in parsed_en_videos:
    url_to_tedvideo_en[v.url] = v
url_to_tedvideo_ko = {}
for v in parsed_ko_videos:
    url_to_tedvideo_ko[v.url] = v
print('Detected %d/%d bilingual videos total' % (len(biling_set), len(monoling_set)))
#with open('ted_ko-20160408.xml', 'r', encoding='utf-8') as fd:
#    contents = fd.read()
sent_ko = []
sent_en = []
vocab_ko = set()
vocab_en = set()
# re-order videos in random order each time
biling_set = list(biling_set)
random.shuffle(biling_set)
for url in biling_set:
    en_video = url_to_tedvideo_en[url]
    ko_video = url_to_tedvideo_ko[url]
    transcript_en = en_video.transcriptions
    transcript_ko = ko_video.transcriptions
    biling_timestamps = set(list(zip(*transcript_en))[0]) & set(list(zip(*transcript_ko))[0])
    transcript_en_dict = dict(transcript_en)
    transcript_ko_dict = dict(transcript_ko)
    # unzip and check timestamp consistency
    #if list(zip(*transcript_en))[0] != list(zip(*transcript_ko))[0]:
        #print('en', list(zip(*transcript_en))[0])
        #print('ko', list(zip(*transcript_ko))[0])
        #print('Timestamps inconsistent for: %s. Skipping.' % url)
    #    continue
        # or should we just print only the matching timestamps?
    for ts in biling_timestamps:
        if len(transcript_ko_dict[ts]) >= ko_len_threshold and len(transcript_en_dict[ts]) >= en_len_threshold:
            sent_ko.append(transcript_ko_dict[ts])
            sent_en.append(transcript_en_dict[ts])
            for wd in transcript_ko_dict[ts].split():
                vocab_ko.add(wd)
            for wd in transcript_en_dict[ts].split():
                vocab_en.add(wd)
            #fd_ko.write(transcript_ko_dict[ts] + '\n')
            #fd_en.write(transcript_en_dict[ts] + '\n')
assert(len(sent_ko) == len(sent_en))
# train/test/dev sets
# 80/10/10
num_train = int(len(sent_ko) * 0.80)
num_test = int(len(sent_ko) * 0.10)
num_dev = len(sent_ko)-num_train-num_test
train_ko = sent_ko[:num_train]
test_ko = sent_ko[num_train:num_train+num_test]
dev_ko = sent_ko[num_train+num_test:]
train_en = sent_en[:num_train]
test_en = sent_en[num_train:num_train+num_test]
dev_en = sent_en[num_train+num_test:]
assert len(train_ko)==num_train
assert len(test_ko)==num_test
assert len(dev_ko)==num_dev
assert len(train_en)==num_train
assert len(test_en)==num_test
assert len(dev_en)==num_dev
for datasets in [('train', train_ko), ('test', test_ko), ('dev', dev_ko)]:
    with open('ted_biling_%s.ko' % datasets[0], 'w', encoding='utf-8') as fd:
        for d in datasets[1]:
            fd.write(d + '\n')
for datasets in [('train', train_en), ('test', test_en), ('dev', dev_en)]:
    with open('ted_biling_%s.en' % datasets[0], 'w', encoding='utf-8') as fd:
        for d in datasets[1]:
            fd.write(d + '\n')
# write vocab files for NMT
std_vocab = list(['<unk>', '<s>', '</s>'])
with open('ted_biling_vocab.ko', 'w', encoding='utf-8') as fd:
    for v in std_vocab + list(vocab_ko):
        fd.write('%s\n' % v)
with open('ted_biling_vocab.en', 'w', encoding='utf-8') as fd:
    for v in std_vocab + list(vocab_en):
        fd.write('%s\n' % v)