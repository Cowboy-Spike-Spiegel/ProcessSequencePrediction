'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite
computationally intensive.

Author: Niek Tax
'''

from __future__ import print_function, division
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Input
#from tensorflow.keras.utils.data_utils import get_file
#from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv
import time
from datetime import datetime
from math import log


########################################################################################

# global variables
EventLog = "helpdesk.csv"
AsciiOffset = 161
DaySeconds = 86400

########################################################################################
#
# this part of the code opens the file, reads it into three following variables
#
lines = []  # these are all the activity seq
timeSeqs = []   # time sequences (differences between two events)
timeSeqs2 = []  # time sequences (differences between the current and first)

# read EventLog------------------------
csvfile = open('../data/%s' % EventLog, 'r')
spamReader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamReader, None)  # skip the headers

# helper variables---------------------
# window of case time
caseStartTime = None
lastEventTime = None
# information of case
line = ''
times = []
times2 = []
# csv variables
numLines = 0
firstLine = True
lastCase = ''

# the rows are "CaseID, ActivityID, CompleteTimestamp"
for row in spamReader:
    # creates a datetime object from row[2]
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")

    # update event variables
    if row[0] != lastCase:  # 'lastCase' is to save the last executed case for the loop
        # different from the front case, update time for this case
        lastCase = row[0]
        caseStartTime = t
        lastEventTime = t
        # not first line, append them to target
        if firstLine:
            firstLine = False
        else:
            lines.append(line)
            timeSeqs.append(times)
            timeSeqs2.append(times2)
        numLines += 1
        # reset line, times, times2
        line = ''
        times = []
        times2 = []

    # append line
    line += chr(int(row[1]) + AsciiOffset)
    # append times, times2
    timeSinceLastEvent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lastEventTime))
    timediff = DaySeconds * timeSinceLastEvent.days + timeSinceLastEvent.seconds
    times.append(timediff)
    # append times, times2
    timeSinceCaseStart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(caseStartTime))
    timediff2 = DaySeconds * timeSinceCaseStart.days + timeSinceCaseStart.seconds
    times2.append(timediff2)

    # update last event
    lastEventTime = t

# add last case
lines.append(line)
timeSeqs.append(times)
timeSeqs2.append(times2)
numLines += 1

########################################################################################

# average time between events
divisor = np.mean([item for sublist in timeSeqs for item in sublist])
print('divisor: {}'.format(divisor))

# average time between current and first events
divisor2 = np.mean([item for sublist in timeSeqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

########################################################################################

# separate training data into 3 parts
elems_per_fold = int(round(numLines/3))

# fold1--------------------------------
fold1 = lines[:elems_per_fold]
fold1_t = timeSeqs[:elems_per_fold]
fold1_t2 = timeSeqs2[:elems_per_fold]

# fold2--------------------------------
fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeSeqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeSeqs2[elems_per_fold:2*elems_per_fold]

# fold3--------------------------------
fold3 = lines[2*elems_per_fold:]
fold3_t = timeSeqs[2*elems_per_fold:]
fold3_t2 = timeSeqs2[2*elems_per_fold:]

# leave away fold3 for now
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

#########################################################################################################

# append ! to each str of lines
lines_with_exclamation = []
for line in lines:
    line_with_exclamation = line + '!'
    lines_with_exclamation.append(line_with_exclamation)
lines = lines_with_exclamation
# find maximum line size
maxLen = max(map(lambda x: len(x), lines))

# next lines here to get all possible characters for events and annotate them with numbers
chars = map(lambda x: set(x), lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars) # target_chars: all characters in lines
chars.remove('!')   # chars: all characters in lines except '!'
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
print("chars:", chars)
print("target_chars:", target_chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
#print(target_char_indices, target_indices_char)










#########################################################################################################
#
# this part of the code opens the file, reads it into three following variables
#
lines = []
timeSeqs = []
timeSeqs2 = []
timeSeqs3 = []
timeSeqs4 = []

# read EventLog------------------------
csvfile = open('../data/%s' % EventLog, 'r')
spamReader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamReader, None)  # skip the headers

# helper variables---------------------
# window of case time
caseStartTime = None
lastEventTime = None
# information of case
line = ''
times = []
times2 = []
times3 = []
times4 = []
# csv variables
numLines = 0
firstLine = True
lastCase = ''

# the rows are "CaseID, ActivityID, CompleteTimestamp"
for row in spamReader:
    # creates a datetime object from row[2]
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")

    # update event variables
    if row[0] != lastCase:  # 'lastCase' is to save the last executed case for the loop
        caseStartTime = t
        lastEventTime = t
        lastCase = row[0]
        # not first line, append them to target
        if firstLine:
            firstLine = False
        else:
            lines.append(line)
            timeSeqs.append(times)
            timeSeqs2.append(times2)
            timeSeqs3.append(times3)
            timeSeqs4.append(times4)
        numLines += 1
        # reset line, times, times2, time3, time4
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []
    lastEventTime = t

    # append line
    line += chr(int(row[1]) + AsciiOffset)
    # append times
    timeSinceLastEvent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lastEventTime))
    timediff = DaySeconds * timeSinceLastEvent.days + timeSinceLastEvent.seconds
    times.append(timediff)
    # append times2
    timeSinceCaseStart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(caseStartTime))
    timediff2 = DaySeconds * timeSinceCaseStart.days + timeSinceCaseStart.seconds
    times2.append(timediff2)
    # append times3
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timeSinceMidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
    timediff3 = timeSinceMidnight.seconds   # this leaves only time even occur after midnight
    times3.append(timediff3)
    # append times4
    timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()    # day of the week
    times4.append(timediff4)

# add last case
lines.append(line)
timeSeqs.append(times)
timeSeqs2.append(times2)
timeSeqs3.append(times3)
timeSeqs4.append(times4)
numLines += 1

########################################################################################

# separate training data into 3 parts
elems_per_fold = int(round(numLines/3))

# fold1--------------------------------
fold1 = lines[:elems_per_fold]
fold1_t = timeSeqs[:elems_per_fold]
fold1_t2 = timeSeqs2[:elems_per_fold]
fold1_t3 = timeSeqs3[:elems_per_fold]
fold1_t4 = timeSeqs4[:elems_per_fold]
#print(fold1, fold1_t, fold1_t2, fold1_t3, fold1_t4)
with open('output_files/folds/fold1.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeSeq in zip(fold1, fold1_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])

# fold2--------------------------------
fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeSeqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeSeqs2[elems_per_fold:2*elems_per_fold]
fold2_t3 = timeSeqs3[elems_per_fold:2*elems_per_fold]
fold2_t4 = timeSeqs4[elems_per_fold:2*elems_per_fold]
with open('output_files/folds/fold2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold2, fold2_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])

# fold3--------------------------------
fold3 = lines[2*elems_per_fold:]
fold3_t = timeSeqs[2*elems_per_fold:]
fold3_t2 = timeSeqs2[2*elems_per_fold:]
fold3_t3 = timeSeqs3[2*elems_per_fold:]
fold3_t4 = timeSeqs4[2*elems_per_fold:]
with open('output_files/folds/fold3.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold3, fold3_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])

# leave away fold3 for now
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4









#########################################################################################################

next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []


sentences = []
softness = 0
next_chars = []
lines = map(lambda x: x+'!',lines)

sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []

step = 1
for line, line_t, line_t2, line_t3, line_t4 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4):
    for i in range(1, len(line), step):
        # we add iteratively, first symbol of the line, then two first, three...
        sentences.append(line[0: i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        next_chars.append(line[i])
        # append next
        if i == len(line)-1:
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])
print('nb sequences:', len(sentences), sentences)

print('Vectorization...')
num_features = len(chars)+5
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxLen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    # np.set_printoptions(threshold=np.inf)
    leftpad = maxLen-len(sentence)
    next_t = next_chars_t[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char: #this will encode present events to the right places
                # init by 1
                X[i, t+leftpad, char_indices[c]] = 1
        X[i, t+leftpad, len(chars)] = t+1
        X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
        X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
        X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/DaySeconds
        X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
    for c in target_chars:
        if c==next_chars[i]:
            y_a[i, target_char_indices[c]] = 1-softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    y_t[i] = next_t/divisor


#########################################################################################################

# build the model:
print('Build model...')
main_input = Input(shape=(maxLen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

# opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
opt = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)

model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model.fit(X, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxLen, epochs=500)