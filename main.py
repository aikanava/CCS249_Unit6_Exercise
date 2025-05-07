from pos_hmm import HMM

# Training Data
tagged_data = [
    [('THE', 'DET'), ('cat', 'NOUN'), ('sleeps', 'VERB')],
    [('A', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB')],
    [('THE', 'DET'), ('dog', 'NOUN'), ('sleeps', 'VERB')],
    [('MY', 'DET'), ('dog', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV')],
    [('A', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB'), ('loudly', 'ADV')],
    [('YOUR', 'DET'), ('cat', 'NOUN'), ('runs', 'VERB')],
    [('THE', 'DET'), ('bird', 'NOUN'), ('sings', 'VERB'), ('sweetly', 'ADV')],
    [('A', 'DET'), ('bird', 'NOUN'), ('chirps', 'VERB')]
]

model = HMM()
model.train(tagged_data)

# Prediction
sentence = ['The', 'cat', 'meows']
tags = model.viterbi(sentence)
print("Sentence:", sentence)
print("Predicted Tags:", tags)

# Prediction
sentence = ['My', 'dog', 'barks', 'loudly']
tags = model.viterbi(sentence)
print("Sentence:", sentence)
print("Predicted Tags:", tags)