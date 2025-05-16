1. create venv and install requirements
2. download glove to ./glove
3. train models in ./neural_netwroks with commands below:
# 1) BiLSTM (no context)
python blstm.py 128 0.5 256 50

# 2) BiLSTM-Context (direct expansion)
python blstm_context.py 128 0.5 256 50

# 3) CNN (word-level)
python cnn.py 1024 0.6 256 50

# 4) CNN-MFS (with Most-Frequent-Sense labels)
python cnn-mfs.py 1024 0.6 256 50

# 5) Char-CNN (character-level CNN)
python char_cnn.py 128 0.5 256 500

# 6) CNN-Feature (incorporate hand-crafted features)
python cnn_feature.py 1024 0.6 256 50

# 7) CNN-Context (concatenate prev∥curr∥next)
python cnn_context.py 1024 0.6 256 50

# 8) CNN-Context-Rep (separate conv nets on prev/curr/next then concat)
python cnn_context_rep.py 1024 0.6 256 50 128 128
