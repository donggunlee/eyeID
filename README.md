# eyeID

Pytorch version

[Quick run]

1. git clone https://github.com/donggunlee/eyeID.git
2. adjust path in data/train_shuffle.csv, data/valid_shuffle.csv, data/test_shuffle.csv
3. python train.py

[Data]

Train : W1S1, W1S2, W2S1, W2S2
Valid : W3S1
Test  : W3S2

[Model]

{BiLSTM-RNN}*2- {FC}*2

**In accordance with the policy protecting subjects’ personal data, the data have not been made available in public, but the data may be available to academic researchers on request to the corresponding author. (leedg0934@kaist.ac.kr)
