import os
import pandas as pd

METHOD_DIRS = {
    'res/blstm':           'BiLSTM',
    'res/cnn':             'CNN',
    'res/cnn-mfs':         'CNN-MFS',
    'res/char_cnn':        'Char-CNN',
    'res/blstm_context':   'BiLSTM-Context',
    'res/cnn_feature':     'CNN-Feature',
    'res/cnn_context':     'CNN-Context',
    'res/cnn_context_rep': 'CNN-Context-Rep',
    'res/cnn_context_rep_improved': 'CNN-Context-Rep-Improved',
}

rows = []
for d, method in METHOD_DIRS.items():
    # find the .log file
    logs = [f for f in os.listdir(d) if f.endswith('.log')]
    if not logs:
        print(f"⚠️  no .log in {d}, skipping")
        continue

    logf = os.path.join(d, logs[0])
    for line in open(logf, 'r'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = [p.strip() for p in line.split(',')]
        # we expect 13 parts:
        # [units,drop,dense,max_len,th,
        #  acc_val,p_val,r_val,f1_val,
        #  acc_test,p_test,r_test,f1_test]
        if len(parts) < 13:
            continue

        # parse threshold
        try:
            th = float(parts[4])
        except ValueError:
            continue

        if abs(th - 0.5) < 1e-6:
            # pick off the test‐set metrics
            acc_test = float(parts[9])
            p_test   = float(parts[10])
            r_test   = float(parts[11])
            f1_test  = float(parts[12])
            rows.append({
                'Method':    method,
                'Accuracy':  acc_test,
                'Precision': p_test,
                'Recall':    r_test,
                'F1':        f1_test,
            })
            break  # done with this log

df = pd.DataFrame(rows)

# enforce paper’s ordering
order = [
  'BiLSTM','CNN','CNN-MFS','Char-CNN',
  'BiLSTM-Context','CNN-Feature','CNN-Context','CNN-Context-Rep', 'CNN-Context-Rep-Improved'
]
df['Method'] = pd.Categorical(df['Method'], categories=order, ordered=True)
df = df.sort_values('Method').reset_index(drop=True)

# finally, print
print(df.to_markdown(index=False, floatfmt=".4f"))
