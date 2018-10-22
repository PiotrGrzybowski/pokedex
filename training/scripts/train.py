from subprocess import check_call
import time

PYTHON = '/home/developers/Libraries/Python/envs/gpu/bin/python3.5'

learning_rates = [0.0001]
batch_sizes = [64]

i = 0
m = len(learning_rates) * len(batch_sizes)
start_time = time.time()
for lr in learning_rates:
    for batch in batch_sizes:
        print("{}/{}".format(i, m))
        cmd = "{} train_image_classification.py --learning_rate {} --batch_size {}".format(PYTHON, lr, batch)
        check_call(cmd, shell=True)
        i += 1
elapsed_time = time.time() - start_time

print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))