import re
from torch.utils.tensorboard import SummaryWriter

# Path to your saved training log
logfile = "TET_18_Sep_Tune_50e_2t.log"   # change this to your file
writer = SummaryWriter(log_dir="runs/Tune_50e_2t")

# Regex to parse each log line
pattern = re.compile(
    r"Fine-tune Epoch:\[(\d+)/\d+\]\s+"
    r"Train loss=([\d\.]+)\s+Train acc=([\d\.]+)\s+"
    r"Test acc=([\d\.]+)\s+Avg Spikes=([\d\.]+)\s+"
    r"GPU Peak Mem=([\d\.]+)GB\s+Time=([\d\.]+)s"
)

with open(logfile) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epoch = int(m.group(1))
            train_loss = float(m.group(2))
            train_acc = float(m.group(3))
            test_acc = float(m.group(4))
            avg_spikes = float(m.group(5))
            gpu_mem = float(m.group(6))
            epoch_time = float(m.group(7))

            # Log back into TensorBoard under the same names
            writer.add_scalar("FineTune/Loss/Train", train_loss, epoch)
            writer.add_scalar("FineTune/Accuracy/Train", train_acc, epoch)
            writer.add_scalar("FineTune/Accuracy/Test", test_acc, epoch)
            writer.add_scalar("FineTune/Spikes/Avg_per_inference", avg_spikes, epoch)
            writer.add_scalar("FineTune/GPU Memory/Peak (GB)", gpu_mem, epoch)
            writer.add_scalar("FineTune/Time/Epoch_time_sec", epoch_time, epoch)

writer.close()
print("Recovered run written to runs/recovered_finetune/")
