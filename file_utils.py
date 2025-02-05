import os, errno

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def write_metric_files(pairs_file_name, metrics_file_name, loss_history_file_name, pairs_distance, train_loss_history, metrics_history):
    silent_remove(pairs_file_name)
    silent_remove(metrics_file_name)
    silent_remove(loss_history_file_name)
    with open(pairs_file_name, 'w') as outfile:
        outfile.write(' '.join(f"{str(i[0].item())} {str(i[1].item())}\n" for i in pairs_distance))

    with open(metrics_file_name, 'w') as outfile:
        outfile.write(' '.join(f"{i['val_loss']:.4f} {i['val']:.4f} {i['far']:.4f} {i['acc']:.4f}\n" for i in metrics_history))

    with open(loss_history_file_name, 'w') as outfile:
        outfile.write(' '.join(f"{str(i['epoch_mean'])} {' '.join(str(a) for a in i['batch_history'])}\n" for i in train_loss_history))
        