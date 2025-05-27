import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    # Set the model to training mode (enables features like dropout)
    model.train()

    # List to store the loss values for each batch
    loss_value = []

    # Get the current learning rate from the optimizer's parameter groups
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    # Initialize a GradScaler for mixed-precision training
    scaler = GradScaler()

    # Iterate over batches of data in the training loader
    for batch_idx, data in enumerate(tqdm(loader)):
        # Move data to the specified device (e.g., GPU)
        vid = device.data_to_device(data[0])  # Input video data
        vid_lgt = device.data_to_device(data[1])  # Video length (number of frames)
        label = device.data_to_device(data[2])  # Ground truth labels (sign language)
        label_lgt = device.data_to_device(data[3])  # Length of labels (number of glosses)

        # Reset the gradients of the optimizer
        optimizer.zero_grad()

        # Use autocast for automatic mixed precision (AMP) for faster training
        with autocast():
            # Pass data through the model and get the results
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

            # Calculate the loss using the model's criterion
            loss = model.criterion_calculation(ret_dict, label, label_lgt)

        # Check if loss is NaN or infinity, and skip the batch if so
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1]) + '  frames')
            print(str(data[3]) + '  glosses')
            continue

        # Scale the loss and perform backward pass (backpropagation)
        scaler.scale(loss).backward()

        # Update the optimizer's parameters
        scaler.step(optimizer.optimizer)

        # Update the scaler to adjust for mixed precision
        scaler.update()

        # Append the loss value of this batch to the list
        loss_value.append(loss.item())

        # If the current batch index is a multiple of the log interval, print log
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0])
            )

        # Clean up memory by deleting the batch results to avoid excessive memory use
        del ret_dict
        del loss

    # Step the optimizer scheduler at the end of the epoch
    optimizer.scheduler.step()

    # Log the mean training loss for the epoch
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))

    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    # Set the model to evaluation mode (disables dropout and other training-specific features)
    model.eval()

    # Initialize lists to store results
    total_sent = []           # For storing recognized sentences
    total_info = []           # For storing file-related information (e.g., filenames)
    total_conv_sent = []      # For storing conversational sentences, if available
    loss_value = []           # For storing evaluation loss values

    # Initialize a dictionary to track statistics about the dataset (not used in this code)
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}

    # Iterate through the dataset in batches
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")  # Start timer for device-related operations

        # Move data to the appropriate device (CPU or GPU)
        vid = device.data_to_device(data[0])  # Video frames
        vid_lgt = device.data_to_device(data[1])  # Video length
        label = device.data_to_device(data[2])  # Labels (ground truth)
        label_lgt = device.data_to_device(data[3])  # Label lengths (ground truth)

        # Perform inference with the model (no gradient calculation)
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            # Calculate the loss explicitly using the model's criterion
            loss = model.criterion_calculation(ret_dict, label, label_lgt)

            # Check if loss is NaN or infinity, and skip adding it if so
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                continue

            loss_value.append(loss.item())

        # Collect file-related info and recognized sentences
        total_info += [info_dict['fileid'] for info_dict in data[-1]] # Use fileid directly from info dictionary
        total_sent += ret_dict['recognized_sents']  # Recognized sentences
        total_conv_sent += ret_dict['conv_sents']  # Conversational sentences (if any)

    try:
        # Set the evaluation tool flag based on the user input
        python_eval = True if evaluate_tool == "python" else False

        # Write the recognized and conversational sentences to .ctm files
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info, total_conv_sent)

        # Perform evaluation for the conversational sentences
        conv_ret = evaluate(
            prefix=work_dir,  # Working directory
            mode=mode,  # Mode (e.g., train, test)
            output_file="output-hypothesis-{}-conv.ctm".format(mode),  # Output file for conversational sentences
            evaluate_dir=cfg.dataset_info['evaluation_dir'],  # Directory for evaluation files
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],  # Prefix for evaluation files
            output_dir="epoch_{}_result/".format(epoch),  # Output directory for this epoch's results
            python_evaluate=python_eval,  # Use Python evaluation if True
        )

        # Perform evaluation for the regular recognized sentences
        lstm_ret = evaluate(
            prefix=work_dir,  # Working directory
            mode=mode,  # Mode (e.g., train, test)
            output_file="output-hypothesis-{}.ctm".format(mode),  # Output file for recognized sentences
            evaluate_dir=cfg.dataset_info['evaluation_dir'],  # Directory for evaluation files
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],  # Prefix for evaluation files
            output_dir="epoch_{}_result/".format(epoch),  # Output directory for this epoch's results
            python_evaluate=python_eval,  # Use Python evaluation if True
            triplet=True,  # Option to evaluate with triplet
        )
    except:
        # If an error occurs during evaluation, print the error and set a high error rate
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        # Ensure no variables are holding large memory
        pass

    # Log the mean evaluation loss if we have valid loss values
    if loss_value:
        recoder.print_log('\tMean evaluation loss: {:.10f}.'.format(np.mean(loss_value)))
    else:
        recoder.print_log('\tMean evaluation loss: No valid loss values collected.')

    # Cleanup - delete temporary variables to free up memory
    del conv_ret
    del total_sent
    del total_info
    del total_conv_sent
    del vid
    del vid_lgt
    del label
    del label_lgt
    if loss_value:
        del loss_value

    # Log the result for this epoch (e.g., accuracy or error rate)
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")

    # Return the evaluation result (e.g., error rate or performance metric)
    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
