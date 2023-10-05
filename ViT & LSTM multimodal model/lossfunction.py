import torch.nn.functional as F
import torch.nn as nn
#ignore_index ******* very important
def custom_loss(y_true, y_pred, ignore_index):
    y_true = y_true.view(-1)  # Reshape to (batch_size * sequence_length)
    y_pred = y_pred.view(-1, y_pred.size(-1))  # Reshape to (batch_size * sequence_length, num_classes)

    # Calculate the raw cell outputs
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(y_pred, y_true)

    # Create the mask
    mask = (y_true != ignore_index).float()

    # Apply the mask
    loss = loss * mask

    # Return the mean of the loss
    return loss.mean()


