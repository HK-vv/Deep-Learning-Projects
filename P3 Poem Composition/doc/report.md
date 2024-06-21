# P3 Poem Composition

**Name**: 赵子涵
**Major**: Computer Science
**Student ID**: 2023E8013282148

### Task

In this project, we aim to generate text following the given one. Specifically, text are restricted to Chinese poem.

### Dataset

We use the dataset from the course, which contains 57580 poetry in Tang dynasty.

### Model Design

Considering the computing resource accessible currently, we design a simple model for this task.

The model consist three main part, which are, the embedding layer, the time series model GRU, and MLP layer. An illustration is shown above.

<img src="C:\Users\vv\Desktop\Archives\Master\Deep Learning\Projects\P3 Poem Composition\doc\pic\P3_model.png" alt="P3_model" style="zoom:50%;" />

The embedding layer could be implemented as a full connection layer. It tries to learn to embed the one dimension word into high dimension, which contains more information. The GRU is generally a time series model. Contrast to basic RNN, it utilizes the gate control method to settle the problem of vanishing gradient, hence long time memory is preserved. At the same time, it has a more simple structure then LSTM. The MLP layer is implemented by two stacked linear layer, which map the output of GRU to one hot coding as the result. We use cross entropy loss as criterion.

### Result

The parameter we choose are as follows

```python
BATCH_SIZE=100
LR=0.001

embed_dim=100
hidden_dim=1000
num_layer=2
mlp_dim=200
```

The figure below shows the loss descending as training epoch increase.

<img src="C:\Users\vv\Desktop\Archives\Master\Deep Learning\Projects\P3 Poem Composition\doc\pic\training_loss.png" alt="training_loss" style="zoom:50%;" />

Since there are no test set for this task, we wrote nine prompts as the starting line, and make the model compose the rest. See result as below.

```
input text: 下马饮君酒，问君何所之
output text: 下马饮君酒，问君何所之。我有一杯酒，我有一杯酒。我有一杯酒，劝君酒杯酒。我有一杯酒，劝酒
劝君酒。酒酣酒醒酒，酒酣醉如醉。


```

 