Dual-Stage Attention-Based Recurrent Neural Net for Time Series Prediction based off [this blog post](https://chandlerzuo.github.io/about/) by Chandler Zuo. I've extended the code to work for multi-variate time-series and added some pre-processing, but given I'm basically copying the code from his post, the copyright probably belongs to him.

The most recent branch, which uses the PyTorch JIT, is called `jit`.

There is a different [implementation by Zhenye-Na](https://github.com/Zhenye-Na/DA-RNN), but from what I can tell, it's only single-variate.
