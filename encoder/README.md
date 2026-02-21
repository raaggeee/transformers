## Encoder and its components
<img width="326" height="552" alt="image" src="https://github.com/user-attachments/assets/d37a789f-87fe-4d1f-9274-2972f24a77be" />


## About Encoder Layer
Encoder layer is used to understand the contextual relationship between the words.

## Process in Encoder Layer
1. Generate Input Embeddings
2. Add Input Embeddings with positional encodings
3. Now pass the positional encodings to multi head attention layer.
4. Add the output of multi head attention with residual of positional encoding. Then normalize
5. Then pass the output of (4) to Feed Forward Neural Network
6. Add the residual of (4) with the output of Feed Forward Neural Network

## Output
Encoder layer gives 2 outputs to the decoder. The Key and Value of current input embeddings.

