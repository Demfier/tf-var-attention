config = dict(

    experiment='arc',  # Experiment - either qgen or dialogue or arc
    lstm_hidden_units=100,  # Number of hidden units for the LSTM
    embedding_size=300,  # Word embedding dimension
    num_layers=1,  # Number of LSTM layers

    encoder_vocab=29000,  # Vocabulary size on the encoder side: 30000 for dialogue, 40000 for qgen, 29000 for arc
    decoder_vocab=8000,  # Vocabulary size on the decoder side: 30000 for dialogue, 40000 for qgen, 8000 for arc
    encoder_num_tokens=17,  # Number of words/tokens in the input sequence: 20 for dialogue, 30 for qgen, 17 for arc
    decoder_num_tokens=5,  # Number of words/tokens in the generated sequence, 30 for qgen, 5 for arc

    dropout_keep_prob=0.8,  # Dropout keep probability
    initial_learning_rate=0.005,  # Initial learning rate
    learning_rate_decay=0.75,  # Learning rate decay
    min_learning_rate=0.00001,  # Minimum learning rate

    latent_dim=100,  # Dimension of z-latent space
    word_dropout_keep_probability=0.75,  # 1.0 - Word dropout rate for the decoder
    z_temp=1.0,  # Sampling temperature to be multiplied with the standard deviation
    attention_temp=1.0,  # Sampling temperature for the attention/context vector
    use_hmean=True,  # Whether to use N(h_mean, I) or N(0, I) as the prior
    gamma_val=0.1,  # Coefficient of the attention KL loss

    batch_size=100,  # Batch size: 128 for dialogue
    n_epochs=10,  # Number of epochs

    logs_dir='summary_logs/var-seq2seq-var-attn',  # Path to save summary information for Tensorboard
    model_checkpoint_dir='models/var-seq2seq-var-attn-',  # Path to save model checkpoints
    bleu_path='bleu/det-seq2seq-var-attn',  # Path to save model checkpoints
    w2v_dir='../w2v_models/',  # Word2Vec model directory
    data_dir='../data/',  # Directory to store data csv files

    load_checkpoint=7,  # Specify the trained model epoch/checkpoint number to be loaded for evaluation on test set, 0 means last saved checkpoint

)
