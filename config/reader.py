import yaml
import glob
# yaml_path = r"C:\Users\82105\JINLSTM\config\trainconfig\train1.yml"
# with open(yaml_path, 'r') as f:
#     conf = yaml.load(f, Loader=yaml.FullLoader)

class Config:

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        with open(self.yaml_path, 'r') as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)
        self.pred_length = self.conf["pred_length"]
        self.past_length = self.conf["past_length"]
        self.seq_len = self.pred_length + self.past_length

        self.experiment_name = self.conf["experiment_name"]
        self.run_dir = self.conf["run_dir"]
        self.device = self.conf["device"]
        self.tgt_size = self.conf["tgt_size"]


        self.train_start_date = self.conf["train_start_date"]
        self.train_end_date = self.conf["train_end_date"]
        self.validation_start_date = self.conf["validation_start_date"]
        self.validation_end_date = self.conf["validation_end_date"]
        self.test_start_date = self.conf["test_start_date"]
        self.test_end_date = self.conf["test_end_date"]
        self.seed = self.conf["seed"]
        # --- Model configuration --------------------------------------------------------------------------
        self.model_type = self.conf["model"]
        self.dropout_rate = self.conf["dropout_rate"]
        self.d_model = self.conf["d_model"]
        self.n_heads = self.conf["n_heads"]
        self.n_encoder_layers = self.conf["n_encoder_layers"]
        self.n_decoder_layers = self.conf["n_decoder_layers"]
        self.d_ff = self.conf["d_ff"]
        self.d_head = self.conf["d_head"]
        self.embedding_mode = self.conf["embedding_mode"]
        self.dropout_lstm = self.conf["dropout_lstm"]
        self.lstm_hidden_size = self.conf["lstm_hidden_size"]
        # --- Training configuration -----------------------------------------------------------------------

        self.batch_size = self.conf["batch_size"]
        self.optimizer = self.conf["optimizer"]
        self.loss = self.conf["loss"]
        self.l2 = self.conf["l2"]
        self.epochs = self.conf["epochs"]
        self.learning_rate = self.conf["learning_rate"]
        
        # --- Dataset configuration -------------------------------------------------------------------------
        self.dynamic_features = self.conf["dynamic_features"]
        self.static_features = self.conf["static_features"]
        self.dynamic_paths = self.conf["dynamic_paths"]
        self.static_path = self.conf["static_path"]

        self.src_size = len(self.dynamic_features) + len(self.static_features)

    def get_transformer_model_config_dict(self):
        """
            (src_len, src_size, tgt_size, dropout_rate, d_model, 
                 n_heads, n_encoder_layers, n_decoder_layers, d_ff, d_head, pred_len, device, 
                 ):
        """
        return {
            "src_len": self.past_length + self.pred_length,
            "src_size": len(self.dynamic_features) + len(self.static_features),
            "tgt_size": 1,
            "dropout_rate": self.dropout_rate,

            "d_model": self.d_model,

            "n_heads": self.n_heads,

            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,

            "d_ff": self.d_ff,
            "d_head": self.d_head,
            
            "pred_len": self.pred_length,
            
            "device": self.device,
                }
    
    def get_lstm_msv_s2s_model_config_dict(self):
        """
            def __init__(self,src_len,pred_len,src_size,hidden_size,\
                 past_len,tgt_size,\
                dropout_rate):
        """
        return {
            "src_len": self.past_length + self.pred_length,
            "pred_len": self.pred_length,
            "src_size": len(self.dynamic_features) + len(self.static_features),
            "hidden_size": self.lstm_hidden_size,
            "past_len": self.past_length,
            "tgt_size": 1,
            "dropout_rate": self.dropout_rate
        }
    
    def get_loader_config_dict(self, type:str="train"):
        """
src_path:List, start_date, end_date, past_length, pred_length, \
              dynamic_features, \
              batch_size, run_dir, loader_type, \
              save=True, get_datasets=True
        """
        if type == "train":
            start_date = self.train_start_date
            end_date = self.train_end_date
        elif type == "validation":
            start_date = self.validation_start_date
            end_date = self.validation_end_date
        elif type == "test":
            start_date = self.test_start_date
            end_date = self.test_end_date
            
        return {
            "src_path": [self.dynamic_paths, self.static_path],
            "start_date": start_date,
            "end_date": end_date,
            "past_length": self.past_length,
            "pred_length": self.pred_length,
            "dynamic_features": self.dynamic_features,
            "batch_size": self.batch_size,
            "run_dir": self.run_dir,
            "loader_type": "train",
            "save": True,
            "get_datasets": True
        }
