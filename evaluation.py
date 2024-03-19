import torch
import pandas as pd
from pathlib import Path
import glob
from model import transformer
from torch.utils.tensorboard import SummaryWriter
from config import reader
from utils import evalmetrices
import dataloader

def evaluation(model, test_loader,writer, run_dir ,device):
    model.eval()
    q_pred = []
    q_true = []
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            outputs = model(src)
            q_pred.append(outputs)
            q_true.append(tgt)

    q_std = test_loader.dataset.tgt.std().item()
    q_mean = test_loader.dataset.tgt.mean().item()

    q_pred = torch.cat(q_pred, dim=0)
    q_true = torch.cat(q_true, dim=0)

    q_pred = q_pred.detach().cpu().numpy() * q_std + q_mean
    q_true = q_true.detach().cpu().numpy() * q_std + q_mean

    nse = evalmetrices.nse(q_true, q_pred)
    rmse = evalmetrices.rmse(q_true, q_pred)
    pbias = evalmetrices.pbias(q_true, q_pred)
    kge = evalmetrices.kge(q_true, q_pred)

    writer.add_scalar('Test/NSE', nse)
    writer.add_scalar('Test/RMSE', rmse)
    writer.add_scalar('Test/PBIAS', pbias)
    writer.add_scalar('Test/KGE', kge)

    df = pd.DataFrame({"q_pred":q_pred[:,0], "q_true":q_true[:,0]})
    time_index = test_loader.dataset.dates[test_loader.dataset.past_length + test_loader.dataset.pred_length - 1:]
    df.index = time_index
    # make test folder
    test_dir = Path(run_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_dir / "test_result.csv")

    return nse, rmse, pbias, kge, q_pred, q_true

if __name__ == "__main__":
    yaml_path = Path(r"C:\Users\82105\JINLSTM\config\trainconfig\train1.yml")
    config = reader.Config(yaml_path=yaml_path)
    run_dir = config.run_dir
    device = config.device
    num_layers = config.num_layers
    hidden_size_lstm = config.hidden_size_lstm
    dropout_linear = config.dropout_linear
    outlinear_hidden = config.outlinear_hidden
    dropout_lstm = config.dropout_lstm
    pred_length = config.pred_length
    past_length = config.past_length
    test_start_date = config.test_start_date
    test_end_date = config.test_end_date
    rf_path = config.rf_path
    tmax_path = config.tmax_path
    tmin_path = config.tmin_path
    runoff_path = config.runoff_path
    batch_size = config.batch_size
    

    writer = SummaryWriter(run_dir)

    # load best model
    model_paths = glob.glob(str(Path(run_dir) / "models" / "*.pt"))
    # choose best model by nse
    nse_list = [float(i.split("nse")[-1][2:7]) for i in model_paths]
    best_model_path = model_paths[nse_list.index(max(nse_list))]
    model = lstm.LSTM(input_size=3, num_layers=num_layers,hidden_size=hidden_size_lstm, dropout_linear=dropout_linear, outlinear_hidden=outlinear_hidden,
                        dropout_lstm=dropout_lstm, output_size=1, batch_first=True, bidirectional=False)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    test_loader = dataloader.get_loader(rf_path=rf_path, tmax_path=tmax_path, tmin_path=tmin_path, runoff_path=runoff_path,
                                        start_date=test_start_date, end_date=test_end_date,
                                        past_length=past_length, pred_length=pred_length, batch_size=batch_size, run_dir=run_dir, loader_type="test")
    evaluation(model, test_loader, writer, run_dir, device)


    