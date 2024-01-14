from utils import utils
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import mean_absolute_error
import torch


def validation(args, model, dataset, data_source):
    """Train the embedding model.
    args: Commandline arguments
    dataset: Dataset of batch size
    data_source: DataSource class
    """
    model.eval()
    # all_raw_preds, all_preds, all_labels, all_pre_preds = [], [], [], []
    pos_a, pos_b, pos_label = data_source.gen_batch(
        dataset, True)

    with torch.no_grad():
        emb_as, emb_bs = model.emb_model(pos_a), model.emb_model(pos_b)
        # labels = [[label[key] for key in ['nc', 'ec', 'in', 'ie']] for label in pos_label]
        # # labels = torch.tensor(labels).to(utils.get_device())
        # labels = torch.stack(pos_label, dim=0).to(utils.get_device())
        labels = torch.tensor(labels, dtype=torch.float32).to(utils.get_device())
        # labels = torch.tensor(pos_label).to(utils.get_device()) #원본은 float 값 이었음
        pred = model(emb_as, emb_bs)
        raw_pred = model.predict(pred)
        pre_pred = raw_pred.clone().detach()


    mae = mean_absolute_error(labels.cpu(), pre_pred.cpu())

    return mae
    '''
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
            torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
              torch.sum(labels).item() if torch.sum(labels) > 0 else
              float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    # auroc = roc_auc_score(labels, raw_pred)
    auroc = 0.01
    # avg_prec = average_precision_score(labels, raw_pred)
    avg_prec = 0.01
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    '''


if __name__ == "__main__":
    from cbir_subsg.train import main
    main(force_test=True)